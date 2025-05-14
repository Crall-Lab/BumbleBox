#!/usr/bin/env python

import pandas as pd
import numpy as np
import math

#August added back into data_cleaning.py on May 14th, 2025
#Remove tag detections that jump over a threshold number of pixels from one frame to the very next frame
#NOTE: this is not yet robust and its effect needs to be tested - the following scenario presents an issue for the current code:
#Tag 20 is detected in frame 5, 6, and 7
#The first detection, upon review, is the false detection - it is across the nest from where bee 20 actually is.
#The next detection jumps back to the correct position. But which detection actually gets dropped?
#My current understanding upon review and without testing is that the true tag detection would be dropped, which is incorrect.
#I think the third detection would be fine in this case.
#BUT WAIT: in the scenario where bee 20 is tracked in 5,6,7: if detection 6 is the false detection, the diff row between 6 and 7
#would also be flagged, and would detection 7 be removed? Needs testing!
def remove_jumps(interpolated_df):

    unique_ids = interpolated_df["ID"].unique() 
    for bee_id in unique_ids:
        bee_df = interpolated_df[ interpolated_df['ID'] == bee_id]
        bee_sub_df = bee_df.loc[:, ['frame', 'centroidX', 'centroidY']]
        diff_df = bee_sub_df.diff()

    for index, row in diff_df.iterrows():
        #exclude rows that jump more than 500 pixels over a single frame
        if row['frame'] == 1 and math.sqrt(row['centroidX']**2 + row['centroidY']**2) > 500:
            interpolated_df.drop(index, axis=0, inplace=True)
    
    return interpolated_df


#check for multiples of the same tag in each frame
def return_duplicate_bees(df):

    df.drop_duplicates(inplace=True) #drops completely duplicate rows before we look for duplicate tags 
    
    try:
        df['duplicate_in_frame'] = df.duplicated(['video path', 'bee ID', 'frame number', 'col #'], keep = False) #update df to include column that tracks duplicate tag detections based on these columns
        if True in df['duplicate_in_frame'].values:#there are any trues in df.duplicated, return the updated df, else return print(no duplicates!)
            print('Yes, there are duplicate tag readings in the same frame! Theyve been marked True in the duplicates column.')
            return df, 0
            # return None, print('No duplicates in this dataframe!')
        else:
            print('There arent any duplicates in this dataframe!')
            return df, 1
    except:
        print('''An error occured in the function return_duplicate_bees() while trying to create a new column to track whether any tags are duplicates.
                 If the error raised looks like this: 'KeyError: Index(['col #'], dtype='object')' it was raised likely because you havent run the function to add colony numbers to the dataframe''')
        return 1, 1


#Helper function that runs inside of the drop_duplicates_clean function (below)
def resolve_duplicate_by_proximity(duplicate_rows, nearest_row):
    """
    Resolve among a group of duplicate tag detections in the same frame by selecting the one
    closest in space to a known position from a nearby frame.

    Parameters:
    duplicate_rows (DataFrame): Rows representing duplicate tag detections in the same frame.
    nearest_row (Series): The nearest known position of the same bee in another frame.

    Returns:
    tuple: (index, row) of the detection that is closest in space to the reference.
    """
    closest_idx = None
    closest_row = None
    min_distance = float('inf')  # Start with an arbitrarily large distance

    # Loop over each duplicate candidate in the current frame
    for idx, row in duplicate_rows.iterrows():
        # Compute Euclidean distance between this candidate and the known nearby position
        dx = row['centroidX'] - nearest_row['centroidX']
        dy = row['centroidY'] - nearest_row['centroidY']
        dist = math.hypot(dx, dy)

        # Keep track of the one closest to the known nearby point
        if dist < min_distance:
            closest_idx = idx
            closest_row = row
            min_distance = dist

    return closest_idx, closest_row


def drop_duplicates_clean(df, return_val, drop_unresolvable=True):
    """
    Resolves duplicate detections of the same bee ID within a single frame based on spatial proximity
    to known positions in neighboring frames. Keeps the most plausible tag and optionally flags or
    drops unresolved duplicates.

    Parameters:
    df (DataFrame): The tracking data with potential duplicates.
    return_val (int): Returned value from return_duplicate_bees(), 0 if duplicates exist.
    drop_unresolvable (bool): Whether to drop duplicate rows that couldn't be confidently resolved.

    Returns:
    DataFrame: A cleaned DataFrame with resolved duplicates removed and the best candidate retained.
    """
    df = df.copy()  # Work on a copy to avoid modifying original data
    df.drop_duplicates(inplace=True)  # Drop any fully duplicated rows

    # Sanity check to make sure duplicates have already been identified
    if 'duplicate' not in df.columns:
        print("Hey, have you run the return_duplicate_bees() function? I'm not seeing a duplicate column in this dataframe.")
        return df

    # Add helper columns to track which rows were part of a duplicate set and what happened to them
    df['og_duplicate'] = False
    df['unresolvable_duplicate'] = False

    if return_val == 0:
        # Subset to rows marked as duplicates
        duplicates = df[df['duplicate'] == True]

        # Create a table of unique (video, colony, bee ID, frame) combinations with duplicates
        dupe_keys = duplicates[['video path', 'col #', 'bee ID', 'frame number']].drop_duplicates()

        # Loop through each unique duplicated instance
        for _, row in dupe_keys.iterrows():
            vid = row['video path']
            col = row['col #']
            bee = row['bee ID']
            frame = row['frame number']

            # Get all the duplicated rows for this (video, colony, bee ID, frame)
            specific_duplicates = duplicates[
                (duplicates['video path'] == vid) &
                (duplicates['col #'] == col) &
                (duplicates['bee ID'] == bee) &
                (duplicates['frame number'] == frame)
            ]

            # Find other positions of the same bee in other frames (same video)
            nearest_position_v1 = df[
                (df['video path'] == vid) &
                (df['col #'] == col) &
                (df['bee ID'] == bee) &
                (df['frame number'] != frame)
            ]

            # If no known positions exist in other frames, we can't resolve this duplicate
            if nearest_position_v1.empty:
                df.loc[specific_duplicates.index, 'unresolvable_duplicate'] = True
                continue

            # Find the position in another frame that is temporally closest to the duplicate frame
            nearest_position_v2 = nearest_position_v1.iloc[
                (nearest_position_v1['frame number'] - frame).abs().argsort()[:1]
            ]

            # Skip resolution if the nearest frame is too far away to trust
            if nearest_position_v2.empty or abs(nearest_position_v2['frame number'].values[0] - frame) > 16:
                df.loc[specific_duplicates.index, 'unresolvable_duplicate'] = True
                continue

            # Call modular function to find best candidate detection among the duplicates
            idx_to_keep, tag_to_keep = resolve_duplicate_by_proximity(
                specific_duplicates, nearest_position_v2.iloc[0]
            )

            # Drop all other candidates in the same frame with same ID
            drop_idxs = df[
                (df['video path'] == vid) &
                (df['col #'] == col) &
                (df['bee ID'] == bee) &
                (df['frame number'] == frame) &
                ((df['centroidX'] != tag_to_keep['centroidX']) |
                 (df['centroidY'] != tag_to_keep['centroidY']))
            ].index
            df.drop(index=drop_idxs, inplace=True)

            # Mark the kept tag as a resolved original duplicate
            good_idx = df[
                (df['video path'] == vid) &
                (df['col #'] == col) &
                (df['bee ID'] == bee) &
                (df['frame number'] == frame) &
                (df['centroidX'] == tag_to_keep['centroidX']) &
                (df['centroidY'] == tag_to_keep['centroidY'])
            ].index
            df.loc[good_idx, 'duplicate'] = False
            df.loc[good_idx, 'og_duplicate'] = True

        # Optionally remove any unresolved duplicates
        if drop_unresolvable:
            df.drop(df[df['duplicate'] == True].index, inplace=True)

    elif return_val == 1 and isinstance(df, pd.DataFrame):
        # If no duplicates existed, still ensure tracking columns exist
        df['og_duplicate'] = False
        df['unresolvable_duplicate'] = False

    return df




# Updated function to interpolate missing frames only if the gap between them is less than or equal to max_frame_gap
def interpolate(df, max_seconds_gap, actual_frames_per_second):

    max_frame_gap = int(max_seconds_gap * actual_frames_per_second)
    # Ensure the data is sorted by frame
    df.sort_values(by=['ID', 'frame'], inplace=True)
    
    # Group by bee ID
    grouped = df.groupby('ID')
    
    # Placeholder for the new DataFrame with interpolated values
    interpolated_dfs = []
    
    for bee_id, group in grouped:
        # Ensure group is sorted by frame
        group = group.sort_values('frame')
        
        # Calculate the frame difference between consecutive rows
        group['frame_diff'] = group['frame'].diff().fillna(0).astype(int)
        
        # Placeholder list to store the interpolated results for this group
        interpolated_rows = []
        
        # Iterate over the rows of the group
        for i in range(len(group)):
            row = group.iloc[i]
            interpolated_rows.append(row)
            
            # Get the next row if it exists
            if i + 1 < len(group):
                next_row = group.iloc[i + 1]
                # If the frame difference is less than or equal to the max frame gap, interpolate
                if 0 < next_row['frame_diff'] <= max_frame_gap:
                    # Number of frames to interpolate
                    num_frames_to_interpolate = next_row['frame_diff'] - 1
                    # Generate interpolated frames
                    for n in range(1, num_frames_to_interpolate + 1):
                        interp_row = row.copy()
                        ratio = n / next_row['frame_diff']
                        # Interpolate numeric columns
                        for col in ['centroidX', 'centroidY', 'frontX', 'frontY']:
                            interp_row[col] = row[col] + (next_row[col] - row[col]) * ratio
                        # Calculate the correct frame number for the interpolated frame
                        interp_row['frame'] = row['frame'] + n
                        interpolated_rows.append(interp_row)
        
        # Create a DataFrame from the list of rows
        interpolated_group = pd.DataFrame(interpolated_rows)
        
        # Drop the frame_diff column as it is no longer needed
        interpolated_group.drop(columns=['frame_diff'], inplace=True)
        
        # Append the group to the list of DataFrames
        interpolated_dfs.append(interpolated_group)
    
    # Concatenate all the interpolated groups into a single DataFrame
    interpolated_df = pd.concat(interpolated_dfs, ignore_index=True)
    
    # Sorting the DataFrame by 'ID' and 'frame' for better readability
    interpolated_df.sort_values(by=['ID', 'frame'], inplace=True)
    
    return interpolated_df

#Calculate the angle between the center of the ArUco tag and the top of the tag (make sure it points towards the head!), 
#which are automatically calculated and stored in the raw.csv
#Store it in a column as radians and in a column as degrees
#Return the updated matrix with these two new columns
def compute_heading_angle(df):
    """
    Computes the heading angle (in radians and degrees) for each bee, based on
    the vector from centroid to front. Adds two columns:
      - 'heading_angle': angle in radians, range [-π, π]
      - 'heading_angle_deg': angle in degrees, range [0, 360)
    """
    dx = df['frontX'] - df['centroidX']
    dy = df['frontY'] - df['centroidY']
    
    # Radians: [-pi, pi]
    df['heading_angle'] = np.arctan2(dy, dx)
    
    # Degrees: [0, 360)
    df['heading_angle_deg'] = np.degrees(df['heading_angle']) % 360
    
    return df
	

def main():
	print("I am a python module, I am not run by myself. I just contain functions that are imported by other scripts to use!")
	
if __name__ == '__main__':
	
	main()
