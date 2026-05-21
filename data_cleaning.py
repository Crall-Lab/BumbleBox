#!/usr/bin/env python

import pandas as pd
import numpy as np
import math
import os

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
def remove_jumps_old(interpolated_df):

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


def remove_jumps(df, log_path=None, jump_thresh=500, jump_threshold_pixels=None):
    """
    Flags suspicious jumps in ArUco tag tracking data and logs jump rows + neighbors.

    Args:
        df (pd.DataFrame): tracking data with columns ['ID', 'frame', 'centroidX', 'centroidY']
        jump_thresh (float): pixel threshold for detecting jumps
        log_path (str): path to append the jump log CSV file
        video_id (str): identifier for the current video

    Returns:
        pd.DataFrame: DataFrame with 'flagged_as_jump' column added
    """

    if df.empty:
        return df

    if jump_threshold_pixels is not None:
        jump_thresh = float(jump_threshold_pixels)

    cleaned_df = df.copy()
    cleaned_df['flagged_as_jump'] = False

    if log_path is None:
        if 'colony number' in cleaned_df.columns and not cleaned_df.empty:
            colony_number = cleaned_df.iloc[0]['colony number']
        else:
            colony_number = "unknown"
        log_dir = "./jump_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = f"./jump_logs/bumblebox-{colony_number}_jump_log.csv"

    log_entries = []

    for bee_id in cleaned_df['ID'].unique():
        bee_df = cleaned_df[cleaned_df['ID'] == bee_id].sort_values('frame')
        positions = bee_df[['centroidX', 'centroidY']].values
        frames = bee_df['frame'].values
        indices = bee_df.index.values

        for i in range(1, len(positions) - 1):
            frame_prev = frames[i - 1]
            frame_curr = frames[i]
            frame_next = frames[i + 1]

            if frame_curr - frame_prev == 1 and frame_next - frame_curr == 1:
                prev = positions[i - 1]
                curr = positions[i]
                next = positions[i + 1]

                dist_prev = math.dist(prev, curr)
                dist_next = math.dist(curr, next)

                if dist_prev > jump_thresh and dist_next > jump_thresh:
                    jump_index = indices[i]
                    prev_index = indices[i - 1]
                    next_index = indices[i + 1]

                    cleaned_df.loc[jump_index, 'flagged_as_jump'] = True

                    # Add log entries
                    for idx, label in zip([prev_index, jump_index, next_index], ['neighbor', 'jump', 'neighbor']):
                        row = cleaned_df.loc[idx].copy()
                        #row['filename'] = video_id
                        #row['ID'] = bee_id
                        #row['frame'] = frames[idx]
                        
                        #row['centroidX'] = curr[0]
                        #row['centroidY'] = curr[1]
                        #row['label'] = label
                        row = row[['filename', 'ID', 'frame', 'centroidX', 'centroidY']]
                        row['label'] = label
                        log_entries.append(row)

    # Append to CSV log
    if log_entries:
        log_df = pd.DataFrame(log_entries, columns=['filename', 'ID', 'frame', 'centroidX', 'centroidY', 'label'])

        write_header = not os.path.exists(log_path)
        log_df.to_csv(log_path, mode='a', header=write_header, index=False)

    return cleaned_df

def summarize_jump_log(log_path):
    """
    Summarize total jump detections per bee across all videos.

    Args:
        log_path (str): path to the CSV log file

    Prints:
        Total jump counts per bee ID and optional per video.
    """
    if not os.path.exists(log_path):
        print("No log file found.")
        return
    try:
        log_df = pd.read_csv(log_path)
    except pd.errors.EmptyDataError:
        print("Log file is empty or not formatted correctly.")
        return
    
    if log_df.empty:
        print("Log file is empty.")
        return

    # Ensure no duplicate entries
    log_df.drop_duplicates(inplace=True)

    summary = (
        log_df[log_df['label'] == 'jump']
        .groupby('ID')
        .size()
        .reset_index(name='n_jumps')
        .sort_values('n_jumps', ascending=False)
    )
    summary2 = (
        log_df[log_df['label'] == 'jump']
        .groupby(['filename'])
        .size()
        .reset_index(name='n_jumps_per_video')  # Average jumps per video
        .sort_values('n_jumps_per_video', ascending=False)
        #.mean(axis=1, numeric_only=True)  # Calculate the mean
        #.round(2)
    )
    summary1 = summary2['n_jumps_per_video'].mean().round(2)  # Calculate the mean of jumps per video

    print("🐝 Jump Summary by Bee ID:")
    print(summary.to_string(index=False))
    print("\nAverage Jumps per Video:")
    print(summary1, "\n")
    print(summary2.to_string(index=False))


#check for multiples of the same tag in each frame
def return_duplicate_bees(df, duplicate_log_path=None):

    df.drop_duplicates(inplace=True, keep='first') #drops second row of two completely duplicate rows before we look for duplicate tags 
    
    if duplicate_log_path is None:
        colony_number = df.loc[0,'colony number']
        duplicate_log_dir = "./duplicate_logs"
        if not os.path.exists(duplicate_log_dir):
            os.makedirs(duplicate_log_dir)
        duplicate_log_path = f"./duplicate_logs/bumblebox-{colony_number}_duplicate_log.csv"
        write_header = not os.path.exists(duplicate_log_path)
    
    try:
        df['duplicate'] = df.duplicated(['filename', 'ID', 'frame'], keep = False) #update df to include column that tracks duplicate tag detections based on these columns
        if True in df['duplicate'].values:#there are any trues in df.duplicated, return the updated df, else return print(no duplicates!)
            #print('Yes, there are duplicate tag readings in the same frame! Theyve been marked True in the duplicates column.')
            duplicate_df = df[df['duplicate'] == True]  # Subset to only the duplicate rows
            duplicate_df = duplicate_df[['filename', 'ID', 'frame', 'centroidX', 'centroidY', 'duplicate']]  # Select relevant columns for the log
            write_header = not os.path.exists(duplicate_log_path)
            duplicate_df.to_csv(duplicate_log_path, mode='a', header=write_header, index=False)  # Save the duplicates log
            return df, 0
            # return None, print('No duplicates in this dataframe!')
        else:
            #print('There arent any duplicates in this dataframe!')
            return df, 1
    except Exception as e:
        print('''An error occured in the function return_duplicate_bees() while trying to create a new column to track whether any tags are duplicates.''')
        print(e)
        return 1, 1

def summarize_duplicate_log(duplicate_path):
    """
    Summarize total jump detections per bee across all videos.

    Args:
        log_path (str): path to the CSV log file

    Prints:
        Total jump counts per bee ID and optional per video.
    """
    if not os.path.exists(duplicate_path):
        print("No log file found.")
        return
    try:
        log_df = pd.read_csv(duplicate_path)
    except pd.errors.EmptyDataError:
        print("Log file is empty or not formatted correctly.")
        return
    
    if log_df.empty:
        print("Log file is empty.")
        return

    # Ensure no fully duplicate entries - not the same as dropping the duplicates we're interested in looking for, which will have different centroidX/Y values and thus wont be dropped
    log_df.drop_duplicates(inplace=True)

    summary = (
        log_df[log_df['duplicate'] == True]
        .groupby('ID')
        .size()
        .reset_index(name='n_duplicates')
        .sort_values('n_duplicates', ascending=False)
    )
    summary1 = (
        log_df[log_df['duplicate'] == True]
        .groupby(['filename'])
        .size()
        .reset_index(name='n_jumps_per_video')
        .mean()  # Average jumps per video
        .round(2)
    )

    print("🐝 Duplicate Summary by Bee ID:")
    print(summary.to_string(index=False))
    print("\nAverage Duplicates per Video:")
    print(summary1.to_string(index=False))


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
        #print("filename:", row['filename'], 'ID: ', row['ID'], "frame: ", row['frame'], "xy: ", (row['centroidX'], row['centroidY']), "nearest xy: ", (nearest_row['centroidX'], nearest_row['centroidY']))
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


def mark_or_drop_duplicates(df, return_val, mark_duplicates=True, drop_unresolvable=True):
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
    df['unresolvable_duplicate'] = False

    if return_val == 0:
        # Subset to rows marked as duplicates
        duplicates = df[df['duplicate'] == True]

        # Create a table of unique (video, colony, bee ID, frame) combinations with duplicates
        dupe_keys = duplicates[['filename', 'ID', 'frame']].drop_duplicates()

        # Loop through each unique duplicated instance
        for _, row in dupe_keys.iterrows():
            vid = row['filename']
            bee = row['ID']
            frame = row['frame']

            # Get all the duplicated rows for this (video, colony, bee ID, frame)
            specific_duplicates = duplicates[
                (duplicates['filename'] == vid) &
                (duplicates['ID'] == bee) &
                (duplicates['frame'] == frame)
            ]

            # Find other positions of the same bee in other frames (same video)
            nearest_position_v1 = df[
                (df['filename'] == vid) &
                (df['ID'] == bee) &
                (df['frame'] != frame)
            ]

            # If no known positions exist in other frames, we can't resolve this duplicate
            if nearest_position_v1.empty:
                df.loc[specific_duplicates.index, 'unresolvable_duplicate'] = True
                continue

            # Find the position in another frame that is temporally closest to the duplicate frame
            nearest_position_v2 = nearest_position_v1.iloc[
                (nearest_position_v1['frame'] - frame).abs().argsort()[:1]
            ]

            # Skip resolution if the nearest frame is too far away to trust
            if nearest_position_v2.empty or abs(nearest_position_v2['frame'].values[0] - frame) > 16:
                df.loc[specific_duplicates.index, 'unresolvable_duplicate'] = True
                continue

            # Call modular function to find best candidate detection among the duplicates
            idx_to_keep, tag_to_keep = resolve_duplicate_by_proximity(
                specific_duplicates, nearest_position_v2.iloc[0]
            )

            if not mark_duplicates:
                # If not marking duplicates, just keep the best candidate and drop others
                # Notice that both candidates are marked as duplicates right now, so we don't need to mark these again
                # Drop all other candidates in the same frame with same ID
                drop_idxs = df[
                    (df['filename'] == vid) &
                    (df['ID'] == bee) &
                    (df['frame'] == frame) &
                    ((df['centroidX'] != tag_to_keep['centroidX']) |
                    (df['centroidY'] != tag_to_keep['centroidY']))
                ].index
                df.drop(index=drop_idxs, inplace=True)
            
            # Mark the kept tag as a resolved original duplicate
            '''
            good_idx = df[
                (df['filename'] == vid) &
                (df['ID'] == bee) &
                (df['frame'] == frame) &
                (df['centroidX'] == tag_to_keep['centroidX']) &
                (df['centroidY'] == tag_to_keep['centroidY'])
            ].index
            '''
            df.loc[idx_to_keep, 'duplicate'] = False
            
        # Optionally remove any unresolved duplicates
        if drop_unresolvable:
            df.drop(df[df['unresolvable_duplicate'] == True].index, inplace=True)

    elif return_val == 1 and isinstance(df, pd.DataFrame):
        # If no duplicates existed, still ensure tracking columns exist
        df['unresolvable_duplicate'] = False

    return df




# Updated function to interpolate missing frames only if the gap between them is less than or equal to max_frame_gap
def interpolate(df, max_seconds_gap, actual_frames_per_second):

    if df.empty:
        print("The DataFrame is now empty. No interpolation will be performed.")
        return pd.DataFrame()  # Return an empty DataFrame if input is empty
    
    df["interpolated"] = False  # Add a column to track if a row is interpolated

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
        if sum(group['frame_diff']) == 0 and len(grouped) > 1:
            #print(f"No frame differences found for bee ID {bee_id}. Skipping interpolation for this group.")
            continue
        elif sum(group['frame_diff']) == 0 and len(grouped) == 1:
            #print(f"Only one group, with no difference between frames. No interpolation needed.")
            return df
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
                            interp_row[col] = round((row[col] + (next_row[col] - row[col]) * ratio),2)
                        # Calculate the correct frame number for the interpolated frame
                        interp_row['frame'] = row['frame'] + n
                        # Flag this row as interpolated
                        interp_row['interpolated'] = True
                        # Append the interpolated row to the list
                        interpolated_rows.append(interp_row)
        
        # Create a DataFrame from the list of rows
        interpolated_group = pd.DataFrame(interpolated_rows)
        
        # Drop the frame_diff column as it is no longer needed
        interpolated_group.drop(columns=['frame_diff'], inplace=True)
        
        # Append the group to the list of DataFrames
        interpolated_dfs.append(interpolated_group)
    
    if not interpolated_dfs:
        return df

    try:
        # Concatenate all the interpolated groups into a single DataFrame
        interpolated_df = pd.concat(interpolated_dfs, ignore_index=True)
    except Exception as e:
        print(e)
        print("Error concatenating interpolated DataFrames. Maybe there's only one group?")
        if len(grouped) == 1:
            print("Only one group found, returning the single interpolated group.")
            interpolated_df = interpolated_group
        else:
            print("Hit error in attempt to interpolate. Returning the original DataFrame without interpolation.")
            return df
            
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
    df['heading_angle'] = round((np.arctan2(dy, dx)),3)
    
    # Degrees: [0, 360)
    df['heading_angle_deg'] = round((np.degrees(df['heading_angle']) % 360),3)
    
    return df
	

def main():
	print("I am a python module, I am not run by myself. I just contain functions that are imported by other scripts to use!")
	
if __name__ == '__main__':
	
	main()
