#!/usr/bin/env python

import pandas as pd
import numpy as np
import math

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

def main():
	print("I am a python module, I am not run by myself. I just contain functions that are imported by other scripts to use!")
	
if __name__ == '__main__':
	
	main()
