# track_prerecorded_videos.py

"""
This script processes prerecorded videos (e.g., MP4 or MJPEG) by detecting ArUco tags and, optionally,
calculating behavioral metrics such as speed, activity, center distance, pairwise distances, and contact matrices.

Typical usage:
    python3 track_prerecorded_videos.py \
        -v /path/to/videos \
        -d 4X4_50 \
        -b custom \
        -f 5 \
        --metrics

Dependencies:
    - OpenCV (with aruco module)
    - pandas
    - behavioral_metrics.py
    - tag_tracking_utils.py (must be in the same directory or in PYTHONPATH)

Note:
    Several parameters (e.g., moving threshold, contact distance) are currently hard-coded
    and should be moved to a centralized config system, such as setup.py or a future GUI.
"""

import os
import argparse
import pandas as pd
import time
from datetime import datetime
import behavioral_metrics
from tag_tracking_utils import trackTagsFromVid, load_actual_fps
import re

# TODO: Remove these hard-coded constants and place them in setup.py or pass as command-line arguments
MOVING_THRESHOLD = 3.16  # pixels/frame threshold to determine movement vs noise
SPEED_CUTOFF_SECONDS = 4  # max gap to interpolate speed values
PIXEL_CONTACT_DISTANCE = 206.1  # contact distance in pixels (depends on setup)

#need to get rid of now, colony_number, cause these should be found within the function and video and fed to 
#tracking script 
def main(video_folder, dictionary, box_type, fallback_fps, run_metrics):
    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.endswith(".mp4") or file.endswith(".mjpeg"):
                filepath = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                print(f"\nProcessing video: {filename}")

                #filename structure: bumblebox-XX_yyyy-mm-dd_HH_MM_SS
                # Find the time from the filename, create a datetime object called now with the video time
                delimiters = r"[-_]+"
                result = re.split(delimiters, filename)
                #result is list containing: [ bumblebox, XX, yyyy, mm, dd, HH, MM, SS ]
                dt = result[2:]
                dt = [ int(x) for x in dt ]
                now = datetime.datetime(year=dt[0], month=dt[1], day=dt[2], hour=dt[3], minute=dt[4], seconds=dt[5])
                colony_number = result[1]

                df, df2, frame_num = trackTagsFromVid(
                    filepath=filepath,
                    output_dir=root,
                    filename=filename,
                    tag_dictionary=dictionary,
                    box_type=box_type,
                    now=now,
                    colony_number=colony_number
                )

                fps = load_actual_fps(filepath) or fallback_fps

                if run_metrics and not df.empty:
                    print("Running behavioral metrics...")
                    df = behavioral_metrics.compute_speed(df, fps, SPEED_CUTOFF_SECONDS, MOVING_THRESHOLD, root, filename)
                    df = behavioral_metrics.compute_activity(df, fps, SPEED_CUTOFF_SECONDS, MOVING_THRESHOLD, root, filename)
                    df = behavioral_metrics.compute_social_center_distance(df, root, filename)
                    pw_df = behavioral_metrics.pairwise_distance(df, root, filename)
                    contact_df = behavioral_metrics.contact_matrix(pw_df, root, PIXEL_CONTACT_DISTANCE, filename)
                    _ = behavioral_metrics.compute_video_averages(df, root, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run tag tracking and behavioral metrics on prerecorded videos.")
    parser.add_argument('-v', '--video_folder', type=str, required=True, help='Path to the folder containing videos')
    parser.add_argument('-d', '--dictionary', type=str, default='4X4_50', help='ArUco tag dictionary (e.g., 4X4_50)')
    parser.add_argument('-b', '--box_type', type=str, default='custom', choices=['custom', 'koppert'], help='Box type for preset tracking parameters')
    parser.add_argument('-f', '--fps', type=int, default=5, help='Fallback FPS if actual FPS metadata is missing')
    parser.add_argument('--metrics', action='store_true', help='If set, will calculate behavioral metrics')

    args = parser.parse_args()
    
    print(f"Video folder: {args.video_folder}")
    print(f"Aruco dictionary: {args.dictionary}")
    print(f"Box type for Aruco parameters: {args.box_type}")
    print(f"Current datetime: {}")
    print(f"Fallback FPS for metrics: {args.fps}")
    print(f"Behavioral metrics being run: {args.metrics}")
    print("Starting processing...")
    #video_folder, dictionary, box_type, now, colony_number, fallback_fps, run_metrics
    main(
        video_folder=args.video_folder,
        dictionary=args.dictionary,
        box_type=args.box_type,
        now=now,
        colony_number=colony_number,
        fallback_fps=args.fps,
        run_metrics=args.metrics
    )

    print("Processing complete.")
