'''This is an updated script that processes MP4 videos for ArUco tracking.
It's meant to be used to retrack videos with new ArUco parameters, 
or to process videos as a whole after an experiment if they werent tracked during it. 

YOU NEED TO MANUALLY ENTER YOUR NEW ARUCO DICTIONARY AND PARAMETERS IN THE SCRIPT
BY SCROLLING DOWN TO THE PROCESS VIDEO FUNCTION! 
THE CURRENT DEFAULTS WERE BEST FOR THE NEW BUMBLEBOX RIG, which is not the BumbleBox, as of Summer 2024.
These are:

# Set up ArUco detector with optimal parameters.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.02
    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 29
    parameters.adaptiveThreshWinSizeStep = 3
    parameters.polygonalApproxAccuracyRate = 0.08

Info about the script!
It allows for parallel processing of video files using multiprocessing.Pool.
You can choose a specific subfolder to process, and can set the number of CPU cores to use for parallel processing.
The script also supports filtering video files based on a start date or start file name.
The script uses the ciso8601 library for faster date parsing.
The script prints timing information for key steps if the benchmark mode is enabled.
The script uses the tqdm library to display progress bars for parallel processing.
The script uses argparse for command-line options. These are:

--volume: Path to the removable volume (e.g., /media/username/volume).
--folder: Name of the subfolder to process (e.g., bumblebox-11).
--cores: Number of CPU cores to use for processing.
--batch-size: Number of videos to process in each batch when parallel processing.
--start_date: Process only video files from subfolders with dates (YYYY-MM-DD) >= this date.
--start_file: Process only video files with basenames >= this filename.
--limit (-l): Limit the number of files to process (for testing purposes).
--benchmark: Enable targeted benchmarks and print timing information.
--extension: Extension for the output CSV files (default is "_raw.csv").
--pool-mapping: Mapping method for Pool processing (map or imap_unordered).

A simple example command to run the script (running it from the folder containing the script, and inside a virtual environment with all the required libraries installed):
python3 track-tags-to-csv.0.10.py --volume /media/username/volume --folder bumblebox-11 --cores 10
This command will process v print(max_frame_gap)ideos from the chosen volume in the "bumblebox-11" subfolder using 10 CPU cores. The output csv files will have the default "_raw.csv" extension.

A more complicated example command to run the script (running it from the folder containing the script, and inside a virtual environment with all the required libraries installed):
python3 track-tags-to-csv.0.10.py --volume /media/username/volume --folder bumblebox-11 --cores 10 --batch-size 5 --start_date 2024-06-25 --benchmark --extension _newtracks.csv --pool-mapping imap_unordered
This command will process videos from the chosen volume in the "bumblebox-11" subfolder with a start date of 2024-06-25, using 10 CPU cores and a batch size of 5 for parallel processing. 
It will print timing information for key steps and save the output CSV files with the "_newtracks.csv" extension.
'''

#!/usr/bin/env python3

# Set profiling flags:
ENABLE_BENCHMARK = True

import cv2
import numpy as np
import csv
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import warnings
import glob  # still import glob for now (if needed elsewhere)
from tqdm import tqdm  # progress bar
from multiprocessing import Pool
import ciso8601  # fast date parsing
import time  # for benchmarking


# Suppress warnings for clarity
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def list_mp4_files(root):
    """
    Recursively list all .mp4 files under the root directory using os.scandir().
    Returns a list of file paths.
    """
    files = []
    # Use a list as a stack for directories to scan.
    dirs = [root]
    while dirs:
        current_dir = dirs.pop()
        try:
            with os.scandir(current_dir) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(entry.path)
                    elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith('.mp4'):
                        files.append(entry.path)
        except Exception as e:
            print(f"Error scanning {current_dir}: {e}")
    return files

def extract_datetime_from_filename(filename):
    """
    Given a filename like 'bumblebox-13_2024-06-25_22_30_03.mp4',
    extract and return a datetime object using ciso8601 for fast parsing.
    Assumes the datetime part is in the format: YYYY-MM-DD_HH_MM_SS.
    """
    if filename[0] == '.':
        return None
    filename_no_ext, _ = os.path.splitext(filename)
    #print(filename_no_ext)
    parts = filename_no_ext.split('_')
    if len(parts) < 5:
        return None
    date_part = parts[1]  # e.g., "2024-06-25"
    #print(date_part)
    time_parts = parts[2:5]  # e.g., ["22", "30", "03"]
    #print(time_parts[0], time_parts[1], time_parts[2])
    # Convert to ISO8601: "YYYY-MM-DDTHH:MM:SS"
    dt_iso = date_part + "T" + ":".join(time_parts)
    return ciso8601.parse_datetime(dt_iso)

def process_video(video_path):
    """
    Process a single video file, tracking ArUco markers and writing results to a CSV.
    The CSV is saved in the same directory with a '_newtracks.csv' suffix.
    If benchmark mode is enabled, timing information for key steps is printed.
    """
    cv2.setNumThreads(1)  # Limit to a single thread per process

    base_filename = os.path.basename(video_path)
    filename_no_ext, _ = os.path.splitext(base_filename)
    csv_filename = os.path.join(os.path.dirname(video_path), filename_no_ext + out_file_extension)

    # Skip processing if the CSV file already exists.
    if os.path.exists(csv_filename):
        #print(f"CSV file {csv_filename} already exists. Skipping {video_path}.")
        return
    
    # Parse filename for metadata.
    parts = filename_no_ext.split('_')
    first_part = parts[0]
    try:
        colony_number = first_part.split('-')[1]
    except IndexError:
        colony_number = ""
    datetime_str = '_'.join(parts[1:])

    # Set up ArUco detector with optimal parameters.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.02
    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 29
    parameters.adaptiveThreshWinSizeStep = 3
    parameters.polygonalApproxAccuracyRate = 0.08

    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    if ENABLE_BENCHMARK == True:
        # Benchmark timers
        benchmark_list = []
        video_start_time = time.perf_counter() if ENABLE_BENCHMARK else None
        cap_open_time = 0.0
        frame_read_time_total = 0.0
        gray_conversion_time_total = 0.0
        detection_time_total = 0.0
        csv_write_time_total = 0.0

        # Open video file and record the open time.
        start_cap = time.perf_counter()
        cap = cv2.VideoCapture(video_path)
        cap_open_time = time.perf_counter() - start_cap
        if not cap.isOpened():
            print("Could not open video file:", video_path)
            return 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
        frame_idx = 0
        rows = []
        for _ in range(total_frames):
            start = time.perf_counter()
            ret, frame = cap.read()
            read_time = time.perf_counter() - start
            frame_read_time_total += read_time

            if not ret:
                break
            start = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            conv_time = time.perf_counter() - start
            gray_conversion_time_total += conv_time

            start = time.perf_counter()
            markerCorners, markerIds, _ = detector.detectMarkers(gray)
            detect_time = time.perf_counter() - start
            detection_time_total += detect_time

            if markerIds is not None:
                start = time.perf_counter()
                for corners, marker_id in zip(markerCorners, markerIds.flatten()):
                    corners = np.squeeze(corners)
                    centroid = np.mean(corners, axis=0)
                    front = (corners[0] + corners[1]) / 2
                    rows.append([filename_no_ext,
                                    colony_number,
                                    datetime_str,
                                    frame_idx,
                                    int(marker_id),
                                    round(float(centroid[0]), 2),
                                    round(float(centroid[1]), 2),
                                    round(float(front[0]), 2),
                                    round(float(front[1]), 2)])
            frame_idx += 1

        start = time.perf_counter()
        
        with open(csv_filename, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "colony number", "datetime", "frame", "ID",
                            "centroidX", "centroidY", "frontX", "frontY"])
            writer.writerows(rows)

        write_time = time.perf_counter() - start
        csv_write_time_total += write_time
        
        cap.release()

        
            
        print(f"\nBenchmark for {video_path}:")
        print(f"  Video open time: {cap_open_time:.3f} s")
        print(f"  Total frame read time: {frame_read_time_total:.3f} s")
        print(f"  Total gray conversion time: {gray_conversion_time_total:.3f} s")
        print(f"  Total marker detection time: {detection_time_total:.3f} s")
        print(f"  Total CSV write time: {csv_write_time_total:.3f} s")
        total_video_time = time.perf_counter() - video_start_time
        print(f"  Overall processing time: {total_video_time:.3f} s")
        print(f"Tracking complete for {video_path}. Data saved to: {csv_filename}")
        sys.stdout.flush()

        benchmark_list = [0, cap_open_time, frame_read_time_total, gray_conversion_time_total, detection_time_total, csv_write_time_total, total_video_time]

    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open video file:", video_path)
            return 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = []
        frame_idx = 0
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            markerCorners, markerIds, _ = detector.detectMarkers(gray)

            if markerIds is not None:
                for corners, marker_id in zip(markerCorners, markerIds.flatten()):
                    corners = np.squeeze(corners)
                    centroid = np.mean(corners, axis=0)
                    front = (corners[0] + corners[1]) / 2
                    rows.append([filename_no_ext,
                                    colony_number,
                                    datetime_str,
                                    frame_idx,
                                    int(marker_id),
                                    round(float(centroid[0]), 2),
                                    round(float(centroid[1]), 2),
                                    round(float(front[0]), 2),
                                    round(float(front[1]), 2)])
            frame_idx += 1

        with open(csv_filename, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "colony number", "datetime", "frame", "ID",
                            "centroidX", "centroidY", "frontX", "frontY"])
            writer.writerows(rows)

        cap.release()

        benchmark_list = None
    
    return benchmark_list
    

def process_video_batch(video_batch):
    """
    Process a batch (list) of video files.
    Each video in the batch is processed sequentially in this worker.
    Returns the number of videos processed successfully.
    """
    count = 0
    if ENABLE_BENCHMARK:
        benchmark = None
        for video_path in video_batch:
            benchmark_list = process_video(video_path)
            count += 1
            if benchmark == None:
                benchmark = pd.Series([benchmark_list], columns=["videos analyzed", "cap_open_time", "frame_read_time_total", "gray_conversion_time_total", "detection_time_total", "csv_write_time_total", "total_video_time"])
                benchmark['videos analyzed'] = count
            else:
                
                benchmark["videos analyzed"] = count
                benchmark['cap_open_time'] =  ((benchmark['cap_open_time'] * (count - 1)) + benchmark_list[1]) / count
                benchmark['frame_read_time_total'] = ((benchmark['frame_read_time_total'] * (count - 1)) + benchmark_list[2]) / count
                benchmark['gray_conversion_time_total'] = ((benchmark['gray_conversion_time_total'] * (count - 1)) + benchmark_list[3]) / count
                benchmark['detection_time_total'] = ((benchmark['detection_time_total'] * (count - 1)) + benchmark_list[4]) / count
                benchmark['csv_write_time_total'] = ((benchmark['csv_write_time_total'] * (count - 1)) + benchmark_list[5]) / count
                benchmark['total_video_time'] = ((benchmark['total_video_time'] * (count - 1)) + benchmark_list[6]) / count
                
                print(f"\nCurrent average benchmark for {count} videos:")
                print(f"  Video open time: {benchmark_list[1]:.3f} s")
                print(f"  Total frame read time: {benchmark_list[2]:.3f} s")
                print(f"  Total gray conversion time: {benchmark_list[3]:.3f} s")
                print(f"  Total marker detection time: {benchmark_list[4]:.3f} s")
                print(f"  Total CSV write time: {benchmark_list[5]:.3f} s")
                print(f"  Overall processing time: {benchmark_list[6]:.3f} s")
                sys.stdout.flush()

        return count

    else:
        for video_path in video_batch:
            benchmark_list = process_video(video_path) # benchmark_list is None
            count += 1

    return count

def main():
    """
    Main function:
      1. Parses command-line options.
      2. Uses os.scandir() to recursively list and cache .mp4 files.
      3. Optionally filters files based on start_date or start_file.
      4. Splits the video file list into batches (5 per batch).
      5. Processes batches in parallel using a Pool with dynamically adjusted chunk size.
    """
    #global ENABLE_BENCHMARK  # to update the global flag based on the command-line argument

    parser = argparse.ArgumentParser(
        description="Process MP4 videos for ArUco tracking from a specific subfolder."
    )
    parser.add_argument('--volume', type=str, required=True,
                        help='Path to the removable volume (e.g., /media/username/volume).')
    parser.add_argument('--folder', type=str, required=True,
                        help='Name of the subfolder to process (e.g., bumblebox-11).')
    parser.add_argument('--cores', type=int, default=1,
                        help='Number of CPU cores to use for processing.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of videos to process in each batch when parallel processing.')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Process only video files from subfolders with dates (YYYY-MM-DD) >= this date.')
    parser.add_argument('--start_file', type=str, default=None,
                        help='Process only video files with basenames >= this filename.')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit the number of files to process (for testing purposes).')
    parser.add_argument('--benchmark', action='store_true',
                        help='Enable targeted benchmarks and print timing information.')
    parser.add_argument('--extension', type=str, default='_raw.csv',
                        help='Extension for the output CSV files.')
    parser.add_argument('--pool-mapping', type=str, default='imap_unordered', choices=['map', 'imap_unordered'],
                        help='Mapping method for Pool processing.')
                 
    args = parser.parse_args()

    global out_file_extension #doing this to make the extension available to the process_video function easily
    out_file_extension = args.extension
    
    if args.start_date is not None and args.start_file is not None:
        print("Error: Please provide only one of --start_date or --start_file.")
        exit(1)
    
    #ENABLE_BENCHMARK = args.benchmark

    target_folder = os.path.join(args.volume, args.folder)
    if not os.path.isdir(target_folder):
        print(f"Error: {target_folder} is not a valid directory.")
        exit(1)
    
    # Cache directory listing using os.scandir().
    video_files = list_mp4_files(target_folder)
    
    # Filter based on start_date.
    if args.start_date is not None:
        try:
            start_date = ciso8601.parse_datetime(args.start_date)
            print(f"Filtering files with dates >= {start_date}.")
        except ValueError:
            print("Error: start_date must be in YYYY-MM-DD format.")
            exit(1)
        filtered_files = []
        for video_file in video_files:
            print(video_file)
            # Expect the first directory in the relative path to be a date folder.
            rel_path = os.path.relpath(video_file, target_folder)
            print(rel_path)
            parts = rel_path.split(os.sep)
            print(parts)
            if parts:
                folder_name = parts[0]
                try:
                    folder_date = ciso8601.parse_datetime(folder_name)
                    if folder_date >= start_date:
                        print(f"appending {video_file}")
                        filtered_files.append(video_file)
                except ValueError: # If the folder name isn't a valid date, include it by default.
                    filtered_files.append(video_file)
        video_files = filtered_files
    elif args.start_file is not None:
        start_file_basename = os.path.basename(args.start_file)
        start_file_dt = extract_datetime_from_filename(start_file_basename)
        if start_file_dt is None:
            print("Error: Could not parse datetime from --start_file argument.")
            exit(1)
        filtered_files = []
        for video_file in video_files:
            file_dt = extract_datetime_from_filename(os.path.basename(video_file))
            if file_dt is not None and file_dt >= start_file_dt:
                filtered_files.append(video_file)
        video_files = filtered_files
    
    print(f"Found {len(video_files)} video file(s) in {target_folder} to be processed.")
    if args.limit is not None:
        video_files = video_files[:args.limit]
    
    #If we are implementing parallel processing
    if args.cores > 1:
        print(f"Parallel processing enabled with {args.cores} CPU cores.")
        num_processes = args.cores
    # Implement batch processing if batches are bigger than 1
        if args.batch_size > 1:
            batch_size = args.batch_size
            batches = [video_files[i:i+batch_size] for i in range(0, len(video_files), batch_size)]
    
    # Dynamically determine chunk size:
    # Let's aim for about (num_batches / (cores*4)) per chunk.
    #num_batches = len(batches)
    #chunk_size = max(1, num_batches // (args.cores * 4))
    #print(f"Processing {num_batches} batches with a chunk size of {chunk_size}.")

            if args.pool_mapping == 'map':
                start_time = pd.Timestamp.now()
                with Pool(processes=num_processes) as pool:
                    #if not ENABLE_CPROFILE and not ENABLE_YAPPI:
                        # Use imap_unordered to process batches and update progress.
                    #    for _ in tqdm(pool.imap_unordered(process_video_batch, batches, chunksize=chunk_size),
                    #                  total=num_batches,
                    #                  desc="Batches processed"):
                    #        pass
                    #elif ENABLE_CPROFILE:
                    #    pool.map(process_video_batch, batches, chunksize=chunk_size)
                    #else:
                    #pool.map(process_video, video_files)#(process_video_batch, batches) #, chunksize=chunk_size)
                    pool.map(process_video_batch, batches)

                end_time = pd.Timestamp.now()
                print(f"Execution of pool.map with batch size {batch_size} using {num_processes} CPU cores and analyzing {len(video_files)} videos took {end_time - start_time}.")
                
            elif args.pool_mapping == 'imap_unordered':
                start_time = pd.Timestamp.now()
                with Pool(processes=num_processes) as pool:
                    #if not ENABLE_CPROFILE and not ENABLE_YAPPI:
                        # Use imap_unordered to process batches and update progress.
                        for _ in tqdm(pool.imap_unordered(process_video_batch, batches),
                                    total=len(batches),
                                    desc="Batches processed"):
                            pass

                end_time = pd.Timestamp.now()
                print(f"Execution of pool.imap_unordered with batch size {batch_size} using {num_processes} CPU cores and analyzing {len(video_files)} videos took {end_time - start_time}.")

        elif args.batch_size == 1:
            start_time = pd.Timestamp.now()

            if args.pool_mapping == 'imap_unordered':
                with Pool(processes=num_processes) as pool:
                    for _ in tqdm(pool.imap_unordered(process_video, video_files),
                                total=len(video_files),
                                desc="Videos processed"):
                        pass
                end_time = pd.Timestamp.now()
                print(f"Execution of pool.imap_unordered with batch size 1 using {num_processes} CPU cores and analyzing {len(video_files)} videos took {end_time - start_time}.")

            elif args.pool_mapping == 'map':
                with Pool(processes=num_processes) as pool:
                    pool.map(process_video, video_files)
                end_time = pd.Timestamp.now()
                print(f"Execution of pool.map with batch size 1 using {num_processes} CPU cores and analyzing {len(video_files)} videos took {end_time - start_time}.")

    #If we are not implementing parallel processing
    else:
        print("No parallel processing enabled.")
        print("ENABLE_BENCHMARK is", ENABLE_BENCHMARK)
        if ENABLE_BENCHMARK == True:
            print("Benchmark mode enabled.")
            avg_benchmark = []
            count = 0
            start_time = pd.Timestamp.now()
            for video_file in video_files:

                benchmark_list = process_video(video_file)
                if benchmark_list == 1:
                    print(f"Error processing {video_file}.")
                    continue

                if count == 0:
                    avg_benchmark = benchmark_list
                    count = 1
                else:
                    count += 1
                    avg_benchmark[0] = count
                    avg_benchmark[1] =  ((avg_benchmark[1] * (avg_benchmark[0] - 1)) + benchmark_list[1]) / avg_benchmark[0]
                    avg_benchmark[2] = ((avg_benchmark[2] * (avg_benchmark[0] - 1)) + benchmark_list[2]) / avg_benchmark[0]
                    avg_benchmark[3] = ((avg_benchmark[3] * (avg_benchmark[0] - 1)) + benchmark_list[3]) / avg_benchmark[0]
                    avg_benchmark[4] = ((avg_benchmark[4] * (avg_benchmark[0] - 1)) + benchmark_list[4]) / avg_benchmark[0]
                    avg_benchmark[5] = ((avg_benchmark[5] * (avg_benchmark[0] - 1)) + benchmark_list[5]) / avg_benchmark[0]
                    avg_benchmark[6] = ((avg_benchmark[6] * (avg_benchmark[0] - 1)) + benchmark_list[6]) / avg_benchmark[0]

                    print(f"\nCurrent average benchmark for {avg_benchmark[0]} videos:")
                    print(f"  Video open time: {avg_benchmark[1]:.3f} s")
                    print(f"  Total frame read time: {avg_benchmark[2]:.3f} s")
                    print(f"  Total gray conversion time: {avg_benchmark[3]:.3f} s")
                    print(f"  Total marker detection time: {avg_benchmark[4]:.3f} s")
                    print(f"  Total CSV write time: {avg_benchmark[5]:.3f} s")
                    print(f"  Overall processing time: {avg_benchmark[6]:.3f} s")
                    sys.stdout.flush()

            end_time = pd.Timestamp.now()
            print(f"Execution without parallel processing and analyzing {len(video_files)} videos took {end_time - start_time}.")

        else:
            start_time = pd.Timestamp.now()
            for video_file in tqdm(video_files, desc="Videos processed"):
                process_video(video_file)
            end_time = pd.Timestamp.now()
            print(f"Execution without parallel processing and analyzing {len(video_files)} videos took {end_time - start_time}.")

        return 0

if __name__ == '__main__':

    start_time = pd.Timestamp.now()
    status = main()
    end_time = pd.Timestamp.now()
    print(f"Execution took {end_time - start_time}.")
    sys.exit(status)
