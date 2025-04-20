# record_video.py

import os
import sys
import cv2
import time
import pwd
import argparse
import socket
import subprocess
import pandas as pd
from datetime import date, datetime
from picamera2 import Picamera2, Preview
from picamera2.encoders import JpegEncoder
from libcamera import controls
from config_loader import load_config
import behavioral_metrics
import data_cleaning
from tag_tracking_utils import load_actual_fps
from tag_tracking_utils import trackTagsFromVid

config = load_config()
username = pwd.getpwuid(os.getuid())[0]

def create_todays_folder(dirpath):
    today = date.today().strftime('%Y-%m-%d')
    todays_folder_path = os.path.join(dirpath, today)
    print(todays_folder_path)
    if not os.path.exists(todays_folder_path):
        try:
            os.makedirs(todays_folder_path)
        except Exception as e:
            print("Attempting subprocess mkdir due to:", e)
            try:
                subprocess.call(['sudo', 'mkdir', '-p', todays_folder_path])
            except Exception as e:
                print("Subprocess mkdir failed:", e)
                return 1, todays_folder_path
    return 0, todays_folder_path

def picam2_record_mp4(filename, outdir):
    fps = config["camera_settings"]["frames_per_second"]
    shutter_speed = config["camera_settings"]["shutter_speed"]
    width = config["camera_settings"]["width"]
    height = config["camera_settings"]["height"]
    tuning_file = config["camera_settings"]["tuning_file"]
    noise_reduction_mode = config["camera_settings"]["noise_reduction_mode"]
    digital_zoom = config["camera_settings"]["recording_digital_zoom"]
    
    recording_time = config["recording_options"]["recording_time"]

    tuning = Picamera2.load_tuning_file(tuning_file)
    picam2 = Picamera2(tuning=tuning)
    preview = picam2.create_preview_configuration({"format": "YUV420", "size": (width, height)})
    picam2.align_configuration(preview)
    picam2.configure(preview)
    picam2.set_controls({"ExposureTime": shutter_speed})

    if noise_reduction_mode != "Auto":
        try:
            noise_reduction_mode = getattr(controls.draft.NoiseReductionModeEnum, noise_reduction_mode)
            picam2.set_controls({"NoiseReductionMode": noise_reduction_mode})
        except:
            print("Invalid noise_reduction_mode in config.yaml")

    if isinstance(digital_zoom, tuple) and len(digital_zoom) == 4:
        picam2.set_controls({"ScalerCrop": digital_zoom})
    elif digital_zoom is not None:
        print("Invalid recording_digital_zoom in config.yaml")

    picam2.start()

    print("Initializing recording...")
    print("Recording parameters:\n")
    print(f"\tfilename: {filename}")
    print(f"\tdirectory: {outdir}")
    print(f"\trecording time: {recording_time}s")
    print(f"\tframes per second: {fps}")
    print(f"\timage resolution: {width}x{height} pixels")
    print(f"\toutput format: mp4")

    time.sleep(2)
    start_time = time.perf_counter()
    target_interval = 1.0 / fps
    frames_list = []
    frame_index = 0

    print("Beginning video capture...")
    while (time.perf_counter() - start_time) < recording_time:
        now = time.perf_counter()
        expected_time = start_time + frame_index * target_interval
        if now >= expected_time:
            yuv420 = picam2.capture_array()
            frames_list.append(yuv420)
            frame_index += 1

    finished = time.perf_counter() - start_time
    actual_fps = frame_index / finished
    print(f"Finished capturing {frame_index} frames in {finished:.2f} seconds")
    print(f"Actual average FPS: {actual_fps:.2f}")

    output_path = os.path.join(outdir, filename + '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for idx, frame in enumerate(frames_list):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_YUV420p2RGB)
        out.write(rgb_frame)
        print(f"Wrote frame {idx + 1}/{len(frames_list)}")

    out.release()
    cv2.destroyAllWindows()

    # Save actual FPS to file (optional)
    with open(os.path.join(outdir, filename + '_actual_fps.txt'), 'w') as f:
        f.write(f"{actual_fps:.3f}")

    return frames_list, output_path, actual_fps


def main():
    if sys.stdout.isatty():
        print("Running video recording script from terminal")
    else:
        print("Running video recording script via crontab")

    if config["create_composite_nest_images"]:
        now = datetime.now()
        if now.hour == 23 and now.minute == 0:
            return print("Exiting to allow composite image generation to run")

    ret, todays_folder_path = create_todays_folder(config["todays_folder_path"])
    if ret == 1:
        return print("Failed to create today's folder")

    hostname = socket.gethostname()
    now = datetime.now().strftime('_%Y-%m-%d_%H_%M_%S')
    filename = hostname + now

    print("Filename:", filename)

    frames_list, filepath, actual_fps = picam2_record_mp4(filename, todays_folder_path)

    #Idk why I got rid of MJPEG recording, but for when I bring it back - we'll use this function for MJPEG and the trackRAM for MP4
    if config["recording_options"]["track_recorded_videos"] and config["camera_settings"]["codec"] == ".mjpeg":

        # Optional: load user-defined ArUco params if present in config.yaml
        aruco_params = config.get("aruco_params", None)

		# Choose tracking call depending on box_type and whether custom params are provided
        if config["box_type"] is None and aruco_params:
            df, df2, frame_num = trackTagsFromVid(
				filepath,
				todays_folder_path,
				filename,
				config["tag_dictionary"],
				None,  # No preset box_type
				now,
				config["colony_number"],
				aruco_params=aruco_params
			)
        else:
            df, df2, frame_num = trackTagsFromVid(
				filepath,
				todays_folder_path,
				filename,
				config["tag_dictionary"],
				config["box_type"],
				now,
				config["colony_number"]
			)

	#leverage the RAM tracking script to better use the multiprocessing function
    elif config["recording_options"]["track_recorded_videos"] and config["camera_settings"]["codec"] == ".mp4":

        # Optional: load user-defined ArUco params if present in config.yaml
        aruco_params = config.get("aruco_params", None)

		# Choose tracking call depending on box_type and whether custom params are provided
        if config["box_type"] is None and aruco_params:
            df, df2, frame_num = trackTagsFromRAM(
				filename,
				todays_folder_path,
		    		frames_list,
				config["tag_dictionary"],
				None,  # No preset box_type
				now,
				config["colony_number"],
				aruco_params=aruco_params
			)
        else:
            df, df2, frame_num = trackTagsFromRAM(
				filename,
				todays_folder_path,
		    		frames_list,
				config["tag_dictionary"],
				config["box_type"],
				now,
		    		hostname,
				config["colony_number"]
			)

	
	#The duplicate related functions do the following:
	#find and remove any completely duplicate rows 
	#find duplicated tags in the same frame that have different XY coordinates - this happens in Aruco tracking
	#the severity usually depends on aruco parameters you're using, so if you're experimenting with those, watch out for duplicates
	#remove the duplicate tag that is further away from the most recent instance of that tag being tracked (on either side of the frame with the duplicate tags)
	#based on the value of ______ in the config.yaml file, either remove both duplicate tags if there isn't a tag near enough to check against
	#OR flag both of them as being unresolvable duplicates - they'll have a True value in the "unresolvable duplicates" column
	#If 
	if config["data_cleaning"]["remove_jumps"] and not df.empty:

	    df = data_cleaning.remove_jumps(df)

        if config["data_cleaning"]["interpolate_data"] and not df.empty:
            # already have actual_fps from recording
            df = data_cleaning.interpolate(df, config["max_seconds_gap"], actual_fps)

	if config["data_cleaning"]["compute_heading_angle"] and not df.empty:

	    df = data_cleaning.compute_heading_angle(df)

        if not df.empty and config["calculate_behavior_metrics"]:
            behavioral_metrics.calculate_behavior_metrics(
                df,
                actual_fps,
                todays_folder_path,
                filename
            )
            print("Behavioral metrics calculated.")

if __name__ == "__main__":
    main()
