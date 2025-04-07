# ram_capture_tag_tracking.py

import os
import time
import cv2
import socket
import pandas as pd
import argparse
import pwd
from datetime import datetime
from picamera2 import Picamera2
from libcamera import controls
import behavioral_metrics
from data_cleaning import interpolate
from config_loader import load_config
from record_video import create_todays_folder
from tag_tracking_utils import trackTagsFromRAM  # If needed, adjust import

config = load_config()

username = pwd.getpwuid(os.getuid())[0]


def array_capture(recording_time, fps, shutter_speed, width, height, tuning_file, noise_reduction_mode, digital_zoom, outdir=None, filename=None):
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
            print("The variable 'noise_reduction_mode' in config.yaml is invalid. Use: Auto, Off, Fast, HighQuality")

    if isinstance(digital_zoom, tuple) and len(digital_zoom) == 4:
        picam2.set_controls({"ScalerCrop": digital_zoom})
    elif digital_zoom is not None:
        print("Invalid recording_digital_zoom in config.yaml. Use None or 4-value tuple.")

    picam2.start()
    time.sleep(2)

    print("Initializing RAM-based frame capture...")
    print("Recording parameters:\n")
    print(f"\trecording time: {recording_time}s")
    print(f"\tframes per second: {fps}")
    print(f"\timage resolution: {width}x{height} pixels")
    print(f"\toutput image format: RGB888")
    print(f"\toutput storage format: YUV420 frame array (RAM only)")

    target_interval = 1.0 / fps
    start_time = time.perf_counter()
    frame_index = 0
    frames_list = []

    print("Beginning video capture")
    while (time.perf_counter() - start_time) < recording_time:
        now = time.perf_counter()
        expected_time = start_time + frame_index * target_interval
        if now >= expected_time:
            yuv420 = picam2.capture_array()
            frames_list.append([yuv420])
            frame_index += 1

    finished = time.perf_counter() - start_time
    actual_fps = frame_index / finished
    print(f"\nFinished capturing frames to arrays, captured {frame_index} frames in {round(finished, 2)} seconds")
    print(f"That's {round(actual_fps, 2)} frames per second!")
    print("Make sure this corresponds well to your desired framerate.")

    if outdir and filename:
        fps_path = os.path.join(outdir, filename + '_actual_fps.txt')
        with open(fps_path, 'w') as f:
            f.write(f"{actual_fps:.3f}\n")
        print(f"Saved actual FPS to {fps_path}")

    return frames_list, actual_fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_folder_path', type=str, default=config["todays_folder_path"])
    parser.add_argument('-fps', '--frames_per_second', type=int, default=config["frames_per_second"])
    args = parser.parse_args()

    ret, todays_folder_path = create_todays_folder(args.data_folder_path)
    if ret == 1:
        print("Couldn't create today's folder. Exiting.")
        return 1

    hostname = socket.gethostname()
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{hostname}_{now}"

    print(f"Session filename: {filename}")

    frames_list, recorded_fps = array_capture(
        config["recording_time"],
        config["frames_per_second"],
        config["shutter_speed"],
        config["width"],
        config["height"],
        config["tuning_file"],
        config["noise_reduction_mode"],
        tuple(config["recording_digital_zoom"]) if config["recording_digital_zoom"] else None,
        outdir=todays_folder_path,
        filename=filename
    )

    actual_fps = recorded_fps

    # Load user-defined ArUco params if provided
    aruco_params = config.get("aruco_params", None)

    # Only pass aruco_params if box_type is None (i.e., full custom mode)
    if config["box_type"] is None and aruco_params:
        print("Beginning tag tracking using aruco parameters defined in the config file...")
        df, df2, frame_num = trackTagsFromRAM(
            filename, todays_folder_path, frames_list,
            config["tag_dictionary"], None, now, hostname, config["colony_number"],
            aruco_params=aruco_params
        )
    else:
        print("Beginning tag tracking...")
        df, df2, frame_num = trackTagsFromRAM(
            filename, todays_folder_path, frames_list,
            config["tag_dictionary"], config["box_type"], now, hostname, config["colony_number"]
        )

    if config["interpolate_data"] and not df.empty:
        df = interpolate(df, config["max_seconds_gap"], actual_fps)

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
