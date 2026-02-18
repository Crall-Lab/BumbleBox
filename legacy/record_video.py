# record_video.py

import os
import sys
import cv2
import time
import pwd
import socket
import subprocess
from datetime import date, datetime
from picamera2 import Picamera2
from libcamera import controls
from config_loader import load_config
import behavioral_metrics
import data_cleaning
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

    data_root = (
        config.get("todays_folder_path")
        or config.get("system", {}).get("data_root")
        or "/mnt/bumblebox/data"
    )
    ret, todays_folder_path = create_todays_folder(data_root)
    if ret == 1:
        return print("Failed to create today's folder")

    hostname = socket.gethostname()
    now = datetime.now().strftime('_%Y-%m-%d_%H_%M_%S')
    filename = hostname + now

    print("Filename:", filename)

    frames_list, filepath, actual_fps = picam2_record_mp4(filename, todays_folder_path)

    should_track = config.get("recording_options", {}).get("track_recorded_videos", False)
    codec = str(config.get("camera_settings", {}).get("codec", "mp4")).lower().lstrip(".")

    if should_track and codec == "mjpeg":

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

    elif should_track and codec == "mp4":
        aruco_params = config.get("aruco_params", None)
        use_parallel = config.get("recording_options", {}).get("use_parallel_ram_tracking", False)

        if use_parallel:
            from tag_tracking_utils import trackTagsFromRAM_parallel as trackRAM
        else:
            from tag_tracking_utils import trackTagsFromRAM as trackRAM

        print(f"Running {'parallel' if use_parallel else 'serial'} RAM-based ArUco tracking.")

        df, df2, frame_num = trackRAM(
            filename,
            todays_folder_path,
            [[frame] for frame in frames_list],
            config["tag_dictionary"],
            config["box_type"],
            now,
            config["colony_number"],
            aruco_params=aruco_params
        )
    else:
        return print("Tracking not enabled for this recording.")

    # The duplicate/jump handling and interpolation happen after tracking.
    cleaning_cfg = config.get("data_cleaning") or config.get("date cleaning") or {}
    max_seconds_gap = cleaning_cfg.get("max_seconds_gap", 3)
    moving_threshold = config.get("moving_threshold", 3.16)

    if cleaning_cfg.get("remove_jumps", False) and not df.empty:
        threshold = cleaning_cfg.get("jump_threshold_pixels", 500)
        df = data_cleaning.remove_jumps(df, jump_threshold_pixels=threshold)

    if cleaning_cfg.get("interpolate_data", False) and not df.empty:
        df = data_cleaning.interpolate(df, max_seconds_gap, actual_fps)

    if cleaning_cfg.get("compute_heading_angle", False) and not df.empty:
        df = data_cleaning.compute_heading_angle(df)

    if not df.empty and config.get("calculate_behavior_metrics", False):
        behavioral_metrics.calculate_behavior_metrics(
            df,
            actual_fps,
            moving_threshold,
            todays_folder_path,
            filename
        )
        print("Behavioral metrics calculated.")

if __name__ == "__main__":
    main()
