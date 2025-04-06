import cv2
from cv2 import aruco
import pandas as pd
import numpy as np
import os
import time

def track_tags_from_video(filepath, output_dir, filename, tag_dictionary='4X4_50', box_type='custom', aruco_params=None):
    """
    Tracks ArUco tags from a prerecorded video.

    Args:
        filepath (str): Path to the video file.
        output_dir (str): Path to save output CSVs.
        filename (str): Base name for output files.
        tag_dictionary (str): Dictionary name, e.g., '4X4_50'.
        box_type (str or None): 'custom', 'koppert', or None to skip presets.
        aruco_params (dict or None): Optional dictionary of ArUco DetectorParameters to override or define custom settings.

    Returns:
        pd.DataFrame: DataFrame with tagged IDs
        pd.DataFrame: DataFrame with untagged markers (no ID)
        int: Total number of processed frames
    """
    print(f"Tracking tags in video: {filepath}")

    if 'DICT' not in tag_dictionary:
        tag_dictionary = f'DICT_{tag_dictionary.upper()}'
    else:
        tag_dictionary = tag_dictionary.upper()

    if not hasattr(cv2.aruco, tag_dictionary):
        raise ValueError(f"Unknown tag dictionary: {tag_dictionary}")

    dictionary = getattr(cv2.aruco, tag_dictionary)
    tag_dict = aruco.getPredefinedDictionary(dictionary)
    parameters = aruco.DetectorParameters()

    # Apply preset parameters for specific box types
    #Change these to match the BumbleBox!
    if box_type == 'custom':
        parameters.minMarkerPerimeterRate = 0.03
        parameters.adaptiveThreshWinSizeMin = 5
        parameters.adaptiveThreshWinSizeStep = 6
        parameters.polygonalApproxAccuracyRate = 0.06

    elif box_type == 'koppert':
        print("Note: 'koppert' box_type selected, but no presets defined yet.")
        print("You should provide ArUco parameters manually via the aruco_params dictionary.")

    elif box_type is None:
        print("No box_type selected. Skipping preset ArUco parameters.")

    # Override with user-defined ArUco parameters if provided
    if aruco_params:
        print("Applying user-defined ArUco parameters:")
        for param_name, param_value in aruco_params.items():
            if hasattr(parameters, param_name):
                setattr(parameters, param_name, param_value)
                print(f" - {param_name} = {param_value}")
            else:
                print(f"Warning: Unrecognized aruco parameter '{param_name}'")

    detector = aruco.ArucoDetector(tag_dict, parameters)

    cap = cv2.VideoCapture(filepath)
    frame_num = 0
    noID, raw = [], []
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(gray)
            gray = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
        except:
            print("Grayscale conversion failed on frame", frame_num)
            continue

        corners, ids, rejected = detector.detectMarkers(gray)

        for rej in rejected:
            c = rej[0]
            xmean = c[:, 0].mean()
            ymean = c[:, 1].mean()
            x_top = (c[0, 0] + c[1, 0]) / 2
            y_top = (c[0, 1] + c[1, 1]) / 2
            noID.append([frame_num, "X", xmean, ymean, x_top, y_top])

        if ids is not None:
            for i in range(len(ids)):
                c = corners[i][0]
                xmean = c[:, 0].mean()
                ymean = c[:, 1].mean()
                x_top = (c[0, 0] + c[1, 0]) / 2
                y_top = (c[0, 1] + c[1, 1]) / 2
                raw.append([frame_num, int(ids[i]), xmean, ymean, x_top, y_top])

        frame_num += 1

    cap.release()

    df = pd.DataFrame(raw, columns=['frame', 'ID', 'centroidX', 'centroidY', 'frontX', 'frontY'])
    df2 = pd.DataFrame(noID, columns=['frame', 'ID', 'centroidX', 'centroidY', 'frontX', 'frontY'])

    if output_dir is not None:
        df.to_csv(os.path.join(output_dir, filename + '_raw.csv'), index=False)
        df2.to_csv(os.path.join(output_dir, filename + '_noID.csv'), index=False)

    elapsed = time.time() - start
    print(f"Processed {frame_num} frames in {round(elapsed, 2)} seconds")

    return df, df2, frame_num

def load_actual_fps(video_path):
    """
    Attempts to load the actual FPS metadata for a video.
    Expects a file next to the video named '<video_basename>_actual_fps.txt'

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: actual FPS if found, otherwise None
    """
    base = os.path.splitext(video_path)[0]
    fps_path = base + '_actual_fps.txt'
    if os.path.exists(fps_path):
        with open(fps_path, 'r') as f:
            try:
                fps = float(f.readline().strip())
                print(f"Loaded actual FPS from metadata: {fps:.3f}")
                return fps
            except ValueError:
                print(f"Warning: Could not parse FPS from {fps_path}")
    else:
        print(f"No FPS metadata file found for {video_path}")
    return None

def main():
    print("I am a module, not a script.")
    print("Please import me and call the function track_tags_from_video()")
    print("with the appropriate parameters.")
    print("Example usage:")
    print("track_tags_from_video('path/to/video.mp4', 'output/directory', 'output_filename', '4X4_50', 'custom', {'adaptiveThreshWinSizeMin': 5})")
    print("Note: The box_type parameter can be 'custom', 'koppert', or None.")
    print("The aruco_params parameter can be a dictionary of ArUco DetectorParameters.")
    print("For more details, refer to the function docstring.")

if __name__ == '__main__':
    main()

