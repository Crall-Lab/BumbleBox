# tag_tracking_utils.py

import cv2
from cv2 import aruco
import pandas as pd
import numpy as np
import os
import time

def load_actual_fps(filepath):
    try:
        with open(filepath, 'r') as f:
            return float(f.readline().strip())
    except Exception:
        return None

def trackTagsFromVid(filepath, todays_folder_path, filename, tag_dictionary, box_type, now, colony_number, aruco_params=None):
    if tag_dictionary is None:
        tag_dictionary = '4X4_50'

    if 'DICT' not in tag_dictionary:
        tag_dictionary = "DICT_" + tag_dictionary
    tag_dictionary = tag_dictionary.upper()

    if not hasattr(cv2.aruco, tag_dictionary):
        raise ValueError("Unknown tag dictionary: %s" % tag_dictionary)

    tag_dictionary = getattr(cv2.aruco, tag_dictionary)
    tag_dictionary = aruco.getPredefinedDictionary(tag_dictionary)

    parameters = aruco.DetectorParameters()
  
    #Apply preset parameters for specific box types
    #Optimized custom BumbleBox parameters:  
    if box_type == 'custom':
        parameters.minMarkerPerimeterRate = 0.02
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizMax = 31
        parameters.adaptiveThreshWinSizeStep = 3
        parameters.polygonalApproxAccuracyRate = 0.08

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
              
    detector = aruco.ArucoDetector(tag_dictionary, parameters)

    vid = cv2.VideoCapture(filepath)
    frame_num = 0
    noID = []
    raw = []
    start = time.time()

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(gray)
            gray = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
        except:
            continue

        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

        for c in rejectedImgPoints:
            c = c[0]
            xmean = c[:,0].mean()
            ymean = c[:,1].mean()
            xmean_top_point = (c[0,0] + c[1,0]) / 2
            ymean_top_point = (c[0,1] + c[1,1]) / 2
            noID.append([filename, colony_number, now, frame_num, "X", xmean, ymean, xmean_top_point, ymean_top_point])

        if ids is not None:
            for i in range(len(ids)):
                c = corners[i][0]
                xmean = c[:,0].mean()
                ymean = c[:,1].mean()
                xmean_top_point = (c[0,0] + c[1,0]) / 2
                ymean_top_point = (c[0,1] + c[1,1]) / 2
                raw.append([filename, colony_number, now, frame_num, int(ids[i]), xmean, ymean, xmean_top_point, ymean_top_point])

        frame_num += 1

    df = pd.DataFrame(raw, columns=["filename", "colony number", "datetime", "frame", "ID", "centroidX", "centroidY", "frontX", "frontY"])
    df.to_csv(f"{todays_folder_path}/{filename}_raw.csv", index=False)

    df2 = pd.DataFrame(noID, columns=["filename", "colony number", "datetime", "frame", "ID", "centroidX", "centroidY", "frontX", "frontY"])
    df2.to_csv(f"{todays_folder_path}/{filename}_noID.csv", index=False)

    print(f"Tag tracking took {round(time.time() - start, 2)} seconds")
    return df, df2, frame_num



def trackTagsFromRAM(filename, todays_folder_path, frames_list, tag_dictionary, box_type, now, hostname, colony_number, aruco_params=None):
    if tag_dictionary is None:
        tag_dictionary = '4X4_50'

    if 'DICT' not in tag_dictionary:
        tag_dictionary = "DICT_" + tag_dictionary
    tag_dictionary = tag_dictionary.upper()

    if not hasattr(cv2.aruco, tag_dictionary):
        raise ValueError("Unknown tag dictionary: %s" % tag_dictionary)

    tag_dictionary = getattr(cv2.aruco, tag_dictionary)
    tag_dictionary = aruco.getPredefinedDictionary(tag_dictionary)

    parameters = aruco.DetectorParameters()
  
    #Apply preset parameters for specific box types
    #Optimized custom BumbleBox parameters:  
    if box_type == 'custom':
        parameters.minMarkerPerimeterRate = 0.02
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizMax = 31
        parameters.adaptiveThreshWinSizeStep = 3
        parameters.polygonalApproxAccuracyRate = 0.08

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
    detector = aruco.ArucoDetector(tag_dictionary, parameters)

    frame_num = 0
    noID = []
    raw = []
    start = time.time()

    for index, frame in enumerate(frames_list):
        frame = frame[0]

        if index == int(len(frames_list) / 2):
            gray = cv2.cvtColor(frame.copy(), cv2.COLOR_YUV2GRAY_I420)
            cv2.imwrite(f"{todays_folder_path}/{filename}.png", gray)

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_I420)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(gray)
            gray = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
        except:
            continue

        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

        for c in rejectedImgPoints:
            c = c[0]
            xmean = c[:,0].mean()
            ymean = c[:,1].mean()
            xmean_top_point = (c[0,0] + c[1,0]) / 2
            ymean_top_point = (c[0,1] + c[1,1]) / 2
            noID.append([filename, colony_number, now, frame_num, "X", xmean, ymean, xmean_top_point, ymean_top_point])

        if ids is not None:
            for i in range(len(ids)):
                c = corners[i][0]
                xmean = c[:,0].mean()
                ymean = c[:,1].mean()
                xmean_top_point = (c[0,0] + c[1,0]) / 2
                ymean_top_point = (c[0,1] + c[1,1]) / 2
                raw.append([filename, colony_number, now, frame_num, int(ids[i]), xmean, ymean, xmean_top_point, ymean_top_point])

        frame_num += 1

    df = pd.DataFrame(raw, columns=["filename", "colony number", "datetime", "frame", "ID", "centroidX", "centroidY", "frontX", "frontY"])
    df.to_csv(f"{todays_folder_path}/{filename}_raw.csv", index=False)

    df2 = pd.DataFrame(noID, columns=["filename", "colony number", "datetime", "frame", "ID", "centroidX", "centroidY", "frontX", "frontY"])
    df2.to_csv(f"{todays_folder_path}/{filename}_noID.csv", index=False)

    print(f"Tag tracking took {round(time.time() - start, 2)} seconds")
    return df, df2, frame_num


import multiprocessing as mp

def _process_frame_chunk(args):
    """Worker function for processing a chunk of frames."""
    chunk_frames, start_frame_num, filename, colony_number, now, tag_dictionary, parameters = args

    detector = aruco.ArucoDetector(tag_dictionary, parameters)

    raw = []
    noID = []

    for i, frame in enumerate(chunk_frames):
        frame_num = start_frame_num + i
        frame = frame[0]

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_I420)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(gray)
            gray = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
        except:
            continue

        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

        for c in rejectedImgPoints:
            c = c[0]
            xmean = c[:, 0].mean()
            ymean = c[:, 1].mean()
            xmean_top = (c[0, 0] + c[1, 0]) / 2
            ymean_top = (c[0, 1] + c[1, 1]) / 2
            noID.append([filename, colony_number, now, frame_num, "X", xmean, ymean, xmean_top, ymean_top])

        if ids is not None:
            for j in range(len(ids)):
                c = corners[j][0]
                xmean = c[:, 0].mean()
                ymean = c[:, 1].mean()
                xmean_top = (c[0, 0] + c[1, 0]) / 2
                ymean_top = (c[0, 1] + c[1, 1]) / 2
                raw.append([filename, colony_number, now, frame_num, int(ids[j]), xmean, ymean, xmean_top, ymean_top])

    return raw, noID


def trackTagsFromRAM_parallel(filename, todays_folder_path, frames_list, tag_dictionary, box_type, now, colony_number, aruco_params=None, n_chunks=None):
    if tag_dictionary is None:
        tag_dictionary = '4X4_50'

    if 'DICT' not in tag_dictionary:
        tag_dictionary = "DICT_" + tag_dictionary
    tag_dictionary = tag_dictionary.upper()

    if not hasattr(cv2.aruco, tag_dictionary):
        raise ValueError("Unknown tag dictionary: %s" % tag_dictionary)

    tag_dict = getattr(cv2.aruco, tag_dictionary)
    tag_dict = aruco.getPredefinedDictionary(tag_dict)

    parameters = aruco.DetectorParameters()

    #Apply preset parameters for specific box types
    #Optimized custom BumbleBox parameters:  
    if box_type == 'custom':
        parameters.minMarkerPerimeterRate = 0.02
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizMax = 31
        parameters.adaptiveThreshWinSizeStep = 3
        parameters.polygonalApproxAccuracyRate = 0.08
        
    elif aruco_params:
        print("Applying user-defined ArUco parameters:")
        for param_name, param_value in aruco_params.items():
            if hasattr(parameters, param_name):
                setattr(parameters, param_name, param_value)
                print(f" - {param_name} = {param_value}")
            else:
                print(f"Warning: Unrecognized aruco parameter '{param_name}'")

    total_frames = len(frames_list)
    n_chunks = n_chunks or mp.cpu_count()
    chunk_size = (total_frames + n_chunks - 1) // n_chunks

    chunks = [
        (frames_list[i:i + chunk_size], i, filename, colony_number, now, tag_dict, parameters)
        for i in range(0, total_frames, chunk_size)
    ]

    print(f"Starting multiprocessing ArUco tracking with {len(chunks)} chunks...")

    with mp.Pool(processes=len(chunks)) as pool:
        results = pool.map(_process_frame_chunk, chunks)

    # Flatten results
    all_raw = [row for chunk_raw, _ in results for row in chunk_raw]
    all_noID = [row for _, chunk_noID in results for row in chunk_noID]

    df = pd.DataFrame(all_raw, columns=["filename", "colony number", "datetime", "frame", "ID", "centroidX", "centroidY", "frontX", "frontY"])
    df2 = pd.DataFrame(all_noID, columns=["filename", "colony number", "datetime", "frame", "ID", "centroidX", "centroidY", "frontX", "frontY"])

    df.to_csv(f"{todays_folder_path}/{filename}_raw.csv", index=False)
    df2.to_csv(f"{todays_folder_path}/{filename}_noID.csv", index=False)

    print(f"Multiprocessing tag tracking finished — {len(df)} tags, {len(df2)} no-ID detections")
    return df, df2, total_frames

