#!/usr/bin/env python

import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
from cv2 import aruco


def compute_speed(df: pd.DataFrame, fps: float, speed_cutoff_seconds: int) -> pd.DataFrame:
    speed_cutoff_frames = fps * speed_cutoff_seconds
    df_sorted = df.sort_values(by=["ID", "frame"])
    df_sorted["deltaX"] = df_sorted.groupby("ID")["centroidX"].diff()
    df_sorted["deltaY"] = df_sorted.groupby("ID")["centroidY"].diff()
    df_sorted["elapsed frames"] = df_sorted.groupby("ID")["frame"].diff()
    sub_df = df_sorted[df_sorted["elapsed frames"] < speed_cutoff_frames]
    sub_df["speed"] = np.sqrt(sub_df["deltaX"] ** 2 + sub_df["deltaY"] ** 2)
    df_sorted.loc[:, "speed"] = sub_df.loc[:, "speed"]
    df_sorted.drop(columns=["deltaX", "deltaY"], inplace=True)
    return df_sorted


def compute_social_center_distance(df: pd.DataFrame) -> pd.DataFrame:
    social_centers = df.groupby("frame")[["centroidX", "centroidY"]].mean()
    social_centers.columns = ["centerX", "centerY"]
    df = df.merge(social_centers, left_on="frame", right_index=True)
    df["distance_from_center"] = np.sqrt(
        (df["centroidX"] - df["centerX"]) ** 2 + (df["centroidY"] - df["centerY"]) ** 2
    )
    df.drop(columns=["centerX", "centerY"], inplace=True)
    return df


def _resolve_dictionary(tag_dictionary):
    if tag_dictionary is None:
        tag_dictionary = "4X4_50"
    if "DICT" not in tag_dictionary:
        tag_dictionary = "DICT_%s" % tag_dictionary
    tag_dictionary = tag_dictionary.upper()
    if not hasattr(cv2.aruco, tag_dictionary):
        raise ValueError("Unknown tag dictionary: %s" % tag_dictionary)
    return aruco.getPredefinedDictionary(getattr(cv2.aruco, tag_dictionary))


def _build_detector(tag_dictionary, box_type):
    tag_dictionary = _resolve_dictionary(tag_dictionary)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(tag_dictionary, parameters)

    if box_type == "custom":
        parameters.minMarkerPerimeterRate = 0.03
        parameters.adaptiveThreshWinSizeMin = 5
        parameters.adaptiveThreshWinSizeStep = 6
        parameters.polygonalApproxAccuracyRate = 0.06
    elif box_type == "koppert":
        parameters.minMarkerPerimeterRate = 0.03
        parameters.adaptiveThreshWinSizeMin = 5
        parameters.adaptiveThreshWinSizeStep = 6
        parameters.polygonalApproxAccuracyRate = 0.06
    return detector


def trackTagsFromVid(filepath, output_folder, filename, tag_dictionary, box_type):
    detector = _build_detector(tag_dictionary, box_type)

    vid = cv2.VideoCapture(filepath)
    if not vid.isOpened():
        raise IOError(f"Could not open video file: {filepath}")

    frame_num = 0
    noID = []
    raw = []
    start = time.time()

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(gray)
            gray = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
        except Exception:
            print("converting to grayscale didnt work...")
            continue

        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

        for i in range(len(rejectedImgPoints)):
            c = rejectedImgPoints[i][0]
            xmean = c[:, 0].mean()
            ymean = c[:, 1].mean()
            xmean_top_point = (c[0, 0] + c[1, 0]) / 2
            ymean_top_point = (c[0, 1] + c[1, 1]) / 2
            noID.append(
                [
                    frame_num,
                    "X",
                    float(xmean),
                    float(ymean),
                    float(xmean_top_point),
                    float(ymean_top_point),
                ]
            )

        if ids is not None:
            for i in range(len(ids)):
                c = corners[i][0]
                xmean = c[:, 0].mean()
                ymean = c[:, 1].mean()
                xmean_top_point = (c[0, 0] + c[1, 0]) / 2
                ymean_top_point = (c[0, 1] + c[1, 1]) / 2
                raw.append(
                    [
                        frame_num,
                        int(ids[i]),
                        float(xmean),
                        float(ymean),
                        float(xmean_top_point),
                        float(ymean_top_point),
                    ]
                )

        frame_num += 1
        print(f"processed frame {frame_num}")

    vid.release()

    df = pd.DataFrame(raw, columns=["frame", "ID", "centroidX", "centroidY", "frontX", "frontY"])
    df2 = pd.DataFrame(noID, columns=["frame", "ID", "centroidX", "centroidY", "frontX", "frontY"])

    raw_out = os.path.join(output_folder, f"{filename}_raw.csv")
    noid_out = os.path.join(output_folder, f"{filename}_noID.csv")
    df.to_csv(raw_out, index=False)
    df2.to_csv(noid_out, index=False)
    print(f"saved raw csv: {raw_out}")
    print(f"saved noID csv: {noid_out}")

    tracking_time = time.time() - start
    if frame_num > 0:
        print("Average number of tags found: " + str(len(df.index) / frame_num))
        print(
            f"Tag tracking took {tracking_time} seconds, an average of {tracking_time / frame_num} seconds per frame"
        )
    else:
        print(f"Tag tracking took {tracking_time} seconds (0 frames processed).")
    return df, df2, frame_num


def main(video_folder, dictionary, fps=None, box_type="custom"):
    for path, directories, files in os.walk(video_folder):
        for file in sorted(files):
            if not file.lower().endswith((".mp4", ".mjpeg")):
                continue
            filepath = os.path.join(path, file)
            filename, _ = os.path.splitext(file)
            print(f"starting to track tags from {filepath}")
            df, df2, frame_num = trackTagsFromVid(filepath, path, filename, dictionary, box_type)

            if fps is not None and fps > 0 and not df.empty:
                updated = compute_speed(df, fps, 4)
                updated = compute_social_center_distance(updated)
                updated_out = os.path.join(path, f"{filename}_updated.csv")
                updated.to_csv(updated_out, index=False)
                print(f"saved updated csv: {updated_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Track tags from pre-recorded videos in a folder."
    )
    parser.add_argument("--video-folder", required=True, help="Folder containing videos.")
    parser.add_argument(
        "--dictionary",
        default="4X4_50",
        help="ArUco dictionary, e.g. 4X4_50.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional fps for speed metrics output.",
    )
    parser.add_argument(
        "--box-type",
        choices=["custom", "koppert"],
        default="custom",
        help="Tracking parameter preset.",
    )
    args = parser.parse_args()
    main(args.video_folder, args.dictionary, args.fps, args.box_type)
