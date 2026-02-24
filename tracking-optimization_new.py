import argparse
import concurrent.futures
import itertools
import os
import random
import time

import cv2
from tqdm import tqdm


def _get_aruco_default_params():
    """Return OpenCV ArUco default values for the tuned parameter set."""
    params = cv2.aruco.DetectorParameters()
    return (
        float(params.minMarkerPerimeterRate),
        int(params.adaptiveThreshWinSizeMin),
        int(params.adaptiveThreshWinSizeMax),
        int(params.adaptiveThreshWinSizeStep),
        float(params.polygonalApproxAccuracyRate),
    )


def evaluate_params(params, frames, use_gpu, zero_score_streak_limit=0):
    """Evaluate a single parameter combination on a frame subset."""
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.minMarkerPerimeterRate = params[0]
    detector_params.adaptiveThreshWinSizeMin = params[1]
    detector_params.adaptiveThreshWinSizeMax = params[2]
    detector_params.adaptiveThreshWinSizeStep = params[3]
    detector_params.polygonalApproxAccuracyRate = params[4]

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    score = 0
    frame_count = 0
    zero_score_streak = 0
    early_stopped = False
    start_time = time.time()
    for frame in frames:
        if use_gpu:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                gray = gpu_gray.download()
            except Exception:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, marker_ids, _ = detector.detectMarkers(gray)
        marker_count = len(marker_ids) if marker_ids is not None else 0
        score += marker_count
        if marker_count == 0:
            zero_score_streak += 1
        else:
            zero_score_streak = 0
        if zero_score_streak_limit > 0 and zero_score_streak >= zero_score_streak_limit:
            early_stopped = True
            frame_count += 1
            break
        frame_count += 1

    total_time = time.time() - start_time
    avg_score = score / frame_count if frame_count > 0 else 0
    return params, avg_score, total_time, frame_count, early_stopped


def _load_frame_subset(data_path, num_frames_to_test):
    if data_path.lower().endswith((".mp4", ".mjpeg")):
        cap = cv2.VideoCapture(data_path)
        if not cap.isOpened():
            raise IOError("Cannot open video file: " + data_path)
        frames = []
        for _ in range(num_frames_to_test):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        source_mode = "video"
        image_files = None
    elif os.path.isdir(data_path):
        image_files = sorted(
            [
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        image_files = image_files[:num_frames_to_test]
        frames = [cv2.imread(path) for path in image_files]
        frames = [frame for frame in frames if frame is not None]
        source_mode = "images"
    else:
        raise ValueError("data_path must be a video file or directory of images")

    if not frames:
        raise ValueError("No frames were loaded for optimization")
    return frames, source_mode, image_files


def _build_best_detector(best_params):
    best_detector_params = cv2.aruco.DetectorParameters()
    best_detector_params.minMarkerPerimeterRate = best_params[0]
    best_detector_params.adaptiveThreshWinSizeMin = best_params[1]
    best_detector_params.adaptiveThreshWinSizeMax = best_params[2]
    best_detector_params.adaptiveThreshWinSizeStep = best_params[3]
    best_detector_params.polygonalApproxAccuracyRate = best_params[4]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.ArucoDetector(dictionary, best_detector_params)


def _write_tracked_output_video(
    data_path,
    source_mode,
    image_files,
    output_video_path,
    best_detector,
    use_gpu,
):
    if source_mode == "video":
        cap = cv2.VideoCapture(data_path)
        if not cap.isOpened():
            raise IOError("Cannot reopen video file: " + data_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 6
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        pbar = tqdm(total=total_frames, desc="Processing video with best parameters")
        frame_idx = 0
        tracking_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if use_gpu:
                try:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                    gray = gpu_gray.download()
                except Exception:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            marker_corners, marker_ids, _ = best_detector.detectMarkers(gray)
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            cv2.putText(
                frame,
                f"Frame: {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            out.write(frame)
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        out.release()
        total_tracking_time = time.time() - tracking_start
    else:
        if not image_files:
            raise ValueError("No image files found to create tracked output")
        first = cv2.imread(image_files[0])
        if first is None:
            raise ValueError("Could not read first image for tracked output")
        frame_height, frame_width, _ = first.shape
        fps = 6
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        pbar = tqdm(total=len(image_files), desc="Processing images with best parameters")
        frame_idx = 0
        tracking_start = time.time()
        for image_path in image_files:
            frame = cv2.imread(image_path)
            if frame is None:
                pbar.update(1)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = best_detector.detectMarkers(gray)
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            cv2.putText(
                frame,
                f"Frame: {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
        pbar.close()
        out.release()
        total_tracking_time = time.time() - tracking_start

    avg_tracking_time = total_tracking_time / frame_idx if frame_idx > 0 else 0
    return avg_tracking_time


def main():
    parser = argparse.ArgumentParser(
        description="Optimize ArUco detector parameters for a video or image directory."
    )
    parser.add_argument(
        "--data-path",
        default="bumblebox-17_2025-02-14_17_30_03.mp4",
        help="Path to an input .mp4/.mjpeg file or a folder of images.",
    )
    parser.add_argument(
        "--num-frames-to-test",
        type=int,
        default=50,
        help="Number of frames/images to use in parameter search.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of worker processes for grid search.",
    )
    parser.add_argument(
        "--output-video",
        default="tracked_output.mp4",
        help="Output path for tracked preview video.",
    )
    parser.add_argument(
        "--output-params",
        default="optimal_parameters.txt",
        help="Output path for optimized parameter report.",
    )
    parser.add_argument(
        "--param-mode",
        choices=["custom-grid", "aruco-defaults"],
        default="custom-grid",
        help=(
            "Parameter source: 'custom-grid' runs the script's search grid; "
            "'aruco-defaults' uses OpenCV ArUco defaults for all tuned parameters."
        ),
    )
    parser.add_argument(
        "--zero-score-streak-limit",
        type=int,
        default=0,
        help=(
            "If > 0, stop evaluating a parameter set early after this many "
            "consecutive zero-detection frames."
        ),
    )
    parser.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help=(
            "Parallel backend. 'thread' avoids heavy frame serialization and is "
            "recommended on macOS. 'process' can use more CPU but may be slower "
            "to start for large frame sets."
        ),
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help=(
            "Optional cap on number of parameter combinations to evaluate. "
            "Useful for quick tests."
        ),
    )
    parser.add_argument(
        "--shuffle-combinations",
        action="store_true",
        help="Shuffle parameter combinations before applying --max-combinations.",
    )
    args = parser.parse_args()

    if args.param_mode == "aruco-defaults":
        default_params = _get_aruco_default_params()
        min_marker_perimeter_rates = [default_params[0]]
        adaptive_thresh_win_size_min_values = [default_params[1]]
        adaptive_thresh_win_size_max_values = [default_params[2]]
        adaptive_thresh_win_size_step_values = [default_params[3]]
        polygonal_approx_accuracy_rate = [default_params[4]]
        print("Using OpenCV ArUco default parameters only:")
        print(
            f"  minMarkerPerimeterRate={default_params[0]}, "
            f"adaptiveThreshWinSizeMin={default_params[1]}, "
            f"adaptiveThreshWinSizeMax={default_params[2]}, "
            f"adaptiveThreshWinSizeStep={default_params[3]}, "
            f"polygonalApproxAccuracyRate={default_params[4]}"
        )
    else:
        min_marker_perimeter_rates = [0.02]
        adaptive_thresh_win_size_min_values = [3]
        adaptive_thresh_win_size_max_values = [500, 1000, 2000, 10000 ]
        adaptive_thresh_win_size_step_values = [2]
        polygonal_approx_accuracy_rate = [0.06]

    use_gpu = False
    data_path = args.data_path

    frames, source_mode, image_files = _load_frame_subset(data_path, args.num_frames_to_test)

    param_combinations = list(
        itertools.product(
            min_marker_perimeter_rates,
            adaptive_thresh_win_size_min_values,
            adaptive_thresh_win_size_max_values,
            adaptive_thresh_win_size_step_values,
            polygonal_approx_accuracy_rate,
        )
    )
    if args.shuffle_combinations:
        random.shuffle(param_combinations)
    if args.max_combinations is not None and args.max_combinations > 0:
        param_combinations = param_combinations[: args.max_combinations]

    print(
        f"Prepared {len(param_combinations)} parameter combinations "
        f"across {len(frames)} sampled frames using '{args.executor}' executor."
    )

    best_params = None
    best_score = -1
    best_eval_time = 0.0

    print("Starting parallel grid search over parameter combinations...")
    max_workers = min(max(1, args.max_workers), len(param_combinations))
    if args.executor == "thread":
        executor_cls = concurrent.futures.ThreadPoolExecutor
    else:
        executor_cls = concurrent.futures.ProcessPoolExecutor
    with executor_cls(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_params,
                params,
                frames,
                use_gpu,
                args.zero_score_streak_limit,
            ): params
            for params in param_combinations
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Parallel Grid Search",
        ):
            params, score, total_time, frame_count, early_stopped = future.result()
            suffix = " [early-stop]" if early_stopped else ""
            tqdm.write(
                f"Params: {params} -> Avg markers: {score:.2f}, "
                f"Frames: {frame_count}, Time: {total_time:.2f}s{suffix}"
            )
            if score > best_score:
                best_score = score
                best_params = params
                best_eval_time = total_time
            if best_params is not None:
                tqdm.write(
                    f"Current best -> Params: {best_params} | "
                    f"Avg markers: {best_score:.2f}, Time: {best_eval_time:.2f}s"
                )

    if best_params is None:
        raise RuntimeError("Parameter optimization failed to find any valid result")

    print("\nBest parameter combination found:")
    print(f"minMarkerPerimeterRate: {best_params[0]}")
    print(f"adaptiveThreshWinSizeMin: {best_params[1]}")
    print(f"adaptiveThreshWinSizeMax: {best_params[2]}")
    print(f"adaptiveThreshWinSizeStep: {best_params[3]}")
    print(f"polygonalApproxAccuracyRate: {best_params[4]}")
    print(f"Average markers detected: {best_score:.2f}")
    print(f"Total time: {best_eval_time:.2f}s")

    best_detector = _build_best_detector(best_params)
    avg_tracking_time = _write_tracked_output_video(
        data_path=data_path,
        source_mode=source_mode,
        image_files=image_files,
        output_video_path=args.output_video,
        best_detector=best_detector,
        use_gpu=use_gpu,
    )
    print(f"Tracking video saved to {args.output_video}")
    print(f"Average tracking time per frame: {avg_tracking_time:.4f} seconds")

    with open(args.output_params, "w") as f:
        f.write("Optimal ArUco Tracking Parameters:\n")
        f.write(f"minMarkerPerimeterRate: {best_params[0]}\n")
        f.write(f"adaptiveThreshWinSizeMin: {best_params[1]}\n")
        f.write(f"adaptiveThreshWinSizeMax: {best_params[2]}\n")
        f.write(f"adaptiveThreshWinSizeStep: {best_params[3]}\n")
        f.write(f"polygonalApproxAccuracyRate: {best_params[4]}\n")
        f.write(f"Average markers detected: {best_score:.2f}\n")
        f.write(f"Total time: {best_eval_time:.2f}s\n")
        f.write(f"Average tracking time per frame: {avg_tracking_time:.4f} seconds\n")
        f.write(f"Parameter mode: {args.param_mode}\n")
        f.write(f"Zero score streak limit: {args.zero_score_streak_limit}\n")
    print(f"Optimal parameters saved to {args.output_params}")


if __name__ == "__main__":
    main()
