import cv2
import numpy as np
import itertools
import time
from tqdm import tqdm
import os
import concurrent.futures

#User input is required at the top of the main function, first and foremost for the value of each of the parameters for each of the

#for the data_path, and the number of frames to test
#The data_path can be a video file or a directory containing images
#User input is required for the number of frames to test
#If the data is a folder of images, the output fps is set to 6
#If the data is a video file, the output fps is set to the fps of the video
#The output video is saved as tracked_output.mp4
#The optimal parameters are saved to optimal_parameters.txt
#The optimal parameters file is saved in the same directory as the script
#The output video is saved in the same directory as the script
def evaluate_params(params, frames, use_gpu):
    """
    Evaluate a single parameter combination on a list of frames.
    Returns a tuple of (params, average marker detection count).
    """
    # Create a DetectorParameters object with current combination.
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.minMarkerPerimeterRate = params[0]
    detector_params.adaptiveThreshWinSizeMin = params[1]
    detector_params.adaptiveThreshWinSizeMax = params[2]
    detector_params.adaptiveThreshWinSizeStep = params[3]
    detector_params.polygonalApproxAccuracyRate = params[4]
    
    # Create the ArUco dictionary (using the new API)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # Instantiate the detector.
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    
    score = 0
    frame_count = 0
    start_time = time.time()
    for frame in frames:
        # Convert frame to grayscale; use GPU if enabled.
        if use_gpu:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                gray = gpu_gray.download()
            except Exception as e:
                # Fallback to CPU if GPU fails in a process.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        # Detect markers using the new API.
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)
        if markerIds is not None:
            score += len(markerIds)
        frame_count += 1
    total_time = time.time() - start_time
    avg_score = score / frame_count if frame_count > 0 else 0
    return params, avg_score, total_time

def main():
    # --- Parameter Grid Setup ---
    # Ranges selected because the tags are small in a high-res frame.
    min_marker_perimeter_rates = [0.02]
    adaptive_thresh_win_size_min_values = [5]
    adaptive_thresh_win_size_max_values = [29]
    adaptive_thresh_win_size_step_values = [3]
    polygonal_approx_accuracy_rate= [0.04, 0.05, 0.06, 0.07, 0.08]
    
    # Decide whether to use GPU for grayscale conversion.
    # (If using many processes, consider setting this to False if you encounter GPU contention.)
    use_gpu = False # cv2.cuda.getCudaEnabledDeviceCount() > 0
    
    # --- Load a subset of frames for grid search ---
    data_path = "bumblebox-17_2025-02-14_17_30_03.mp4"
    num_frames_to_test = 50

    if data_path.endswith(".mp4") or data_path.endswith(".mjpeg"):
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

    elif os.path.isdir(data_path):
        image_files = [f for f in os.listdir(data_path) if f.endswith(".png")]
        image_files = image_files[:num_frames_to_test]
        frames = [cv2.imread(os.path.join(data_path, f)) for f in image_files]

    
    # Build all parameter combinations.
    param_combinations = list(itertools.product(min_marker_perimeter_rates,
                                                  adaptive_thresh_win_size_min_values,
                                                  adaptive_thresh_win_size_max_values,
                                                  adaptive_thresh_win_size_step_values,
                                                  polygonal_approx_accuracy_rate))
    
    best_params = None
    best_score = -1
    total_time = None

    print("Starting parallel grid search over parameter combinations...")
    # Parallelize grid search over available CPU cores.
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(evaluate_params, params, frames, use_gpu): params 
                   for params in param_combinations}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc="Parallel Grid Search"):
            params, score, total_time = future.result()
            tqdm.write(f"Params: {params} -> Avg markers: {score:.2f}, Time: {total_time:.2f}s")
            if score > best_score:
                best_score = score
                best_params = params
                track_time = total_time

    print("\nBest parameter combination found:")
    print(f"minMarkerPerimeterRate: {best_params[0]}")
    print(f"adaptiveThreshWinSizeMin: {best_params[1]}")
    print(f"adaptiveThreshWinSizeMax: {best_params[2]}")
    print(f"adaptiveThreshWinSizeStep: {best_params[3]}")
    print(f"polygonalApproxAccuracyRate: {best_params[4]}")
    print(f"Average markers detected: {best_score:.2f}")
    print(f"Total time: {track_time:.2f}s")

    # --- Save the optimal parameters to a text file ---
    optimal_params_file = "optimal_parameters.txt"
    with open(optimal_params_file, "w") as f:
        f.write("Optimal ArUco Tracking Parameters:\n")
        f.write(f"minMarkerPerimeterRate: {best_params[0]}\n")
        f.write(f"adaptiveThreshWinSizeMin: {best_params[1]}\n")
        f.write(f"adaptiveThreshWinSizeMax: {best_params[2]}\n")
        f.write(f"adaptiveThreshWinSizeStep: {best_params[3]}\n")
        f.write(f"polygonalApproxAccuracyRate: {best_params[4]}\n")
        f.write(f"Average markers detected: {best_score:.2f}\n")
        f.write(f"Total time: {track_time:.2f}s\n")
    print(f"Optimal parameters saved to {optimal_params_file}")

    # --- Create a video showing the tracking using the best parameters ---
    # Reopen the video for full processing.
    if data_path[-1:-4] == ".mp4":
        cap = cv2.VideoCapture(data_path)
        if not cap.isOpened():
            raise IOError("Cannot reopen video file: " + data_path)
        
        # Prepare VideoWriter for output.
        output_video_path = "tracked_output.mp4"
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    
    else:
        img = cv2.imread(frames[0])
        frame_height, frame_width, _ = img.shape
        fps = 6
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("tracked_output.mp4", fourcc, fps, (frame_width, frame_height))

    
    # Create detector parameters with the best settings.
    best_detector_params = cv2.aruco.DetectorParameters()
    best_detector_params.minMarkerPerimeterRate = best_params[0]
    best_detector_params.adaptiveThreshWinSizeMin = best_params[1]
    best_detector_params.adaptiveThreshWinSizeMax = best_params[2]
    best_detector_params.adaptiveThreshWinSizeStep = best_params[3]
    best_detector_params.polygonalApproxAccuracyRate = best_params[4]
    # Create the detector using the new API.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    best_detector = cv2.aruco.ArucoDetector(dictionary, best_detector_params)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing video with best parameters")
    frame_idx = 0

    # Start timing the tracking process.
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
            except Exception as e:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        markerCorners, markerIds, rejectedCandidates = best_detector.detectMarkers(gray)
        if markerIds is not None:
            cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    total_tracking_time = time.time() - tracking_start
    avg_tracking_time = total_tracking_time / frame_idx if frame_idx > 0 else 0
    pbar.close()
    print(f"Tracking video saved to {output_video_path}")
    print(f"Average tracking time per frame: {avg_tracking_time:.4f} seconds")

    # Append the average tracking time per frame to the optimal parameters file.
    with open(optimal_params_file, "a") as f:
        f.write(f"Average tracking time per frame: {avg_tracking_time:.4f} seconds\n")

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
