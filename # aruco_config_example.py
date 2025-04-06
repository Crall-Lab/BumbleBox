# aruco_config_example.py

# This file defines a custom ArUco DetectorParameters dictionary
# You can import this into your tracking script or load it dynamically.

aruco_params = {
    "adaptiveThreshWinSizeMin": 5,
    "adaptiveThreshWinSizeStep": 4,
    "adaptiveThreshWinSizeMax": 23,
    "adaptiveThreshConstant": 7,
    "minMarkerPerimeterRate": 0.03,
    "maxMarkerPerimeterRate": 4.0,
    "polygonalApproxAccuracyRate": 0.06,
    "minCornerDistanceRate": 0.05,
    "minDistanceToBorder": 3,
    "cornerRefinementMethod": 1  # 0=none, 1=subpix, 2=contour
}

# Example usage in your script:
# Import the dictionary and pass it to the tracking function like below:

# from aruco_config_example import aruco_params
# track_tags_from_video('path', 'outdir', 'vidname', '4X4_50', box_type=None, aruco_params=aruco_params)
