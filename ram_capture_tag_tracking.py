#!/usr/bin/env python

from picamera2 import Picamera2, Preview
import cv2
from cv2 import aruco
import pandas as pd
import argparse
from record_video import create_todays_folder
from setup import colony_number
import socket
import time
from datetime import date
from datetime import datetime
import sys
from sys import getsizeof
from libcamera import controls
import os
import behavioral_metrics
import setup
import logging
from data_cleaning import interpolate
import pwd

username = pwd.getpwuid(os.getuid())[0]
logging.basicConfig(filename=f'/home/{username}/Desktop/BumbleBox/logs/log.log',encoding='utf-8',format='%(filename)s %(asctime)s: %(message)s', filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# to do - 
# transfer tag tracking code to video recording so it happens after videos record as well
# figure out how to fine tune frames per second

def array_capture(recording_time, fps, shutter_speed, width, height, tuning_file, noise_reduction_mode, digital_zoom):
    
    '''load tuning file and initialize the camera (setting the format and the image size)'''
    tuning = Picamera2.load_tuning_file(tuning_file)
    picam2 = Picamera2(tuning=tuning)
    preview = picam2.create_preview_configuration({"format": "YUV420", "size": (width,height)})
    picam2.align_configuration(preview) #might cause an issue?
    picam2.configure(preview)
    
    '''set shutterspeed (or exposure time)'''
    picam2.set_controls({"ExposureTime": shutter_speed}) #"NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Fast})
    
    '''set noise reduction mode'''
    if noise_reduction_mode != "Auto":
        try:
            picam2.set_controls({"NoiseReductionMode": noise_reduction_mode})
        except:
            print("The variable 'noise_reduction_mode' in the setup.py script is set incorrectly. Please change it and save that script. It should be 'Auto', 'Off', 'Fast', or 'HighQuality'")
    
    '''set digital zoom'''
    if digital_zoom == type(tuple) and len(digital_zoom) == 4:
        picam2.set_controls({"ScalerCrop": digital_zoom})
    
    elif digital_zoom != None:
        print("The variable 'recording_digital_zoom' in the setup.py script is set incorrectly. It should be either 'None' or a value that looks like this: (offset_x,offset_y,new_width,new_height) for ex. (1000,2000,300,300)")
    
    '''start the camera'''
    picam2.start()
    time.sleep(2)
    
    print("Initializing data capture...")
    print("Recording parameters:\n")
    print(f"recording time: {recording_time}s")
    print(f"frames per second: {fps}")
    print(f"image width: {width} pixels")
    print(f"image width: {height} pixels")
    print(f"shutter speed: {shutter_speed} microseconds")
    
    start_time = time.time()
    
    frames_list = []
    i = 0
    
    while ( (time.time() - start_time) < recording_time):
        
        timestamp = time.time() - start_time
        print(f'frame {i} timestamp: {round(timestamp,2)} seconds')
        yuv420 = picam2.capture_array()
        frames_list.append([yuv420])
        #yuv420 = yuv420[0:3040, :]
        #frames_dict[f"frame_{i:03d}"] = [yuv420, timestamp]
        time.sleep(1/(fps+1))
        i += 1
    
    finished = time.time()-start_time
    print(f'finished capturing frames to arrays, captured {i} frames in {round(finished,2)} seconds')
    rate = i / finished
    print(f'\nthats {round(rate,2)} frames per second!\n\n\nMake sure this corresponds well to your desired framerate and write this number into the actual_frame_rate setting in settings.py. FPS is a bit experimental for tag tracking and mp4 recording at the moment... Thats the tradeoff for allowing a higher framerate.')
    sizeof = getsizeof(frames_list)
    
    return frames_list
        
        
        
def trackTagsFromRAM(filename, todays_folder_path, frames_list, tag_dictionary, box_type, now, hostname, colony_number):

    if tag_dictionary is None:
        tag_dictionary = '4X4_50'
        if isinstance(tag_dictionary, str):
            if 'DICT' not in tag_dictionary:
                tag_dictionary = "DICT_%s" % tag_dictionary
            tag_dictionary = tag_dictionary.upper()
            if not hasattr(cv2.aruco, tag_dictionary):
                raise ValueError("Unknown tag dictionary: %s" % tag_dictionary)
            tag_dictionary = getattr(cv2.aruco, tag_dictionary)
    else:
        if 'DICT' not in tag_dictionary:
            tag_dictionary = "DICT_%s" % tag_dictionary
        tag_dictionary = tag_dictionary.upper()
        if not hasattr(cv2.aruco, tag_dictionary):
            raise ValueError("Unknown tag dictionary: %s" % tag_dictionary)
        tag_dictionary = getattr(cv2.aruco, tag_dictionary)
    tag_dictionary = aruco.getPredefinedDictionary(tag_dictionary) 
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(tag_dictionary, parameters)
    
    if box_type=='custom':
        parameters.minMarkerPerimeterRate=0.03
        parameters.adaptiveThreshWinSizeMin=5
        parameters.adaptiveThreshWinSizeStep=6
        parameters.polygonalApproxAccuracyRate=0.06
        
    elif box_type=='koppert':
        #change these!
        parameters.minMarkerPerimeterRate=0.03
        parameters.adaptiveThreshWinSizeMin=5
        parameters.adaptiveThreshWinSizeStep=6
        parameters.polygonalApproxAccuracyRate=0.06
        
    elif box_type==None:
        pass
        #parameters.minMarkerPerimeterRate=0.03
        #parameters.adaptiveThreshWinSizeMin=5
        #parameters.adaptiveThreshWinSizeStep=6
        #parameters.polygonalApproxAccuracyRate=0.06

    frame_num = 0
    noID = []
    raw = []
    augs_csv = []
    
    start = time.time()

    for index, frame in enumerate(frames_list):
        
        frame = frame[0]
        
        if index == int(len(frames_list) / 2 ):
            #logging.debug(f"{todays_folder_path}{filename}.png'")
            print(f"{todays_folder_path}/{filename}.png'")
            frame_to_write = frame.copy()
            gray = cv2.cvtColor(frame_to_write, cv2.COLOR_YUV2GRAY_I420)
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            #cl1 = clahe.apply(gray)
            #gray = cv2.cvtColor(cl1,cv2.COLOR_GRAY2RGB) #needlessly converts to 3d array from 2d array, less data when sticking with the 2d array
            cv2.imwrite(todays_folder_path + "/" + filename + '.png', gray)
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_I420)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(gray)
            gray = cv2.cvtColor(cl1,cv2.COLOR_GRAY2RGB)
            
        except:
            print('converting to grayscale didnt work...')
            continue
        
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        
        #for troubleshooting
        #frame_markers = aruco.drawDetectedMarkers(gray.copy(), corners, ids)
        #resized = cv2.resize(frame_markers, (1352,1013), interpolation = cv2.INTER_AREA)
        #cv2.imshow("frame",resized)
        #cv2.waitKey(5000)

        for i in range(len(rejectedImgPoints)):
            c = rejectedImgPoints[i][0]
            xmean = c[:,0].mean() #for calculating the centroid
            ymean = c[:,1].mean() #for calculating the centroid
            xmean_top_point = (c[0,0] + c[1,0]) / 2 #for calculating the top point of the tag
            ymean_top_point = (c[0,1] + c[1,1]) / 2 #for calculating the top point of the tag
            noID.append( [filename, colony_number, now, frame_num, "X", float(xmean), float(ymean), float(xmean_top_point), float(ymean_top_point)] )
            
        if ids is not None:
            for i in range(len(ids)):
                c = corners[i][0]
                xmean = c[:,0].mean() #for calculating the centroid
                ymean = c[:,1].mean() #for calculating the centroid
                xmean_top_point = (c[0,0] + c[1,0]) / 2 #for calculating the top point of the tag
                ymean_top_point = (c[0,1] + c[1,1]) / 2 #for calculating the top point of the tag
                raw.append( [filename, colony_number, now, frame_num, int(ids[i]), float(xmean), float(ymean), float(xmean_top_point), float(ymean_top_point)] )
        frame_num += 1
        print(f"processed frame {index}")  
    
    
    try:
        df = pd.DataFrame(raw)
        df = df.rename(columns = {0:'filename', 1:'colony number', 2:'datetime', 3:'frame', 4:'ID', 5:'centroidX', 6:'centroidY', 7:'frontX', 8:'frontY'})
        df.to_csv(todays_folder_path + "/" + filename + '_raw.csv', index=False)
        print(f'saved raw csv to {todays_folder_path}{filename}_raw.csv')
    except Exception as e:
        logger.exception("Exception occurred: %s", str(e))
        
    try:
        df2 = pd.DataFrame(noID)
        df2 = df2.rename(columns = {0:'filename', 1:'colony number', 2:'datetime', 3:'frame', 4:'ID', 5:'centroidX', 6:'centroidY', 7:'frontX', 8:'frontY'})
        df2.to_csv(todays_folder_path + "/" + filename + '_noID.csv', index=False)
        print(f'saved noID csv to {todays_folder_path}{filename}_noID.csv')
    except Exception as e:
        logger.exception("Exception occurred: %s", str(e))
        
    

    print("Average number of tags found: " + str(len(df.index)/frame_num))
    tracking_time = time.time() - start
    print(f"Tag tracking took {round(tracking_time,2)} seconds, an average of {round(tracking_time / frame_num,2)} seconds per frame") 
    
    if df.empty == True:
        logger.warning("df is empty")
        
    if df2.empty == True:
        logger.warning("df2 is empty")
    
    return df, df2, frame_num
    
    
def main():
    
    #check whether this script is being run via cron or in terminal/via Geany or some other text editor
    if sys.stdout.isatty():
        print("Running tag tracking script from terminal")
        logger.debug("Running tag tracking script from terminal")
    else:
        logger.debug("Running tag tracking script via crontab")
        print("Running tag tracking script via crontab")
    
    if setup.create_composite_nest_images == True:
        
        gen_im_time = datetime(1970, 1, 1, 23, 0, 0)
        now = datetime.now()
        
        if gen_im_time.hour == now.hour and gen_im_time.minute == now.minute:
            logger.debug("Exiting because the generate_nest_images.py script is running now")
            return print("ending now because the image generation function should be running")
        
        
    parser = argparse.ArgumentParser(prog='Record a video, either an mp4 or mjpeg video! Program defaults to mp4 currently.')
    parser.add_argument('-p', '--data_folder_path', type=str, default=setup.data_folder_path, help='a path to the folder you want to collect data in. Default is /mnt/bumblebox/data')
    parser.add_argument('-t', '--recording_time', type=int, default=setup.recording_time, help='the video recording time in seconds')
    parser.add_argument('-fps', '--frames_per_second', type=int, default=setup.frames_per_second, choices=range(0,11), help='the number of frames recorded per second of video capture. At the moment this is still a bit experimental, we have gotten up to 6fps to work for mjpeg, and up to 10fps for mp4 videos.')
    parser.add_argument('-afps', '--actual_frames_per_second', type=float, default=setup.actual_frames_per_second, help='the number of frames recorded per second of video capture. At the moment this is still a bit experimental, we have gotten up to 6fps to work for mjpeg, and up to 10fps for mp4 videos.')
    parser.add_argument('-sh', '--shutter', type=int, default=setup.shutter_speed, help='the exposure time, or shutter speed, of the camera in microseconds (1,000,000 microseconds in a second!!)')
    parser.add_argument('-w', '--width', type=int, default=setup.width, help='the width of the image in pixels')
    parser.add_argument('-ht', '--height', type=int, default=setup.height, help='the height of the image in pixels')
    parser.add_argument('-d', '--dictionary', type=str, default=setup.tag_dictionary, help='type "aruco.DICT_" followed by the size of the tag youre using (either 4X4 (this is the default), 5X5, 6X6, or 7X7) and the number of tags in the dictionary (either 50 (also the default), 100, 250, or 1000).')
    parser.add_argument('-tf', '--tuning_file', type=str, default=setup.tuning_file, help='this is a file that helps improve image quality by running algorithms tailored to particular camera sensors.\nBecause the BumbleBox by default images in IR, we use the \'imx477_noir.json\' file by default')
    parser.add_argument('-b', '--box_type', type=str, default=setup.box_type, choices=['custom','koppert'], help='an option to choose a default set of tracking parameters for either the custom bumblebox or the koppert box adaptation')
    parser.add_argument('-nr', '--noise_reduction', type=str, default=setup.noise_reduction_mode, choices=['Auto', 'Off', 'Fast', 'HighQuality'], help='an option to "digitally zoom in" by just recording from a portion of the sensor. This overrides height and width values, and can be useful to crop out glare that is negatively impacting image quality. Takes 4 values inside parentheses, separated by commas: 1: number of pixels to offset from the left side of the image 2: number of pixels to offset from the top of the image 3: width of the new cropped frame in pixels 4: height of the new cropped frame in pixels')
    parser.add_argument('-z', '--digital_zoom', type=tuple, default=setup.recording_digital_zoom, help='an option to "digitally zoom in" by just recording from a portion of the sensor. This overrides height and width values, and can be useful to crop out glare that is negatively impacting image quality. Takes 4 values inside parentheses, separated by commas: 1: number of pixels to offset from the left side of the image 2: number of pixels to offset from the top of the image 3: width of the new cropped frame in pixels 4: height of the new cropped frame in pixels')
    args = parser.parse_args()
    
    ret, todays_folder_path = create_todays_folder(args.data_folder_path)
    if ret == 1:
        print("Couldn't create todays folder to store data in. Exiting program. Sad!")
        return 1
        
    hostname = socket.gethostname()
    now = datetime.now()
    now = now.strftime('%Y-%m-%d_%H-%M-%S')
    filename = hostname + '_' + now

    print(filename)
    print(args.data_folder_path)
    print(todays_folder_path)
    print(args.recording_time)
    print(args.frames_per_second)

    frames_list = array_capture(args.recording_time, args.frames_per_second, args.shutter, args.width, args.height, args.tuning_file, args.noise_reduction, args.digital_zoom)

    print('about to track tags!')
    df, df2, frame_num = trackTagsFromRAM(filename, todays_folder_path, frames_list, args.dictionary, args.box_type, now, hostname, colony_number)
    
    if setup.interpolate_data == True and df.empty == False:
        df = interpolate(df, setup.max_seconds_gap, setup.actual_frames_per_second)
    
    if df.empty == False and setup.calculate_behavior_metrics == True:
        
        behavioral_metrics.calculate_behavior_metrics(df, setup.actual_frames_per_second, todays_folder_path, filename)
        print("Calculating behavior metrics")
        
        
        
        
if __name__ == '__main__':
    
    main()
    logging.shutdown()
