#!/usr/bin/env python
import os

'''BumbleBox settings! These settings will be exported to the other scripts.'''

'''Don't mess with this line, it just lets the computer know where this folder is stored'''
bumblebox_dir = os.getcwd()

'''video recording / data capture settings'''
tag_tracking = True

'''only happens if tag_tracking is set to True! Record tag locations every x minutes for the given recording_time duration'''
tag_tracking_frequency = 2

'''in pixels (4056 is the max width for the HQ camera'''
width = 4056

'''in pixels (3040 is the max height for HQ camera)'''
height = 3040

'''in seconds'''
recording_time = 20

'''make a recording every X minutes'''
recording_frequency = 30

'''in microseconds'''
shutter_speed = 2500

'''in seconds'''
frames_per_second = 6

''' the options are 'mp4' or 'mjpeg' '''
codec = 'mp4'

'''this takes values 0-100, but is only relevant for mjpeg video recording'''
quality = 95

infrared_recording = True

data_folder_path = '/mnt/bumblebox/data'

'''This takes options 'Auto', 'HighQuality', 'Fast', or 'Off'. Would recommend using 'Auto' to start off - using HighQuality will impact the max framerate possible'''
noise_reduction_mode = 'Auto' 

'''takes the form (x,y,w,h), for example: (1000,1000,500,500) would record a square of 500x500 pixels that are offset 1000 pixels from the left of the image, and 1000 pixels down from the top of the image'''
recording_digital_zoom = None

''' ArUco settings '''

tag_dictionary = '4X4_50'

'''the options are None, 'custom', or 'koppert' - set this to either custom or koppert to access preset tracking settings for'''
box_type = 'custom'

#parameters




''' preview settings '''

preview_time = 30 #seconds
shutter_speed = 2500 #microseconds
preview_width = 2028 #pixels (these should match your camera's aspect ratio - the HQ camera (imx477 sensor) has about a 4:3 aspect ratio, so to obtain a correct-looking preview, keep those dimensions 
preview_height = 1520 #pixels
infrared_preview = True #if set to true, uses an algorithm for the HQ camera sensor that favors infrared lighting
preview_window = 'QTGL' #either 'QTGL', 'QT', or for lite operating systems 'DRM'
preview_digital_zoom = None

if infrared_preview == True:
	tuning_file = 'imx477_noir.json' # algorithm that runs based on the camera type (this automatically turns on 

elif infrared_preview == False:
	tuning_file = 'imx477.json'
 



# to do:
# zoom - still gotta implement in both recording and in preview
# aruco parameters
# best ssh preview?
# maybe its better to just choose None for preview width and height?