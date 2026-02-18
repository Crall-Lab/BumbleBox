#!/usr/bin/env python
import os

'''BumbleBox settings! These settings will be exported to the other scripts.'''

'''Don't mess with this line, it just lets the computer know where this folder is stored'''
bumblebox_dir = os.getcwd()


'''This should be a number with two digits surrounded by quotes, ie 'XX' or "XX", and should match the number of the Raspberry Pi hostname, ie bumblebox-XX (if you don't know what a hostname is, you can name your computer, and we use the hostname to let us know which colony is which, especially when the Pis are connected to each other'''
colony_number = '01'


'''video recording / data capture settings'''
tag_tracking = True


'''only happens if tag_tracking is set to True! Record tag locations every x minutes for the given recording_time duration'''
tag_tracking_frequency = 5


'''in pixels (4056 is the max width for the HQ camera'''
width = 4056

'''in pixels (3040 is the max height for HQ camera)'''
height = 3040


'''in seconds, used whether tag_tracking is True or False'''
recording_time = 20


'''make a recording every X minutes'''
recording_frequency = 30


track_recorded_videos = True


'''in microseconds'''
shutter_speed = 2500


'''fps in seconds'''
frames_per_second = 5
actual_frames_per_second = 4.4


''' the options are 'mp4' or 'mjpeg' '''
codec = 'mp4'

'''this takes values 0-100, but is only relevant for mjpeg video recording'''
quality = 95

infrared_recording = True

if infrared_recording == True:
	tuning_file = 'imx477_noir.json' # algorithm that runs based on the camera type (this automatically turns on 

elif infrared_recording == False:
	tuning_file = 'imx477.json'

data_folder_path = '/mnt/bumblebox/data'

'''This takes options 'Auto', 'HighQuality', 'Fast', or 'Off'. Would recommend using 'Auto' to start off - using HighQuality will impact the max framerate possible'''
noise_reduction_mode = 'Auto' 

'''takes the form (x,y,w,h), for example: (1000,1000,500,500) would record a square of 500x500 pixels that are offset 1000 pixels from the left of the image, and 1000 pixels down from the top of the image'''
recording_digital_zoom = None

''' ArUco settings '''

tag_dictionary = '4X4_50'

'''the options are None, 'custom', or 'koppert' - set this to either custom or koppert to access preset tracking settings for those boxes'''
box_type = 'custom'

''' Interpolates data between missing frames if set to True. For example, if a bee is tracked in frames 2,3, and 6, this would draw a line
between the bee's position at frames 3 and 6, and fill in its position along that line for frames 4 and 5.
max_seconds_gap sets the seconds threshold after which the data will not be interpolated between two positions. 
This is useful because if a bee is tracked in frame 1 and frame 50, you wouldnt want to assume it walked in a straight line between these two frames'''

interpolate_data = True
max_seconds_gap = 3


calculate_behavior_metrics = True
behavior_metrics = ['speed', 'activity', 'distance from center', 'pairwise distance', 'contacts', 'video averages']
contact_distance = 2.0 #in centimeters


if box_type == 'custom':
	pixel_contact_distance = 206.1
	moving_threshold = 3.16 #moving 3.16 pixels / frame seems to be the cutoff between noise/non-moving bee and a moving bee (see bimodal distribution of log10 of the speed scores) 
	#add pcd for koppert box


'''preview settings'''

preview_time = 120 #seconds

shutter_speed = 2500 #microseconds

preview_width = 3000 #pixels (these should match your camera's aspect ratio - the HQ camera (imx477 sensor) has about a 4:3 aspect ratio, so to obtain a correct-looking preview, keep those dimensions 

preview_height = 2250 #pixels

infrared_preview = True #if set to true, uses an algorithm for the HQ camera sensor that favors infrared lighting

preview_window = 'QTGL' #either 'QTGL', 'QT', or for lite operating systems 'DRM'

preview_digital_zoom = (275,0,3300,3040)

if infrared_preview == True:
	preview_tuning_file = 'imx477_noir.json' # algorithm that runs based on the camera type (this automatically turns on 

elif infrared_preview == False:
	preview_tuning_file = 'imx477.json'


'''Composite nest image for brood labelling - creation settings. Images are automatically set to generate at 11pm when create_composite_nest_images = True'''
create_composite_nest_images = True

number_of_images = 60 #the number of images that will be used from today's data folder to create the image (if there are fewer than the number listed here, it will use all images)

composite_images_per_day = 1 #Needs to be 1 for now

