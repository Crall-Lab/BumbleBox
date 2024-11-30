#!/usr/bin/env python

#switch from using environment variables to pulling variables from setup.py
#figure out how to both run from command line and using the variables set by setup.py
#testing the commit functionality in VS code!

from picamera2 import Picamera2, Preview
import time
import argparse
import os
import setup

def rpi4_preview(preview_time, shutter_speed, width, height, preview_digital_zoom, preview_tuning_file, preview_window):
	
	tuning = Picamera2.load_tuning_file(preview_tuning_file)
	
	picam2 = Picamera2(tuning=tuning)
	
	if type(preview_digital_zoom) == tuple and len(preview_digital_zoom) == 4:
		width = preview_digital_zoom[2]
		height = preview_digital_zoom[3]
		camera_config = picam2.create_preview_configuration({"size": (width, height)})
	else:
		print("No digital zoom detected (or its set incorrectly), now setting the width and height")
		camera_config = picam2.create_preview_configuration({"size": (width, height)})
	
	picam2.align_configuration(camera_config)
	picam2.configure(camera_config)
	print('\n\n\n')
	print('Here are the current camera configuration settings:\n')
	print(camera_config)
	print('\n\n')
	picam2.set_controls({"ExposureTime": shutter_speed})
	print(f"Setting shutter speed to {shutter_speed}")
	
	if preview_digital_zoom != None:
		picam2.set_controls({"ScalerCrop": preview_digital_zoom})
	
	if preview_window == 'QTGL':
		picam2.start_preview(Preview.QTGL)
		picam2.start()
		
	elif preview_window == 'QT':
		picam2.start_preview(Preview.QT)
		picam2.start()
		
	elif preview_window == 'DRM':
		print("Fyi, QTGL is the default preview window, and the DRM preview window is usually meant for Raspberry Pi lite operating systems.\nI havent gotten it to work on the Raspberry Pi 4b... Here goes nothing!")
		picam2.start_preview(Preview.DRM)
		picam2.start()
		
	time.sleep(preview_time)
		
	picam2.stop_preview()
	picam2.stop()


def main():
	
	try:
		#preview_time = os.environ["PREVIEW_TIME"]
		preview_time = setup.preview_time
		print(f'preview time: {preview_time} seconds')
		#shutter_speed = os.environ["SHUTTER_SPEED"]
		shutter_speed = setup.shutter_speed
		print(f'shutter speed: {shutter_speed} microseconds ')
		#preview_width = os.environ["PREVIEW_WIDTH"]
		preview_width = setup.preview_width
		print(f'preview width: {preview_width} pixels')
		#preview_height = os.environ["PREVIEW_HEIGHT"]
		preview_height = setup.preview_height
		print(f'preview height: {preview_height} pixels')
		preview_digital_zoom = setup.preview_digital_zoom
		#tuning_file = os.environ["TUNING_FILE"]
		preview_tuning_file = setup.preview_tuning_file
		print(f'tuning file used: {preview_tuning_file}')
		print("if the name of the tuning file has \"noir\" in it, that means its calibrating for you having taken out the IR filter in the camera. So hopefully you did that, or else it might look funky!") 
		#preview_window = os.environ["PREVIEW_WINDOW"]
		preview_window = setup.preview_window
		print(f'preview window being used: {preview_window}')
		
		rpi4_preview(preview_time, shutter_speed, preview_width, preview_height, preview_digital_zoom, preview_tuning_file, preview_window)
		
	except KeyError:
		
		print("Okay, no variables have been set yet, Im gonna check if this script is being run via the command line with inputs.")
		
		try:
			parser = argparse.ArgumentParser(prog='Open a preview window for a Raspberry Pi - connected camera.')
			parser.add_argument('-t', '--preview_time', type=int, help='the video recording time in seconds. There is no default value for this, so this is required input in order for the script to run.')
			parser.add_argument('-sh', '--shutter', type=int, default=2500, help='the exposure time, or shutter speed, of the camera in microseconds (1,000,000 microseconds in a second!!)')
			parser.add_argument('-w', '--width', type=int, default=1352, help='the width of the preview image in pixels (you should be able to adjust the window size after it opens though')
			parser.add_argument('-ht', '--height', type=int, default=1013, help='the height of the preview image in pixels')
			parser.add_argument('-z', '--preview_digital_zoom', type=tuple, default=None, help='an option to "digitally zoom in" by just recording from a portion of the sensor. This overrides height and width values, and can be useful to crop out glare that is negatively impacting image quality. Takes 4 values inside parentheses, separated by commas: 1: number of pixels to offset from the left side of the image 2: number of pixels to offset from the top of the image 3: width of the new cropped frame in pixels 4: height of the new cropped frame in pixels')
			parser.add_argument('-tf', '--preview_tuning_file', type=str, default='imx477_noir.json', help='this is a file that helps improve image quality by running algorithms tailored to particular camera sensors.\nBecause the BumbleBox by default images in IR, we use the \'imx477_noir.json\' file by default')
			args = parser.parse_args()
			
			print("Yes it is!")
			print(f'preview time: {args.preview_time} seconds')
			print(f'shutter speed: {args.shutter} microseconds')
			print(f'preview width: {args.width} pixels')
			print(f'preview height: {args.height} pixels')
			print(f'tuning file used: {args.tuning_file}\n')
			
			rpi4_preview(args.preview_time, args.shutter, args.width, args.height, args.preview_digital_zoom, args.preview_tuning_file)
			
		except:
			
			print("You haven't set any variables for this preview script yet (by running the setup script)")
			print("Im going to run the camera preview with default values, shown below.\n")
			print("To exit the preview, first click on this window (the one youre reading this text in), then you should press and hold the control button and then press the C button.\n")
			
			print(f'preview time: 30 seconds')
			print(f'shutter speed: 2500 microseconds ')
			print(f'preview width: 1352 pixels')
			print(f'preview height: 1013 pixels')
			print(f'tuning file used: "imx477_noir.json"\n')
			 
			rpi4_preview(30, 2500, 1352, 1013, "imx477_noir.json", "QTGL")
	
	
if __name__ == '__main__':
	
	main()
