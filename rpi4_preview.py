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
    
    if isinstance(preview_digital_zoom, tuple) and len(preview_digital_zoom) == 4:
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
    
    if isinstance(preview_digital_zoom, tuple) and len(preview_digital_zoom) == 4:
        picam2.set_controls({"ScalerCrop": preview_digital_zoom})
    elif preview_digital_zoom is not None:
        print("preview_digital_zoom is set incorrectly; expected None or a 4-item tuple.")
    
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
    parser = argparse.ArgumentParser(prog='Open a preview window for a Raspberry Pi-connected camera.')
    parser.add_argument('-t', '--preview_time', type=int, default=setup.preview_time, help='the preview time in seconds')
    parser.add_argument('-sh', '--shutter', type=int, default=setup.shutter_speed, help='the exposure time in microseconds')
    parser.add_argument('-w', '--width', type=int, default=setup.preview_width, help='preview width in pixels')
    parser.add_argument('-ht', '--height', type=int, default=setup.preview_height, help='preview height in pixels')
    parser.add_argument('-z', '--preview_digital_zoom', type=tuple, default=setup.preview_digital_zoom, help='optional digital zoom tuple (offset_x,offset_y,new_width,new_height)')
    parser.add_argument('-tf', '--preview_tuning_file', type=str, default=setup.preview_tuning_file, help='camera tuning file')
    parser.add_argument('-pw', '--preview_window', type=str, default=setup.preview_window, choices=['QTGL', 'QT', 'DRM'], help='preview window backend')
    args = parser.parse_args()

    print(f'preview time: {args.preview_time} seconds')
    print(f'shutter speed: {args.shutter} microseconds')
    print(f'preview width: {args.width} pixels')
    print(f'preview height: {args.height} pixels')
    print(f'tuning file used: {args.preview_tuning_file}')
    print(f'preview window being used: {args.preview_window}')

    rpi4_preview(
        args.preview_time,
        args.shutter,
        args.width,
        args.height,
        args.preview_digital_zoom,
        args.preview_tuning_file,
        args.preview_window,
    )
    
    
if __name__ == '__main__':
    
    main()
