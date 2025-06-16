#!/usr/bin/env python3

"""Label nest components in bumblebee nest."""

__appname__ = 'labelnests.py'
__author__ = 'August Easton-Calabria (eastoncalabr@wisc.edu), 206-919-8233'
__version__ = '0.0.1'

import os
import sys
import csv
import subprocess
import glob
import json
import numpy as np
import pandas as pd
#import skvideo.io
#import io
#from PIL import Image as pil_im
#import tkinter as tk
#from tkinter import filedialog
#import setup


def labelNest(directory):
    #with this code, have set up my own labelmerc file in the BumbleBox folder with config settings.
    #also went in and changed the __init__ file in the config folder, hashing out this code: (in order to have labels with repeating words)
    #if key == "labels" and value is not None and len(value) != len(set(value)):
    #    raise ValueError(
    #        "Duplicates are detected for config key 'labels': {}".format(value)
    #    )
    
    print('Running Labelme now... I have preloaded the labels to use for these images. Of course, you dont need to use labels that dont apply to your circumstances. Use the designated shapes for each label that is provided. For example, in order to draw the perimeter of the nest, Ill use the polygon tool, because the label says Nest perimeter (polygon). Right click to choose what types of shapes you want to use!')
    print(directory)
    print(type(directory))

    bumblebox_dir = os.path.dirname(os.path.realpath(__file__))
    print(bumblebox_dir)
    
    subprocess.run(['labelme', bumblebox_dir, '--config', bumblebox_dir + '/labelmerc', '--output', directory + '/Labelled Nest Files' ])
    #subprocess.run(['labelme', dir, '--labels', 'Arena perimeter (polygon),Nest perimeter (polygon),Eggs perimeter (polygons),Eggs (points),Larvae (circles),Pupae (circles),Queen larva (circles),Queen pupae,Wax pots (circles),full nectar pot (circles),empty wax pots (circles),pollen balls (circles), nectar source (circle)'])

    #subprocess.run(['labelme', dir, '--labels', 'nest perimeter,eggs (perimeter),eggs (circles),larvae,pupae,queen cell,wax pot,full nectar pot,empty wax pot,pollen ball, nectar source',
    #                '--shapes', 'polygon,polygon,points,cirlce,circle,circle,circle,circle,circle,circle,circle,circle', '--colors', 'yellow,orange,red,green,blue,purple,turquoise,magenta,pink'
    #                ]) #edit this line to add colours
    print('huh... is this on?')
    jsons = sorted(glob.glob(os.path.join(directory + '/Labelled Nest Files', '*.json')))
    if len(jsons) < 1:
        print('''Looks like you didn't save your work into a .json file inside Labelme, or you didnt save it to the Labelled Nest Files.\n''')
        print(directory + '/Labelled Nest Files')
    for file in jsons: #to do: check whether a CSV already exists? Or just overwrite the old one, depending on how long it takes?
        with open(file.strip(".json") + ".csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["object index","label","shape","x", "y", "radius"])
            nest = json.load(open(file))
            object_index = -1
            for shape in nest['shapes']:
                object_index += 1
                label = shape['label']
                shape_type = shape['shape_type']
                if shape_type == "circle":
                    x_center = shape['points'][0][0] #x coordinate of point defining center of circle
                    y_center = shape['points'][0][1] #y coordinate of point defining center of circle
                    x_perimeter = shape['points'][1][0] #x coordinate of point defining the perimeter of the cirlce
                    y_perimeter = shape['points'][1][1] #y coordinate of point defining the perimeter of the cirlce

                    radius = ((x_center-x_perimeter)**2 + (y_center-y_perimeter)**2) ** 0.5
                    writer.writerow([object_index, label, shape_type, x_center, y_center, radius])
                
                if shape_type == "point":
                    x = shape['points'][0][0]
                    y = shape['points'][0][1]
                    writer.writerow([object_index, label, shape_type, x, y, np.nan])

                if shape_type == "polygon":
                    for point in shape['points']:
                        x = point[0]
                        y = point[1]
                        writer.writerow([object_index,label, shape_type, x, y, np.nan])
    print('''Okay, just finished! Before quitting Labelme, you should have saved your work in a .json file by pressing save inside the popup label GUI.
          \nIf you did that, I just generated an accompanying CSV file that you can open and check out.\nIt has all your saved points in it.''') 
    return 0

def preload_annotations(image_dir):
    """Propagates annotations from previous image if current image has no label."""
    print("Preloading annotations from previous image (if available)...")
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    label_dir = os.path.join(image_dir, 'Labelled Nest Files')

    for i in range(1, len(image_paths)):
        prev_img = os.path.basename(image_paths[i - 1])
        curr_img = os.path.basename(image_paths[i])
        prev_json = os.path.join(label_dir, prev_img.replace('.png', '.json'))
        curr_json = os.path.join(label_dir, curr_img.replace('.png', '.json'))

        # Only copy if current .json does not exist but previous one does
        if os.path.exists(prev_json) and not os.path.exists(curr_json):
            print(f'Copying {prev_json} -> {curr_json}')
            subprocess.run(['cp', prev_json, curr_json])

def labelNest_v2(directory):
    print('Launching LabelMe... Your preloaded labels should appear if available.')
    print(f"Directory: {directory}")
    
    # Make sure Labelled Nest Files folder exists
    if not os.path.exists(directory + '/Labelled Nest Files'):
        os.makedirs(directory + '/Labelled Nest Files')
    
    # Step 1: Preload labels from previous image if needed
    preload_annotations(directory)

    # Step 2: Run LabelMe on this folder
    bumblebox_dir = os.path.dirname(os.path.realpath(__file__))
    subprocess.run(['labelme', directory, '--config', os.path.join(bumblebox_dir, 'labelmerc'), '--output', os.path.join(directory, 'Labelled Nest Files')])

    # Step 3: Convert to CSV after LabelMe exits
    jsons = sorted(glob.glob(os.path.join(directory + '/Labelled Nest Files', '*.json')))
    if len(jsons) < 1:
        print('No annotations found. Did you save your work in LabelMe?')
        return

    for file in jsons:
        with open(file.strip(".json") + ".csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["object index","label","shape","x", "y", "radius"])
            nest = json.load(open(file))
            for i, shape in enumerate(nest['shapes']):
                label = shape['label']
                shape_type = shape['shape_type']
                if shape_type == "circle":
                    cx, cy = shape['points'][0]
                    px, py = shape['points'][1]
                    radius = ((cx - px)**2 + (cy - py)**2)**0.5
                    writer.writerow([i, label, shape_type, cx, cy, radius])
                elif shape_type == "point":
                    x, y = shape['points'][0]
                    writer.writerow([i, label, shape_type, x, y, np.nan])
                elif shape_type == "polygon":
                    for x, y in shape['points']:
                        writer.writerow([i, label, shape_type, x, y, np.nan])
    print('CSV export complete. You’re good to go!')


def label_single_image_with_preload(image_path):
    """Open a single image in LabelMe, copying annotations from the previous frame if available."""
    image_path = os.path.abspath(image_path)
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]

    label_dir = os.path.join(image_dir, 'Labelled Nest Files')
    os.makedirs(label_dir, exist_ok=True)

    # Look for the previous image in sorted order
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    idx = image_files.index(image_path)

    if idx > 0:
        prev_image = os.path.basename(image_files[idx - 1])
        print(f"Image files: {image_files}")
        print(f"idx: {idx}")
        print(f"Previous image: {prev_image}")
        prev_json = os.path.join(label_dir, prev_image.replace('.png', '.json'))
        curr_json = os.path.join(label_dir, base_name + '.json')

        # Copy previous annotations if current doesn't exist
        if os.path.exists(prev_json) and not os.path.exists(curr_json):
            print(f"Preloading: Copying {prev_json} → {curr_json}")
            subprocess.run(['cp', prev_json, curr_json])

    # Launch LabelMe on the single image
    print(f"Opening {image_path} in LabelMe...")
    bumblebox_dir = os.path.dirname(os.path.realpath(__file__))
    subprocess.run([
    'labelme',
    '--nosort',
    os.path.abspath(image_path),
    '--config', os.path.join(bumblebox_dir, 'labelmerc'),
    '--output', label_dir
    ])





"------------------------------------------------------"

def main(argv):
    
    directory = input("Please provide the folder with your nest images, and please make sure its on the computer that you're working on, and not a drive.\nInside that folder we'll make a folder called Labelled Nest Files, where we will store the data from our nest labelling\n\nNest data folder path:")
    if not os.path.exists(directory + '/Labelled Nest Files'):
        os.makedirs(directory + '/Labelled Nest Files')
    #if not os.path.exists('/Users/aec/Desktop/2023 bombus workshop pngs' + '/Nest Images/Labelled Nest Files'):
        #os.makedirs('/Users/aec/Desktop/2023 bombus workshop pngs' + '/Nest Images/Labelled Nest Files')
    #directory = directory + '/Nest Images'
    
    
    #vids2medianimg(directory)
    labelNest_v2(directory)


def main_v2(argv):
    image_path = input("Enter the full path to the image you want to annotate (e.g., frame_005.png): ").strip()
    if not os.path.exists(image_path):
        print("That image doesn't exist.")
        return

    label_single_image_with_preload(image_path)


if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    import sys
    status = main_v2(sys.argv)
    sys.exit(status)
