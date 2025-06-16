#LabelNests_v2

import os
import sys
import csv
import subprocess
import glob
import json
import numpy as np
import pandas as pd
import shutil

def label_single_image_with_preload(image_path):
    """Open a single image in LabelMe, copying annotations from the previous frame if available."""
    image_path = os.path.abspath(image_path)
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]

    #label_dir = os.path.join(image_dir, 'Labelled Nest Files')
    #os.makedirs(label_dir, exist_ok=True)

    # Look for the previous image in sorted order
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    image_names = [os.path.basename(p) for p in image_files]
    if image_name not in image_names:
        print(f"⚠️ Error: {image_name} not found in image directory.")
        return
    idx = image_names.index(image_name)


    if idx > 0:
        prev_image = image_names[idx - 1]
        print(f"Found previous image: {prev_image}")
        #print(f"Image files: {image_files}")
        print(f"idx: {idx}")
        print(f"Previous image: {prev_image}")
        prev_json = os.path.join(image_dir, prev_image.replace('.png', '.json'))
        curr_json = os.path.join(image_dir, base_name + '.json')

        # Copy previous annotations if current doesn't exist
        if os.path.exists(prev_json) and not os.path.exists(curr_json):
            print(f"Preloading: Copying {prev_json} → {curr_json}")
            # Copy the JSON file
            shutil.copy(prev_json, curr_json)

            # Patch the contents of the json to point to the new image
            with open(curr_json, 'r+') as f:
                data = json.load(f)
                data['imagePath'] = image_name #f"../{image_name}"  # or just image_name if same dir
                data.pop('imageData', None)  # this removes the key entirely
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()


    # Launch LabelMe on the single image
    print(f"Opening {image_path} in LabelMe...")
    bumblebox_dir = os.path.dirname(os.path.realpath(__file__))
    subprocess.run([
    'labelme',
    '--config', os.path.join(bumblebox_dir, 'labelmerc'),
    '--nosort',
    os.path.abspath(image_path)
    ])


def main(argv):
    image_path = input("Enter the full path to the image you want to annotate (e.g., frame_005.png): ").strip()
    if not os.path.exists(image_path):
        print("That image doesn't exist.")
        return

    label_single_image_with_preload(image_path)


if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    import sys
    status = main(sys.argv)
    sys.exit(status)