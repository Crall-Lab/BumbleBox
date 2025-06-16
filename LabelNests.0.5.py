# LabelNests_v2

# ---- Standard library imports ----

import os                # For path manipulations and filesystem operations
import sys               # For system-specific parameters and functions
import csv               # (Unused here, but generally for reading/writing CSV files)
import subprocess        # To launch external programs like LabelMe from within Python
import glob              # To search for files using Unix-style pathname matching
import json              # To read/write and manipulate JSON annotation files
import shutil            # For copying files, used here to duplicate .json annotations
import base64            # To encode image files as base64 strings (LabelMe expects this for imageData)

# ---- Third-party libraries ----
import labelme           # Provides access to LabelFile methods, like loading and encoding image files


def label_single_image_with_preload(image_path):
    """
    Open a single image in LabelMe, copying annotations from the previous frame if available.
    Also embeds the image in the JSON using base64 so LabelMe will load it correctly.
    """
    image_path = os.path.abspath(image_path)       # Get absolute path to image
    image_dir = os.path.dirname(image_path)        # Get folder containing the image
    image_name = os.path.basename(image_path)      # Get just the image filename (e.g., frame_002.png)
    base_name = os.path.splitext(image_name)[0]    # Strip extension to get the base name (e.g., frame_002)

    # Get all .png images in the directory, sorted by name
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    image_names = [os.path.basename(p) for p in image_files]

    # Check that the image actually exists in the folder
    if image_name not in image_names:
        print(f"⚠️ Error: {image_name} not found in image directory.")
        return

    idx = image_names.index(image_name)            # Index of the current image in sorted list

    # Only do preload logic if there is a previous image
    if idx > 0:
        prev_image = image_names[idx - 1]          # Get previous image filename
        prev_json = os.path.join(image_dir, prev_image.replace('.png', '.json'))  # Path to its .json
        curr_json = os.path.join(image_dir, base_name + '.json')                  # Path to current .json

        # Copy the previous annotations if current one doesn't already exist
        if os.path.exists(prev_json) and not os.path.exists(curr_json):
            print(f"Preloading: Copying {prev_json} → {curr_json}")
            shutil.copy(prev_json, curr_json)      # Copy previous JSON to current one

            # Load the current image into memory and convert it to base64 (what LabelMe expects)
            loaded_image_data = labelme.LabelFile.load_image_file(image_path)     # Read raw bytes from PNG
            image_data_utf = base64.b64encode(loaded_image_data).decode('utf-8')  # Encode to base64 → UTF-8

            # Modify the copied JSON to reference the new image
            with open(curr_json, 'r+') as f:
                data = json.load(f)                  # Parse the JSON into a Python dictionary
                data['imagePath'] = image_name       # Update imagePath to point to the current image
                data['imageData'] = image_data_utf   # Embed the actual image data (so it renders in LabelMe)

                f.seek(0)                            # Move the file pointer to the beginning of the file
                                                    # So when we write, we overwrite the old contents

                json.dump(data, f, indent=2)         # Dump the updated JSON back to file with 2-space indentation

                f.truncate()                         # Remove any leftover old content from previous write
                                                    # Necessary because new content might be shorter than original
                                                    # Without this, leftover characters could corrupt the file


    # Launch LabelMe GUI for manual annotation of this single image
    print(f"Opening {image_path} in LabelMe...")
    bumblebox_dir = os.path.dirname(os.path.realpath(__file__))   # Get current script's directory
    subprocess.run([
        'labelme',
        '--config', os.path.join(bumblebox_dir, 'labelmerc'),     # Load custom label set if defined
        '--nosort',                                                # Prevent LabelMe from sorting filenames
        os.path.abspath(image_path)                                # Open the target image
    ])


def main(argv):
    """
    Main script logic that asks the user for an image and launches the labeling function.
    """
    image_path = input("Enter the full path to the image you want to annotate (e.g., frame_005.png): ").strip()
    if not os.path.exists(image_path):
        print("That image doesn't exist.")
        return

    label_single_image_with_preload(image_path)


if __name__ == "__main__": 
    """
    Only run the script logic if executed directly (not when imported as a module)
    """
    import sys
    status = main(sys.argv)
    sys.exit(status)
