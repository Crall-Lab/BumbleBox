import cv2
from cv2 import aruco
import setup
from picamera2 import Picamera2
import time
from statistics import mean 

#edit code to add in rejected image points
def test_tracking(preview_time, width, height, tag_dictionary, box_type, shutter_speed, digital_zoom, tuning_file):
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
    
    
    '''load tuning file and initialize the camera (setting the format and the image size)'''
    tuning = Picamera2.load_tuning_file(tuning_file)
    picam2 = Picamera2(tuning=tuning)
    preview = picam2.create_preview_configuration({"format": "YUV420", "size": (width,height)})
    picam2.align_configuration(preview) #might cause an issue?
    picam2.configure(preview)
    
    '''set shutterspeed (or exposure time)'''
    picam2.set_controls({"ExposureTime": shutter_speed})
    
    '''set digital zoom'''
    if digital_zoom == type(tuple) and len(digital_zoom) == 4:
        picam2.set_controls({"ScalerCrop": digital_zoom})
    
    elif digital_zoom != None:
        print("The variable 'recording_digital_zoom' in the setup.py script is set incorrectly. It should be either 'None' or a value that looks like this: (offset_x,offset_y,new_width,new_height) for ex. (1000,2000,300,300)")
    
    '''start the camera'''
    picam2.start()
    time.sleep(2)
    
    start_time = time.time()
    
    tag_avg_list = []
    
    while ( (time.time() - start_time) < preview_time):
        
        array = picam2.capture_array()
        
        try:
            print(array.shape)
            gray = cv2.cvtColor(array, cv2.COLOR_YUV2GRAY_I420)
            print(gray.shape)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(gray)
            gray = cv2.cvtColor(cl1,cv2.COLOR_GRAY2RGB)
            
        except:
            print('converting to grayscale didnt work...')
            pass
            
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        frame_markers = aruco.drawDetectedMarkers(gray.copy(), corners, ids)
        try:
            tag_avg_list.append(len(ids))
        except:
            print("no tags found in this frame")
        resized = cv2.resize(frame_markers, (1352,1013), interpolation = cv2.INTER_AREA)
        cv2.imshow("frame",resized)
        #cv2.waitKey(5000)
        k = cv2.waitKey(5000) & 0xFF
        
        try:
            print(f"Numbers of tags tracked: {len(ids)}")
        except:
            print("No tags found")
        if k == 27:
            cv2.destroyAllWindows()
            break
        
    try:
        print(f"Average number of tags tracked: {round(mean(tag_avg_list),2)}")
    except:
        print("Average number of tags tracked: 0")
def main():
    
    test_tracking(setup.preview_time, setup.width, setup.height, setup.tag_dictionary, setup.box_type, setup.shutter_speed, setup.recording_digital_zoom, setup.tuning_file)
    
if __name__ == '__main__':
    
    main()

    


    
