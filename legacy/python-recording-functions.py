#!/usr/bin/env python

from picamera2 import Picamera2, Preview
from picamera2.encoders import JpegEncoder
from picamera2.encoders import H264Encoder
from sys import getsizeof
import time
import numpy as np
import cv2
from PIL import Image
import os
import subprocess
import ffmpeg

def picam2_record_mjpeg(filename, outdir, recording_time=20, quality=95, size=(4056,3040), imformat="RGB888", fps=4, buffer_count=2):
	frame_duration_us = int(1/fps * 10**6)
	picam2 = Picamera2()
	video_config = picam2.create_video_configuration(main={"size": size, "format": imformat}, controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)}, buffer_count=2)
	picam2.align_configuration(video_config)
	picam2.configure(video_config)

	picam2.start_preview()
	encoder = JpegEncoder(q=quality)
	output = outdir+filename

	picam2.start()
	time.sleep(2)
	picam2.start_encoder(encoder,output,pts=outdir+filename+"_pts.txt")
	#picam2.start_recording(encoder,output,pts=outdir+filename+"_pts.txt")

	time.sleep(recording_time)
	
	picam2.stop()
	picam2.stop_encoder()
	
	


	
def folder_jpgs2mjpeg(dirpath):
	cmd = 'for i in ' + dirpath + '*.jpg: do ffmpeg -i "$i" "${i%.*}.mjpeg"; done'
	os.system(cmd)
	




def picam2_YUV420array2mjpeg(recording_time=1, quality="maximum", fps=6, outdir='/mnt/bombusbox/burstcapturejpgs/', filename='testpicam2_yuv2vid', imtype="y"): #make options "yuv", "rgb", "y", "all"
	
	picam2 = Picamera2()
	preview = picam2.create_preview_configuration({"format": "YUV420", "size": (4032,3040)})
	picam2.configure(preview)
	picam2.start()

	time.sleep(2)
	start_time = time.time()
	
	frames_dict = {}
	i = 0
	
	print("beginning video capture")
	while ( (time.time() - start_time) < recording_time) or (i <= fps*recording_time):
		timestamp = time.time() - start_time

		yuv420 = picam2.capture_array()
		
		if imtype == "y":
			yuv420 = yuv420[0:3040, :]
		frames_dict[f"frame_{i:03d}"] = [yuv420, timestamp]
		time.sleep(1/(fps+1))
		i += 1
	print(f'finished capturing frames to arrays, captured {i} frames in {time.time()-start_time}')
	sizeof = getsizeof(frames_dict)
	print(f"Size of dictionary storing the frames (in bytes): {sizeof}")
	
	if imtype =="y":
		for frame_number in frames_dict:
			imlist = frames_dict[frame_number]
			array = imlist[0]
			y_im = Image.fromarray(array, mode="L") #L meaning a single 8-bit channel, greyscale
			y_im.save(outdir+frame_number+".jpg", quality=quality)
	
	elif imtype =="yuv":
		for frame_number in frames_dict:
			imlist = frames_dict[frame_number]
			array = imlist[0]
			yuv_im = Image.fromarray(array, mode="YCbCr") #YCbCr meaning 3x8-bit channels for YUV
			yuv_im.save(outdir+filename+".jpg", quality=quality)
	
	elif imtype =="rgb":
		for frame_number in frames_dict:
			imlist = frames_dict[frame_number]
			array = imlist[0]
			yuv_im = Image.fromarray(array, mode="YCbCr")
			rgb_im = cv2.cvtColor(yuv_im, cv2.COLOR_YUV420p2RGB)
			rgb_im.save(outdir+filename+".jpg", quality=quality)
	
	elif imtype == "all":
		for frame_number in frames_dict:
			imlist = frames_dict[frame_number]
			array = imlist[0]
			yuv_im = Image.fromarray(array, mode="YCbCr")
			y_im = Image.fromarray(array, mode="L")
			rgb_im = cv2.cvtColor(yuv_im, cv2.COLOR_YUV420p2RGB)
			y_im.save(outdir+filename+".jpg", quality=quality)
			yuv_im.save(outdir+filename+".jpg", quality=quality)
			rgb_im.save(outdir+filename+".jpg", quality=quality)
	
	else:
		print("Error: imtype must equal either 'y', 'yuv', 'rgb', or 'all'")
		return print("Exit code: 1")
	
	print("now turning the jpgs into an mjpeg video...")
	#cmd = 'for i in ' + outdir + '*.jpg: do ffmpeg -i "$i" "${i%.*}.mjpeg"; done'
	#cmd = 'for i in /mnt/bombusbox/burstcapturejpgs/*.jpg; do ffmpeg -i "$i" "${i%.*}.mjpeg"; done'
	cmd = 'ffmpeg -i /mnt/bombusbox/burstcapturejpgs/*.jpg -vf fps=10 -pix_fmt yuv420p /mnt/bombusbox/burstcapturejpgs/output.mp4'
	os.system(cmd)
	print("Success?")

#def write_jpgs2mp4(numframes, jpgdir='/mnt/bombusbox/burstcapturejpgs', cv2.VideoWriter_fourcc(*'mp4'), fps=10, size=(4032,3040):
	


def picam2_YUV420arraycapture_timetest(recording_time=10, fps=15, quality="maximum", outdir='/mnt/bombusbox/burstcapturejpgs/', filename='testpicam2_yuv2vid', container="dict",imtype="y"): #make options "yuv", "rgb", "y", "all"

	picam2 = Picamera2()
	preview = picam2.create_preview_configuration({"format": "YUV420", "size": (4032,3040)})
	picam2.configure(preview)
	picam2.start()

	time.sleep(2)
	start_time = time.time()
	
	frames_dict = {}
	i = 0
	
	print("beginning video capture")
	while ( (time.time() - start_time) < recording_time): #or (i <= fps*recording_time):
		timestamp = time.time() - start_time
		
		yuv420 = picam2.capture_array()
		
		if imtype=="y" and container=="dict": 
			yuv420 = yuv420[0:3040, :]
			frames_dict[f"frame_{i:03d}"] = [yuv420, timestamp]
			#time.sleep(1/(fps+10))
			i += 1
		
	print(f'finished capturing frames to arrays, captured {i} frames in {time.time()-start_time}')
	sizeof = getsizeof(frames_dict)
	print(f"Size of dictionary storing the frames: {sizeof}")
	
	for frame in frames_dict:
		print(frame_number, frames_dict[frame_number][1])
		
	if imtype =="y":
		timestamps = {}
		save_time = time.time()
		for frame_number in frames_dict:
			imlist = frames_dict[frame_number]
			frame_array = imlist[0]
			y_im = Image.fromarray(frame_array, mode="L") #L meaning a single 8-bit channel, greyscale
			y_im.save(outdir+frame_number+".jpg", quality=quality)
			timestamps[frame_number] = time.time() - save_time
		
		end_time = time.time()
		
		for frame_number in timestamps:
			print(f'{frame_number}: {timestamps[frame_number]} seconds')
			
		print(f'wrote {int(len(frames_dict))} images in {end_time - save_time} seconds.\nThats an average of {int(len(frames_dict)) / (end_time - save_time)} frames per second!')











def x265vidwrite(filename, images, framerate=5, vcodec='libx265'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
        print(images.shape)
    n,channels,height,width = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='yuv420p', r=framerate, s='{}x{}'.format(width, height))
            .output(filename, pix_fmt='yuv420p', vcodec=vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()











### date is 5/30, trying to use just opencv to write to .mp4 or h265
def arraycapture_2_mp4(recording_time=2, codec='mp4', fps=6, outdir='/mnt/bombusbox/', filename='testpicam2_yuv2vid', imtype="yuv"):
	
	picam2 = Picamera2()
	preview = picam2.create_preview_configuration({"format": "YUV420", "size": (4032,3040)})
	picam2.configure(preview)
	picam2.start()

	time.sleep(2)
	start_time = time.time()
	
	#frames_dict = {}
	frames_list = []
	i = 0
	
	print("beginning video capture")
	while ( (time.time() - start_time) < recording_time): #or (i <= fps*recording_time):
		timestamp = time.time() - start_time
		
		yuv420 = picam2.capture_array()
		print(yuv420.shape)
		frames_list.append([yuv420])
		#yuv420 = yuv420[0:3040, :]
		#frames_dict[f"frame_{i:03d}"] = [yuv420, timestamp]
		time.sleep(1/(fps+1))
		i += 1
		
	print(f'finished capturing frames to arrays, captured {i} frames in {time.time()-start_time}')
	#sizeof = getsizeof(frames_dict)
	sizeof = getsizeof(frames_list)
	#print(f"Size of dictionary storing the frames: {sizeof}")
	
	out = None
	if codec == 'mp4':
		vid_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter('/mnt/bumblebox/data/testvideo_rgb.mp4',vid_fourcc,10,(4032,3040))
		#out = cv2.VideoWriter('testvideo1.mp4',vid_fourcc,10,(4032,3040))
		#out = cv2.VideoWriter('testvideo1.mp4', cv2.CAP_FFMPEG,vid_fourcc,10,(4032,3040)) #apiPreference=cv2.CAP_FFMPEG
	
	'''	
	#elif codec == 'h265':
		
	#	x265vidwrite('testx265.mp4',frames_list)
	
		
	
		#look into this for HEVC encoding: https://stackoverflow.com/questions/61260182/how-to-output-x265-compressed-video-with-cv2-videowriter
		#vid_fourcc = cv2.VideoWriter_fourcc(*'hvc1')
		#out = cv2.VideoWriter('testh265.mp4', cv2.CAP_FFMPEG,vid_fourcc,5,(4032,3040)) #apiPreference=cv2.CAP_FFMPEG
		
		import ffmpeg

		(
			ffmpeg
			.input('0.mp4')
			.filter('fps', fps=5, round='up')
			.output('out.mp4', vcodec='libx265')
			.run()
		)
		
	'''
	
	for i, im_array in enumerate(frames_list):
		frame = im_array[0]
		print(frame.shape)
		#frame = cv2.resize(frame, (4032, 3040))
		rgb_im = cv2.cvtColor(frame, cv2.COLOR_YUV420p2RGB)
		#gray_im = cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_I420)
		#frame = im_array[0]
		#print(rgb_im.shape)
		#img = Image.fromarray(frame)#, mode="YCbCr")
		#print(img.size)
		#print(img.shape)
		#cv2.imwrite(str(i) + "gray.jpg", gray_im)
		#cv2.imwrite(str(i) + "rawyuv.jpg", frame)
		#cv2.imwrite(str(i) + "rgb.jpg", rgb_im)
		#cv2.imwrite(str(i) + ".jpg", rgb_im)
		out.write(rgb_im)
		#print(f'wrote frame {img}') 
	out.release()
	cv2.destroyAllWindows()
	'''
	
# okay this is trying to get the best video compression to work for our videos, making them into HEVC compressed mp4s. Need to use FFMPEG to do it,
# so this function would be run inside of the create mp4 from RAM function, just if HEVC compression is wanted
# otherwise you can just use cv2.VideoWriter to make mp4s. 
# this should be considered an alternative for more experienced users
# right now this is just copied off of the link from above, need to test it and modify it to work inside of the function above
def writeMp4withHEVC_CompressionFromRAM(outdir, filename, frames_list, width, height, fps):
	
	import cv2
	import numpy as np
	import subprocess as sp
	import shlex

	output_filename = filename + '.mp4'

	# Open ffmpeg application as sub-process
	# FFmpeg input PIPE: RAW images in BGR color format
	# FFmpeg output MP4 file encoded with HEVC codec.
	# Arguments list:
	# -y                   Overwrite output file without asking
	# -s {width}x{height}  Input resolution width x height (1344x756)
	# -pixel_format bgr24  Input frame color format is BGR with 8 bits per color component
	# -f rawvideo          Input format: raw video
	# -r {fps}             Frame rate: fps (25fps)
	# -i pipe:             ffmpeg input is a PIPE
	# -vcodec libx265      Video codec: H.265 (HEVC)
	# -pix_fmt yuv420p     Output video color space YUV420 (saving space compared to YUV444)
	# -crf 24              Constant quality encoding (lower value for higher quality and larger output file). ## This was originally 24 but I changed it to 18 for testing purposes
	# {output_filename}    Output file name: output_filename (output.mp4)
	process = sp.Popen(shlex.split(f'ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -crf 18 {output_filename}'), stdin=sp.PIPE)

	# Build synthetic video frames and write them to ffmpeg input stream.
	for i in frames_list:
		img = i[0]

		# Write raw video frame to input stream of ffmpeg sub-process.
		process.stdin.write(img.tobytes())

	# Close and flush stdin
	process.stdin.close()

	# Wait for sub-process to finish
	process.wait()

	# Terminate the sub-process
	process.terminate()  # Note: We don't have to terminate the sub-process (after process.wait(), the sub-process is supposed to be closed).
		
	
		
    
def trackTagsFromRAM(frames_list):
	
	#edit dict to be passed from outer function?
	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50) 
	parameters = aruco.DetectorParameters_create()
	
	#edit these to be passed into it from outer function?
	parameters.minMarkerPerimeterRate=0.03
	parameters.adaptiveThreshWinSizeMin=5
	parameters.adaptiveThreshWinSizeStep=6
	parameters.polygonalApproxAccuracyRate=0.06

	frame_num = 0
	noID = []
	raw = []

	for i in frames_list:
		frame = i[0]
		
		try:
		   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		   cl1 = clahe.apply(gray)
		   gray = cv2.cvtColor(cl1,cv2.COLOR_GRAY2RGB)
			
		except:
			continue
			
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)

		for i in range(len(rejectedImgPoints)):
			c = rejectedImgPoints[i][0]
			xmean = c[:,0].mean() #for calculating the centroid
			ymean = c[:,1].mean() #for calculating the centroid
			xmean_top_point = (c[0,0] + c[1,0]) / 2 #for calculating the top point of the tag
			ymean_top_point = (c[0,1] + c[1,1]) / 2 #for calculating the top point of the tag
			noID.append( [frame_num, "X", float(xmean), float(ymean), float(xmean_top_point), float(ymean_top_point), 100, None] ) #[[float(xmean), float(ymean)], [float(xmean_top_point), float(ymean_top_point)]] )
		
		if ids is not None:
			for i in range(len(ids)):
				c = corners[i][0]
				xmean = c[:,0].mean() #for calculating the centroid
				ymean = c[:,1].mean() #for calculating the centroid
				xmean_top_point = (c[0,0] + c[1,0]) / 2 #for calculating the top point of the tag
				ymean_top_point = (c[0,1] + c[1,1]) / 2 #for calculating the top point of the tag
				raw.append( [frame_num, int(ids[i]),float(xmean), float(ymean), float(xmean_top_point), float(ymean_top_point), 100, None] ) #[[float(xmean), float(ymean)], [float(xmean_top_point), float(ymean_top_point)]] )

		frame_num += 1
		
	df = pandas.DataFrame(raw)
	df = pandas.DataFrame(raw)
	df = df.rename(columns = {0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'check'})
	df.to_csv(identified_csv, index=False)
	df2 = pandas.DataFrame(noID)
	df2 = pandas.DataFrame(noID)
	df2 = df2.rename(columns = {0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'check'})
	df2.to_csv(potential_csv, index=False)

	print("Average number of tags found: " + str(len(df.index)/frame_num))
	
	return df, df2, frame_num
	'''
		
if __name__ == '__main__':
	
    #picam2_YUV420arraycapture_timetest()
	arraycapture_2_mp4()
	#picam2_YUV420array2mjpeg()
	#picam2_record_mjpeg('newbees1.mjpeg','/mnt/bombusbox/')
	

