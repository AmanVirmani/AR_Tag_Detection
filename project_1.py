import cv2
import numpy as np
import argparse
from utility import *

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-q", "--question", required=False, help="Question Number to work on solution for",
					default='1', type=str)
	ap.add_argument("-o", "--output", required=False, help="Output path to save the videos",
					default='None', type=str)
	ap.add_argument("-i", "--input", required=False, help="Input video path",
				default='./data/Tag0.mp4', type=str)
	#ap.add_argument("-q", "--question", required=False, help="Question Number to work on solution for",
	#                default=1, nargs="*", type=int)
	args = vars(ap.parse_args())

	if args['question'] == '1':
		decodeTag(args['input'], args['output'])
	elif args['question'] == '2a':
		lenaSuperimpose(args['input'], args['output'])
	elif args['question'] == '2b':
		projectCube(args['input'], args['output'])

def projectCube(input,output):
	cap = cv2.VideoCapture(input)
	if (cap.isOpened()== False):
		print("Error opening video stream or file")

	count = -1
	images = []
	s = 1 #0.5   #scaling
	while(cap.isOpened()):
		count += 1
		print('Processing Frame No. {}'.format(count))
		ret, frame = cap.read()
		if ret == True:
			h,w = frame.shape[:2]
			h,w = int(h*s), int(w*s)
			frame = cv2.resize(frame,(w,h))
			gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.bilateralFilter(gray_frame, 15, 75, 75)
			ret, gray = cv2.threshold(gray_frame, 200, 255, 0)
			cnts, _ = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			#cnts = sorted(cnts, key=cv2.contourArea,reverse=True)
			squares=[]
			for cnt in cnts:
				cnt_len = cv2.arcLength(cnt, True)
				cnt = cv2.approxPolyDP(cnt, 0.1*cnt_len, True)
				if len(cnt) == 4:
					if 2000*s < cv2.contourArea(cnt) < 17500*s:
						squares.append(cnt)

			std_size = 200
			cube_list = []
			for pts in squares:
				src_pts = order_points(pts.reshape((4,2)))
				tgt_pts = order_points(np.array([[0,0],[0,std_size],[std_size,std_size],[std_size,0]]))
				H = find_homography(src_pts, tgt_pts)
				axis = np.float32([[0, 0, 0], [0, 200, 0], [200, 200, 0], [200, 0, 0], [0, 0, -200], [0, 200, -200], [200, 200, -200],[200, 0, -200]])
				mask = np.full(frame.shape, 0, dtype='uint8')
				tag_id = 1
				if not tag_id == None:
					r, t, K = calculator(H)
					points, jac = cv2.projectPoints(axis, r, t, K, np.zeros((1, 4)))
					img = draw_cube(mask, points)
					cube_list.append(img.copy())
			if cube_list != []:  # empty cube list
				for cube in cube_list:
					mask = np.full(frame.shape, 0, dtype='uint8')
					temp = cv2.add(mask, cube.copy())
					mask = temp
					cube_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
					r, cube_bin = cv2.threshold(cube_gray, 10, 255, cv2.THRESH_BINARY)
					mask_inv = cv2.bitwise_not(cube_bin)
					mask_3d = frame.copy()
					mask_3d[:, :, 0] = mask_inv
					mask_3d[:, :, 1] = mask_inv
					mask_3d[:, :, 2] = mask_inv
					img_masked = cv2.bitwise_and(frame, mask_3d)
					final_image = cv2.add(img_masked, mask)
					images.append(final_image)
					cv2.imshow("Lena", final_image)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break # Break the loop
		else:
			break

	if output is not None:
		saveVideo(images,output)
	cap.release()
	cv2.destroyAllWindows()

def lenaSuperimpose(input,output):
	cap = cv2.VideoCapture(output)
	# Check if camera opened successfully
	if (cap.isOpened()== False):
		print("Error opening video stream or file")
	count = -1
	images= []
	# Read until video is completed
	while(cap.isOpened()):
		count +=1
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.bilateralFilter(gray_frame, 15, 75, 75)
			ret, gray = cv2.threshold(gray_frame, 200, 255, 0)
			cnts, _ = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			cnts = sorted(cnts, key=cv2.contourArea,reverse=True)
			squares=[]
			for cnt in cnts:
				cnt_len = cv2.arcLength(cnt, True)
				cnt = cv2.approxPolyDP(cnt, 0.1*cnt_len, True)
				if len(cnt) == 4:
					if 2000 < cv2.contourArea(cnt) < 17500:
						squares.append(cnt)

			std_size = 200
			for pts in squares:
				src_pts = order_points(pts.reshape((4,2)))
				tgt_pts = order_points(np.array([[0,0],[0,std_size],[std_size,std_size],[std_size,0]]))
				H = find_homography(src_pts, tgt_pts)
				tag_warped = four_point_transform(frame,src_pts)
				tag_angle, tag_id = decode_tag(tag_warped)
				lena = cv2.imread('./data/Lena.png')
				lena = cv2.resize(lena,(200,200))
				if tag_angle == 90:
					lena = cv2.rotate(lena, cv2.ROTATE_90_COUNTERCLOCKWISE)
				elif tag_angle == 180:
					lena = cv2.rotate(lena, cv2.ROTATE_180)
				elif tag_angle == 270:
					lena = cv2.rotate(lena, cv2.ROTATE_90_CLOCKWISE)
				H_inv = np.linalg.inv(H)
				H_inv /= H_inv[2,2]
				dst = warpPerspective(lena,H_inv,(frame.shape[1],frame.shape[0]))
				lena = dst.astype(np.uint8)

			mask = np.full(frame.shape, 0, dtype='uint8')
			temp = cv2.add(mask, lena.copy())
			mask = temp
			lena_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
			r, lena_bin = cv2.threshold(lena_gray, 10, 255, cv2.THRESH_BINARY)
			mask_inv = cv2.bitwise_not(lena_bin)
			mask_3d = frame.copy()
			mask_3d[:, :, 0] = mask_inv
			mask_3d[:, :, 1] = mask_inv
			mask_3d[:, :, 2] = mask_inv
			img_masked = cv2.bitwise_and(frame, mask_3d)
			final_image = cv2.add(img_masked, mask)
			images.append(final_image)
			cv2.imshow("Lena_%d.png"%count, final_image)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break # Break the loop
		else:
			break
	cap.release()
	cv2.destroyAllWindows()
	if output is not None:
		saveVideo(images,output)

def decodeTag(input,output):
	cap = cv2.VideoCapture(input)
	# Check if camera opened successfully
	if (cap.isOpened()== False):
		print("Error opening video stream or file")
	count = -1
	images1 = []
	images2 = []
	# Read until video is completed
	while(cap.isOpened()):
		count +=1
		print('Processing Frame No. {}'.format(count))
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			#gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#gray = cv2.bilateralFilter(gray_frame, 15, 75, 75)
			#ret, gray = cv2.threshold(gray, 200, 255, 0)
			squares = contour_generator(frame)
			squares2 = contour_generator2(frame)

			img = cv2.drawContours(frame.copy(), squares, -1, (255,128,0), 3)
			img2 = cv2.drawContours(frame.copy(), squares2, -1, (255,128,0), 3)
			images1.append(img)
			images2.append(img2)
			#cv2.imshow('frame', img)
			#cv2.waitKey(0)
			continue
			std_size = 200
			for pts in squares:
				src_pts = order_points(pts.reshape((4,2)))
				tag_warped = four_point_transform(gray,src_pts)
				if tag_warped.shape[0] <= 0 or tag_warped.shape[1] <= 0:
					print('tag not found in frame {}'.format(count))
					continue
				cv2.imshow('tag',tag_warped)
				cv2.waitKey(0)
				tag_angle, tag_id = decode_tag(tag_warped)
				if tag_angle is None:
					continue
				print('tag angle : {} degrees'.format(tag_angle))
				print('tag id : {}'.format(np.array(tag_id,dtype=int)))
		else:
			break
	cap.release()
	cv2.destroyAllWindows()
	saveVideo(images1,'data/sanchit.avi')
	saveVideo(images2,'data/aman.avi')

if __name__=='__main__':
	main()

#TODO : add stop code while displaying video in all functions
#TODO : improve runtime

