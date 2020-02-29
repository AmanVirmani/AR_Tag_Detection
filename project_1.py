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
			std_size = 200
			cube_list = []
			squares = contour_generator(frame)
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
					#cv2.imshow("Lena", final_image)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break # Break the loop
		else:
			break

	if output is not None:
		saveVideo(images,output)
	cap.release()
	cv2.destroyAllWindows()

def lenaSuperimpose(input,output):
	cap = cv2.VideoCapture(input)
	# Check if camera opened successfully
	if (cap.isOpened()== False):
		print("Error opening video stream or file")
	count = -1
	images= []
	# Read until video is completed
	while(cap.isOpened()):
		count +=1
		print('Processing Frame No. {}'.format(count))
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			std_size = 200
			squares = contour_generator(frame)
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
			#cv2.imshow("Lena_%d.png"%count, final_image)
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
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			squares = contour_generator(frame)

			img = cv2.drawContours(frame.copy(), squares, -1, (255,128,0), 3)
			images1.append(img)
			#continue
			for pts in squares:
				src_pts = order_points(pts.reshape((4,2)))
				tag_warped = four_point_transform(frame,src_pts)
				if tag_warped.shape[0] <= 0 or tag_warped.shape[1] <= 0:
					print('tag not found in frame {}'.format(count))
					continue
				tag_angle, tag_id = decode_tag(tag_warped)
				if tag_angle is None:
					continue
				print('tag angle : {} degrees'.format(tag_angle))
				print('tag id : {}'.format(np.array(tag_id,dtype=int)))
				final_image = cv2.putText(img,'Tag_id '+str(np.array(tag_id)),
										  (pts[0][0][0],pts[0][0][1]), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,cv2.LINE_AA)
				final_image = cv2.putText(img,'Tag_angle '+ str(tag_angle)+' degrees',
										  (pts[0][0][0], pts[0][0][1]+50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2,
										  cv2.LINE_AA)
				cv2.imwrite('data/tagid_{}.jpg'.format(count), final_image)
				#cv2.imshow('e',final_image);cv2.waitKey(0);cv2.destroyAllWindows()
				#break
		else:
			break
	cap.release()
	cv2.destroyAllWindows()
	#if output is not None:
	#	saveVideo(images1,output)

if __name__=='__main__':
	main()

#TODO : add stop code while displaying video in all functions
#TODO : improve runtime

