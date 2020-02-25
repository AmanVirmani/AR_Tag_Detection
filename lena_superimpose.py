import cv2
import numpy as np
from collections import deque

#Finding the Homography of the AR tag from World Coordinate frame to Image Coordinate frame
def find_homography(source_pts, target_pts):
	combined_pts = np.hstack((order_points(source_pts),order_points(target_pts)))
	A = []

	for pts in combined_pts:
		x_t = pts[0]
		y_t = pts[1]
		x_s = pts[2]
		y_s = pts[3]

		A_first_row = [x_t,y_t,1,0,0,0,-x_s*x_t,-x_s*y_t,-x_s]
		A_second_row = [0,0,0,x_t,y_t,1,-y_s*x_t,-y_s*y_t,-y_s]
		A.append(np.vstack((A_first_row,A_second_row)))
	A = np.vstack(A)

	U,S,V = np.linalg.svd(A)
	V = V[8] # last column of V

	h = V/V[8]

	H = np.reshape(h,(3,3))

	return H

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped, M

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def isWhite(tag):
	ret, tag = cv2.threshold(tag,200,255,0)
	count_white = cv2.countNonZero(tag)
	return count_white > 0.9*tag.size

def tagMatrix(tag):
	tag = cv2.resize(tag,(200,200))
	bit_size = int(tag.shape[0]/8)
	a=0;b=0;
	ar_tag = np.zeros((8,8))
	for i in range(0,tag.shape[0],bit_size):
		for j in range(0,tag.shape[1],bit_size):
			if isWhite(tag[i:i+bit_size,j:j+bit_size]) :
				ar_tag[a][b] = 1
			else :
				ar_tag[a][b] = 0
			b +=1
		a +=1
		b = 0
	return ar_tag

def decode_tag(tag):
	ar_tag = tagMatrix(tag)
	tag_angle = orientation(ar_tag)
	tag_id = deque([ar_tag[4,3],ar_tag[4,4],ar_tag[3,4],ar_tag[3,3]])
	tag_id.rotate(tag_angle%90)
	return tag_angle, tag_id


def orientation(ar_tag):
	if ar_tag[2, 2]:
		orientation = 180
	elif ar_tag[2, 5]:
		orientation = 90
	elif ar_tag[5, 2]:
		orientation = 270
	elif ar_tag[5, 5]:
		orientation = 0
	else :
		orientation = None
		print('No tag detected')
	return orientation

def image_overlay(frame,pts,angle) :
	lena = cv2.imread('./data/Lena.png')
	lena = cv2.resize(lena,(200,200))
	if angle == 90:
		lena = cv2.rotate(lena, cv2.ROTATE_90_COUNTERCLOCKWISE)
	elif angle == 180:
		lena = cv2.rotate(lena, cv2.ROTATE_180)
	elif angle == 270:
		lena = cv2.rotate(lena, cv2.ROTATE_90_CLOCKWISE)
	#cv2.imshow('lena',lena)
	#cv2.waitKey(0)
	pts1 = order_points(np.float32([[0,0],[200,0],[0,200],[200,200]]))
	#pts2 = np.float32(order_points(squares[0].reshape((4,2))))
	pts2 = np.float32(order_points(pts))
	M = cv2.getPerspectiveTransform(pts1,pts2)
	#dst = warpPerspective(lena,M,(frame.shape[1],frame.shape[0]))
	#cv2.imshow('dst',dst)
	#cv2.waitKey(0)
	dst = cv2.warpPerspective(lena,M,(frame.shape[1],frame.shape[0]))
	overlay = cv2.add(frame,dst)
	return overlay

def superimpose():
	frame = cv2.imread('test_frame.jpg')
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

	print(squares)
	std_size = 200
	for pts in squares:
		src_pts = order_points(pts.reshape((4,2)))
		tgt_pts = order_points(np.array([[0,0],[0,std_size],[std_size,std_size],[std_size,0]]))
		H = find_homography(src_pts, tgt_pts)
		print(H)



def decodeTag():
	cap = cv2.VideoCapture('./data/Tag0.mp4')
	# Check if camera opened successfully
	if (cap.isOpened()== False):
		print("Error opening video stream or file")

	# Read until video is completed
	while(cap.isOpened()):
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

			cv2.drawContours(frame, squares, -1, (255,128,0), 3)

			warped,M = four_point_transform(gray, squares[0].reshape((4,2)))

			tag_angle, tag_id = decode_tag(warped)
			final = image_overlay(frame, squares[0].reshape((4,2)),tag_angle)
			cv2.imshow('final',final)
			# Press Q on keyboard to  exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break # Break the loop
		else:
			break
	cap.release()
	cv2.DestroyAllWindows()

if __name__=='__main__':
	superimpose()
