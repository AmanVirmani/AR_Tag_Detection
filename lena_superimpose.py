import cv2
import numpy as np
from collections import deque

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	#lena = cv2.imread('./data/Lena.png')
	#lena_resize = cv2.resize(lena, warped.shape)
	#lena_warped = cv2.warpPerspective(lena_resize,np.linalg.inv(M),(maxWidth,maxHeight))
	#cv2.imshow('test',lena_warped)
	#cv2.waitKey(0)
	# return the warped image
	return warped, M

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
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
	#count_black_pixels = 0
	#for i in range(tag.shape[0]):
	#	for j in range(tag.shape[1]):
	#		if tag[i][j] == 0 :
	#			count_black_pixels += 1
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



cap = cv2.VideoCapture('./data/Tag0.mp4')
# Check if camera opened successfully
if (cap.isOpened()== False):
	print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:
		# Display the resulting frame
		cv2.imshow('Frame',frame)
		cv2.imwrite('test_frame.jpg',frame)
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
		cv2.drawContours(gray, squares, -1, (255,128,0), 3)
					#cv2.drawContours(frame, cnt, -1, (255,128,0), 3)

		#cv2.imshow('Frame', gray)
		warped, M = four_point_transform(gray_frame, squares[0].reshape((4,2)))
		cv2.namedWindow('tag', cv2.WINDOW_KEEPRATIO)
		cv2.imshow('tag', warped)
		cv2.waitKey(0)
		decode_tag(warped)
		#break
		lena = cv2.imread('./data/Lena.png')
		lena_resize = cv2.resize(lena, warped.shape)
		cv2.imshow('lena', lena_resize)
		cv2.waitKey(0)
		break
		# Press Q on keyboard to  exit
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break # Break the loop
	else:
		break
	# When everything done, release the video capture object
cap.release()
# Closes all the frames
#cv2.DestroyAllWindows()