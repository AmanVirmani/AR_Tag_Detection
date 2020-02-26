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
	dst = warpPerspective(lena,M,(frame.shape[1],frame.shape[0]))
	cv2.imshow('dst', dst.astype(np.uint8))
	cv2.waitKey(0)
	#dst = cv2.warpPerspective(lena,M,(frame.shape[1],frame.shape[0]))
	#dst = get_warp_perspective(lena,M,(frame.shape[1],frame.shape[0]))
	overlay = cv2.add(frame,dst)
	return overlay

def to_img(mtr):
	V,H,C = mtr.shape
	img = np.zeros((H,V,C), dtype='int')
	for i in range(mtr.shape[0]):
		img[:,i] = mtr[i]
	return img

def to_mtx(img):
	H,V,C = img.shape
	mtr = np.zeros((V,H,C), dtype='int')
	for i in range(img.shape[0]):
		mtr[:,i] = img[i]
	return mtr

def warpPerspective(img, M, dsize):
	mtr = to_mtx(img)
	R,C = dsize
	dst = np.zeros((R,C,mtr.shape[2]))
	for i in range(mtr.shape[0]):
		for j in range(mtr.shape[1]):
			res = np.dot(M, [i,j,1])
			i2,j2,_ = (res / res[2] + 0.5).astype(int)
			if i2 >= 0 and i2 < R:
				if j2 >= 0 and j2 < C:
					dst[i2,j2] = mtr[i,j]
	return to_img(dst)

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
		lena = cv2.imread('data/Lena.png')
		H_inv = np.linalg.inv(H)
		H_inv /= H_inv[2,2]
		dst = Warp(lena,H_inv,frame.shape)
		#dst = cv2.warpPerspective(lena,H_inv,(frame.shape[1],frame.shape[0]))
		#dst = get_warp_perspective(lena,H_inv,lena.shape)
		#dst = wrap_final(lena,frame, H,(frame.shape[1],frame.shape[0]))
		#dst = warpImage(lena,frame,H,(frame.shape[1],frame.shape[0]))
		cv2.imshow('warp',dst)
		cv2.waitKey(0)
		print('finish')

def Projection_matrix(H):
# Calibration Matrix
    K = np.array([[1406.08415449821,0,0],[2.20679787308599, 1417.99930662800,0],[1014.13643417416, 566.347754321696,1]])
    K_reshape = np.reshape(K,(3,3))
    K_trans = np.transpose(K_reshape)
    K_inv = np.linalg.inv(K_trans)
    K_inv_H = np.dot(K_inv,H)
    K_inv_h1 = K_inv_H[:,0]
    K_inv_h2 = K_inv_H[:,1]
    lambd = 2/(np.linalg.norm(K_inv_h1)+np.linalg.norm(K_inv_h2))

    if np.linalg.det(K_inv_H) < 0:
        B = -lambd*K_inv_H
    else:
        B = lambd*K_inv_H

    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]
    r1 = b1
    r2 = b2
    r3 = np.cross(r1,r2)
    t=b3
    R_t=np.vstack([r1,r2,r3,t])
    P = np.dot(K_trans,np.transpose(R_t))
    return P

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

			#warped,M = four_point_transform(gray, squares[0].reshape((4,2)))

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

def projectCube():
	cap = cv2.VideoCapture('./data/Tag0.mp4')
	# Check if camera opened successfully
	if (cap.isOpened() == False):
			print("Error opening video stream or file")
	while (cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		ret, frame = cap.read()
		if ret == True:
			if ret == True:
				gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				gray = cv2.bilateralFilter(gray_frame, 15, 75, 75)
				gray = cv2.bilateralFilter(gray_frame, 15, 75, 75)
				ret, gray = cv2.threshold(gray_frame, 200, 255, 0)
				ret, gray = cv2.threshold(gray_frame, 200, 255, 0)
				cnts, _ = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				cnts, _ = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
				cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
				squares = []
				squares = []
				for cnt in cnts:
					for cnt in cnts:
						cnt_len = cv2.arcLength(cnt, True)
						cnt_len = cv2.arcLength(cnt, True)
						cnt = cv2.approxPolyDP(cnt, 0.1 * cnt_len, True)
						cnt = cv2.approxPolyDP(cnt, 0.1 * cnt_len, True)
						if len(cnt) == 4:
							if len(cnt) == 4:
								if 2000 < cv2.contourArea(cnt) < 17500:
									if 2000 < cv2.contourArea(cnt) < 17500:
										squares.append(cnt)
										squares.append(cnt)
				cv2.drawContours(frame, squares, -1, (255, 128, 0), 3)
				warped, M = four_point_transform(gray, squares[0].reshape((4, 2)))
				tag_angle, tag_id = decode_tag(warped)
				P = Projection_matrix(M)

if __name__=='__main__':
	#superimpose()
	decodeTag()
	#lena = cv2.imread('data/Lena.png')
	#lena = cv2.resize(lena,(200,200))
	#M = np.array([0.5531129032258064,0.08637096774193548,1053,-0.04280645161290323,0.25533870967741934,552,0.00018548387096774192,-0.000011290322580645161,1])
	#M =M.reshape((3,3))
	#dst = warpPerspective(lena,M,(512,512))
	#cv2.imshow('dst',dst.astype(np.uint8))
	#cv2.waitKey(0)