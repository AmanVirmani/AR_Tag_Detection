import cv2
import numpy as np
from collections import deque

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

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
	tag_gray = cv2.cvtColor(tag,cv2.COLOR_BGR2GRAY)
	ret, tag = cv2.threshold(tag_gray,128,255,0)
	count_white = cv2.countNonZero(tag)
	return count_white > 0.5*tag.size

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
	pts1 = order_points(np.float32([[0,0],[200,0],[0,200],[200,200]]))
	pts2 = np.float32(order_points(pts))
	M = find_homography(pts1,pts2)
	dst = warpPerspective(lena,M,(frame.shape[1],frame.shape[0]))
	overlay = cv2.add(frame,dst)
	return overlay

def to_img(mtr):
	V,H,C = mtr.shape
	img = np.zeros((H,V,C), dtype='int')
	for i in range(mtr.shape[0]):
		img[:,i] = mtr[i]
	return img.astype(np.uint8)

def to_mtx(img):
	H,V,C = img.shape
	mtr = np.zeros((V,H,C), dtype='int')
	for i in range(img.shape[0]):
		mtr[:,i] = img[i]
	return mtr

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	print(rect.shape)
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
	H = find_homography(rect,dst)
	warped = warpPerspective(image, H, (maxWidth, maxHeight))
	return warped

def warpPerspective(img, M, dsize=None):
	mtr = to_mtx(img)
	R,C = dsize
	dst = np.zeros((R,C,mtr.shape[2]))
	for i in range(mtr.shape[0]):
		for j in range(mtr.shape[1]):
			res = np.dot(M, [i,j,1])
			i2,j2,_ = (res / res[2] + 0.5).astype(int)
			if i2 >= 0 and i2 < R:
				if j2 >= 0 and j2 < C:
					for splat in range(3):
						try :
							dst[i2+splat,j2+splat] = mtr[i,j]
						except :
							print('oops')
						try:
							dst[i2-splat,j2-splat] = mtr[i,j]
						except:
							print('oops')

	return to_img(dst)

def projectCube():
	images= []
	cap = cv2.VideoCapture('./data/Tag0.mp4')
	#out = cv2.VideoWriter('projectCube.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
	if (cap.isOpened()== False):
		print("Error opening video stream or file")

	three_d_axis = np.float32([[0, 0, 0], [0, 500, 0], [500, 500, 0], [500, 0, 0], [0, 0, -300], [0, 500, -300], [500, 500, -300], [500, 0, -300]])
	count = -1
	while(cap.isOpened()):
		count += 1
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

			print(squares)
			std_size = 200
			cube_list = []
			for pts in squares:
				src_pts = order_points(pts.reshape((4,2)))
				tgt_pts = order_points(np.array([[0,0],[0,std_size],[std_size,std_size],[std_size,0]]))
				H = find_homography(src_pts, tgt_pts)
				print(H)
				#tag_warped = four_point_transform(frame,src_pts)
				#tag_angle, tag_id = decode_tag(tag_warped)
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

					cv2.imshow("Lena", final_image)
					cv2.waitKey(1)
	cv2.destroyAllWindows()
	print('done')

def draw_cube(img, imgpts):  # To draw the cube
	imgpts = np.int32(imgpts).reshape(-1, 2)
	# draw ground floor in green
	img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 255), 3)

	# draw pillars in blue color
	for i, j in zip(range(4), range(4, 8)):
		img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 0), 3)

	# draw top layer in red color
	img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
	return img

def calculator(h):
	K = np.array(
		[[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]]).T
	h = np.linalg.inv(h)
	inv_K = np.linalg.inv(K)
	b_new = np.dot(inv_K, h)
	b1 = b_new[:, 0].reshape(3, 1)
	b2 = b_new[:, 1].reshape(3, 1)
	r3 = np.cross(b_new[:, 0], b_new[:, 1])
	b3 = b_new[:, 2].reshape(3, 1)
	L = 2 / (np.linalg.norm((inv_K).dot(b1)) + np.linalg.norm((inv_K).dot(b2)))
	r1 = L * b1
	r2 = L * b2
	r3 = (r3 * L * L).reshape(3, 1)
	t = L * b3
	r = np.concatenate((r1, r2, r3), axis=1)

	return r, t, K

def get_krt_matrix(inv_h):
	k_mat = np.array(
		[[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]]).T
	inv_k_mat = np.linalg.inv(k_mat)
	b_mat = np.matmul(inv_k_mat, inv_h)
	b1 = b_mat[:, 0].reshape(3, 1)
	b2 = b_mat[:, 1].reshape(3, 1)
	r3 = np.cross(b_mat[:, 0], b_mat[:, 1])
	b3 = b_mat[:, 2].reshape(3, 1)
	scalar = 2 / (np.linalg.norm(inv_k_mat.dot(b1)) + np.linalg.norm(inv_k_mat.dot(b2)))
	t = scalar * b3
	r1 = scalar * b1
	r2 = scalar * b2
	r3 = (r3 * scalar * scalar).reshape(3, 1)
	r_mat = np.concatenate((r1, r2, r3), axis=1)
	return r_mat, t, k_mat

if __name__=='__main__':
	projectCube()