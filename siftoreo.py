#environment ml2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from math import asin, acos, cos, sin
from scipy.spatial.transform import Rotation as R
import glob
import os 
import yaml
import re
from mpl_toolkits.mplot3d import Axes3D
import time

t0 = time.clock()


MIN_MATCH_COUNT = 6
input_file_name = ['oreo,250,90,210,1.png','oreo,-250,90,210,2.png','oreo,250,-90,210,3.png','oreo,250,90,210,4.png','oreo,250,90,-210,5.png','oreo,250,90,210,6.png']

K = np.array([[575.8157348632812, 0.0, 319.5],
		[0.0, 575.8157348632812, 239.5],
		[0.0, 0.0, 1.0]])
dist = np.zeros((4,1))

image_list = []
image_mask = []
yml_pose = []
imagedep = []

image_list_every4 = []
mask_list_every4 = []
pose_list_every4 = []
imagedep_list_every4 = []

count = 0
for filename in glob.glob('C:/Users/zszentim/Documents/ENGG6100Assign3/rutgers_apc_dataset/all_data/oreo_mega_stuf-image-?-?-?-?.png'):
	image_list.append(filename)
	count = count + 1
	if count%4 == 0:
		image_list_every4.append([image_list[-1],image_list[-2],image_list[-3],image_list[-4]])
	
count = 0
for filename in glob.glob('C:/Users/zszentim/Documents/ENGG6100Assign3/rutgers_apc_dataset/all_data/oreo_mega_stuf-mask-?-?-?-?.png'):
	image_mask.append(filename)
	count = count + 1
	if count%4 == 0:
		mask_list_every4.append([image_mask[-1],image_mask[-2],image_mask[-3],image_mask[-4]])

count = 0
for filename in glob.glob('C:/Users/zszentim/Documents/ENGG6100Assign3/rutgers_apc_dataset/all_data/oreo_mega_stuf-pose-?-?-?-?.yml'):
	yml_pose.append(filename)
	count = count + 1
	if count%4 == 0:
		pose_list_every4.append([yml_pose[-1],yml_pose[-2],yml_pose[-3],yml_pose[-4]])

count = 0
for filename in glob.glob('C:/Users/zszentim/Documents/ENGG6100Assign3/rutgers_apc_dataset/all_data/oreo_mega_stuf-depth-?-?-?-?.png'):
	imagedep.append(filename)
	count = count + 1
	if count%4 == 0:
		imagedep_list_every4.append([imagedep[-1],imagedep[-2],imagedep[-3],imagedep[-4]])

best_j = []
file = open("resultoreo.txt","w") 
for fileimg,filemask,pose,depthimg in zip(image_list_every4, mask_list_every4,pose_list_every4,imagedep_list_every4):

	a,b,c,clutter,e,f = fileimg[0].split("-")
	best_img = ''
	best_img_angle = 1000
	best_image = 0
	numj = 0
	len_good_list = 0
	best_diff_angle = 10000
	#best_diff_angle = 0
	smallest = 10000
	for test_img in input_file_name:
		#best_trans = np.zeros((3,3))
		
		#best_angle = 1000
		#print(test_img)
		try:
			item,z,y,t,last = test_img.split(",")
			flag,ext = last.split(".")

			flag = int(flag)
			xdep = int(z)
			zorigin = int(y)
			yorigin = int(t)




			img1 = cv.imread(test_img)          # queryImage
			#img1 = cv.GaussianBlur(img1,(3,3),cv.BORDER_DEFAULT) 
			img1 = cv.GaussianBlur(img1,(5,5),cv.BORDER_DEFAULT) 
			imglist1 = cv.imread(fileimg[0]) # trainImage
			imglist2 = cv.imread(fileimg[1]) # trainImage
			imglist3 = cv.imread(fileimg[2]) # trainImage
			imglist4 = cv.imread(fileimg[3]) # trainImage
			
			img2 = np.mean([imglist1,imglist2,imglist3,imglist4], axis=0)
			img2 = img2.astype(np.uint8)
			
			mask = cv.imread(filemask[0]) # trainImage
			mask = cv.normalize(mask,  mask, 0, 1, cv.NORM_MINMAX)
			#PATH = 'C:/Users/zszentim/Documents/ENGG6100Assign3/rutgers_apc_dataset/all_data/cheezit_big_original-pose-C-1-2-0.yml'

			img2 = img2*mask
			img1 = cv.normalize(img1,  img1, 0, 255, cv.NORM_MINMAX)
			img2 = cv.normalize(img2,  img2, 0, 255, cv.NORM_MINMAX)


			# Initiate SIFT detector
			sift = cv.xfeatures2d.SIFT_create()

			# find the keypoints and descriptors with SIFT
			kp1, des1 = sift.detectAndCompute(img1,None)
			kp2, des2 = sift.detectAndCompute(img2,None)

			# BFMatcher with default params
			bf = cv.BFMatcher()
			matches = bf.knnMatch(des1,des2,k=2)
			newm = []
			# Apply ratio test
			good = []
			good_without_list = []
			for m,n in matches:
				if m.distance < 0.75*n.distance: #0.6 for cheezit but I should try a higher value, might get better results
					good.append([m])
					good_without_list.append(m)
					
			#cv.drawMatchesKnn expects list of lists as matches.
			img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
			#cv.imshow('img3',img3)
			#cv.waitKey(0)

			if len(good)>MIN_MATCH_COUNT:
				src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_without_list ]).reshape(-1,1,2)
				dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_without_list ]).reshape(-1,1,2)
				dsrc_pts = np.float32([ kp1[m.queryIdx].pt for m in good_without_list ]).reshape(-1,2)
				ddst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_without_list ]).reshape(-1,2)

				newsrc_pts = np.zeros([len(dsrc_pts[:,0]),3],dtype=np.float)
				newdst_pts = np.zeros([len(ddst_pts[:,0]),2],dtype=np.float)
				if flag == 1:
					newsrc_pts[:,0] = xdep 
					newsrc_pts[:,2] = dsrc_pts[:,0] - zorigin
					newsrc_pts[:,1] = yorigin - dsrc_pts[:,1]
					newdst_pts[:,0] = 320 -ddst_pts[:,0]
					newdst_pts[:,1] = 240 - ddst_pts[:,1] 
				elif flag == 2:
					newsrc_pts[:,0] = xdep 
					newsrc_pts[:,2] = zorigin - dsrc_pts[:,0]
					newsrc_pts[:,1] = yorigin - dsrc_pts[:,1]
					newdst_pts[:,0] = 320 -ddst_pts[:,0]
					newdst_pts[:,1] = 240 - ddst_pts[:,1] 
				elif flag == 3:
					newsrc_pts[:,0] = dsrc_pts[:,0] - xdep 
					newsrc_pts[:,2] = zorigin
					newsrc_pts[:,1] = yorigin - dsrc_pts[:,1]
					newdst_pts[:,0] = 320 -ddst_pts[:,0]
					newdst_pts[:,1] = 240 - ddst_pts[:,1] 
				elif flag == 4:
					newsrc_pts[:,0] = xdep - dsrc_pts[:,0]
					newsrc_pts[:,2] = zorigin
					newsrc_pts[:,1] = yorigin - dsrc_pts[:,1]
					newdst_pts[:,0] = 320 -ddst_pts[:,0]
					newdst_pts[:,1] = 240 - ddst_pts[:,1] 
				elif flag == 5:
					newsrc_pts[:,0] = dsrc_pts[:,0] - xdep 
					newsrc_pts[:,2] = dsrc_pts[:,1] - zorigin 
					newsrc_pts[:,1] = yorigin 
					newdst_pts[:,0] = 320 -ddst_pts[:,0]
					newdst_pts[:,1] = 240 - ddst_pts[:,1] 
				elif flag == 6:
					newsrc_pts[:,0] = dsrc_pts[:,0] - xdep
					newsrc_pts[:,2] = zorigin - dsrc_pts[:,1]
					newsrc_pts[:,1] = yorigin 
					newdst_pts[:,0] = 320 -ddst_pts[:,0]
					newdst_pts[:,1] = 240 - ddst_pts[:,1] 
					
					
					
					
					
					
					
				(_,rotation_vector, translation_vector, inliers) = cv.solvePnPRansac(np.float32(newsrc_pts), np.float32(newdst_pts), np.float32(K), dist, reprojectionError=12)
				#_, rvecs, tvecs, inliers  = cv.solvePnPRansac(newsrc_pts, dst_pts, K,dist)
				M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,2.0)
				matchesMask = mask.ravel().tolist()
				h,w,d = img1.shape
				pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
				dst = cv.perspectiveTransform(pts,M)
				img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
				diff_angle = len(good)
			else:
				print( "Not enough matches are found - {}/{}".format(len(good_without_list), MIN_MATCH_COUNT) )
				matchesMask = None
				

			fs = cv.FileStorage(pose[0], cv.FILE_STORAGE_READ) 
			fr = fs.getNode("object_rotation_wrt_camera")
			frmat = fr.mat()
			
			rmatrix = cv.Rodrigues(rotation_vector)
			
			rnewt = np.transpose(rmatrix[0])
			#print(rnewt,frmat)
			r_ab = np.matmul(rnewt, frmat)

			diff_angle = np.rad2deg(np.arccos((np.trace(r_ab) - 1) / 2))
			#newmatrix = R.from_matrix(rmatrix[0])
			#newmatrix = newmatrix.as_euler('zyx', degrees=True)
			
			#small = np.sum(abs(newmatrix))
			
			
		except:
			print('Exception Occurred')
			continue
			
		#print(diff_angle)
		#if small < smallest:
		#if diff_angle < best_diff_angle:
		if len(good) > len_good_list:
			len_good_list = len(good)
			#print(rmatrix[0])
			#smallest = small
			#best_diff_angle = diff_angle
			best_img_angle = rmatrix[0]
			best_img = test_img
			best_image = dst

	
	
	fs = cv.FileStorage(pose[0], cv.FILE_STORAGE_READ) 
	fr = fs.getNode("object_rotation_wrt_camera")
	frmat = fr.mat()
	#print(best_img_angle)
	#rmatrix = cv.Rodrigues(best_img_angle)
	
	rnewt = np.transpose(best_img_angle)
	#print(rnewt,frmat)
	r_ab = np.matmul(rnewt, frmat)

	diff_angle = np.rad2deg(np.arccos((np.trace(r_ab) - 1) / 2))
	#print(diff_angle)
	#if diff_angle < best_angle:
	#best_angle = diff_angle
	#numj = j
	
	
	#cheezit rotation calibration file for letters C is off by 180. Calibration file is wrong
	
	def find_between( s, first, last ):
		try:
			start = s.index( first ) + len( first )
			end = s.index( last, start )
			return s[start:end]
		except ValueError:
			return ""
		
	with open(pose[0]) as f:
		for line in f:
			match = find_between(line, "object_translation_wrt_camera: [", "]" )
			if match:
				trans = find_between(line, "object_translation_wrt_camera: [", "]" )



	trans= np.matrix(trans)

	#print('Hi')
	blank = np.zeros((480,640))
	rect = cv.polylines(blank,[np.int32(best_image)],True,255,1, cv.LINE_AA)
	# cv.imshow('image',rect)
	# cv.waitKey(0)

	cv.fillPoly(rect, pts =[best_image.astype(int)], color=(255))
	
	mom = cv.moments(rect)

	cX = int(mom["m10"] / mom["m00"])
	cY = int(mom["m01"] / mom["m00"])
	#print(cX,cY)
	imgdep = cv.imread(depthimg[3],0)   
	ztrans = imgdep[cY,cX]*0.01 + 0.64
	pixelhor = (ztrans*np.tan(np.deg2rad(31)))/320
	pixelver = (ztrans*np.tan(np.deg2rad(24.3)))/240
	#pixel = 0.0012044
	#ztrans = 0.69


	xtrans = (cX - 320)*pixelhor
	ytrans = (cY - 240)*pixelver
	transmat = np.array([xtrans,ytrans,ztrans])
	dist = np.linalg.norm(transmat-trans)
		
	file.write(pose[0] + "    " +best_img +'    '+ str(diff_angle)+ '    '+ 'b'+'    '+str(dist) + '    ' + 'c' + '    '+clutter+"\n")
	
#print(best_j)
	
	
t1 = time.clock()

total = t1-t0
print(total)
	
	
	
	
	
	
	
	
	