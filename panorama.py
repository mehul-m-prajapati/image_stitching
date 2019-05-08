#!/usr/bin/env python3
import cv2
import numpy as np
import random
from scipy.interpolate import RectBivariateSpline


# Constants
INPUT_IMG_DIR = "./project_images/"
SIFT_OUT_IMG = "2.png"
RANSAC_OUT_IMG = "3.png"
STITCH_OUT_IMG = "4.png"
STITCH_OUT_ALL_IMG = "Output_AllStitched.png"
RATIO_TEST_THRESOLD = 0.77

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
		
    return frame


def get_sift_matches(img1, img2):
	"""
	
	SIFT descriptor
	
	"""	
	i1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	i2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create() 

	k1, des1 = sift.detectAndCompute(i1, None)
	k2, des2 = sift.detectAndCompute(i2, None)

	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)

	# Apply ratio test
	final_matches = []
	
	for u, v in matches:
		if u.distance < RATIO_TEST_THRESOLD * v.distance:
			final_matches.append([u])
	
	# cv2.drawMatchesKnn expects list of lists as matches.
	img_out = cv2.drawMatchesKnn(img1, k1, img2, k2, final_matches, None, flags=2)
	
	cv2.imwrite(SIFT_OUT_IMG, img_out)
	cv2.imshow("SIFT Matches", img_out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	flat_matches = np.asarray(final_matches)
	
	return flat_matches, k1, k2
	
	
def get_homography(img1, img2):
	"""
	
	Get best homography matrix after running RANSAC algorithm
	
	"""
	# ----------------------------------
	# 2. SIFT Operator
	# ----------------------------------
	# find descriptors with SIFT and match keypoints
	matches, k1, k2 = get_sift_matches(img1, img2)
	
	# ----------------------------------
	# 3. RANSAC
	# ----------------------------------	
	if len(matches[:,0]) >= 4:
		src = np.float32([ k1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		dst = np.float32([ k2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)

	else:
		raise AssertionError("Can't find enough keypoints.")  

	dst = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
	dst[0:img1.shape[0], 0:img1.shape[1]] = img1

	return trim(dst)
	
	
def stitch_N_images(img_ls, num):
	"""
	
	"""
	cnt = 1
	out_img_idx = 4
	orig_len = len(img_ls)
	
	if num < 1:
		print ("Need more than 1 image")
		return -1
	
	for idx in range(num-1):	
		
		img1 = img_ls.pop(0)
		img2 = img_ls.pop(0)

		# -------------
		#  Stitching
		# -------------
		op_img_12 = stitch_2_images(img1, img2, out_img_idx)
				
		out_img_idx = out_img_idx + 1
		img_ls.append(op_img_12)	
	
	return op_img_12

	
def stitch_2_images(img1, img2, out_img_key):
	"""
	
	Stitch and save image on to the disk.
	
	"""
	out_img_name = str(out_img_key) + ".png"
	
	# ----------------------------------
	# 4. Stitch images
	# ----------------------------------
	print("Stitching images now. Please wait for a while ...")

	stitch_img = get_homography(img1, img2)
		
	print("Stitched image output:", out_img_name)
	stitch_img = np.uint8(stitch_img)
	cv2.imwrite(out_img_name, stitch_img)
	cv2.imshow("Stitched", stitch_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return stitch_img

	
def read_input_images(num):
	"""
	
	Process user given images.
	
	"""
	img_list = []
	
	for idx in range(num):		
		# ---------------------------
		# ----- Read Images ---------
		# ---------------------------
		print("Enter image {} name: ".format(idx+1))
		input_img_name = input()
		
		# Read image
		color_img = cv2.imread(INPUT_IMG_DIR + input_img_name)

		if color_img is None:
			print("Exception: Image not found: ", input_img_name)
			exit(1)
		
		img_list.append(color_img)
	
	
	# Stitch all images
	all_stitched = stitch_N_images(img_list, len(img_list))
	cv2.imwrite(STITCH_OUT_ALL_IMG, all_stitched)
	
	
# Main
if __name__ == "__main__":
	
	# Start reading images
	print("How many input images?")
	n = input()
	total = int(n)
	read_input_images(total)
	