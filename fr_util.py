# -*- coding: utf-8 -*-

import sys
import os
import pdb

# Cosine similarity 
import numpy as np
from numpy import dot
from numpy.linalg import norm
from PIL import Image

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1




#******************** Constants
K_MODEL_IMAGE_SIZE = (160, 160) # (Width, Height) # for now it is same as true but should check (256, 256)
get_model_img_size = lambda: K_MODEL_IMAGE_SIZE

K_PAIR_ENCODING = {'same': 0, 'diff': 1}
get_pair_encoding = lambda: K_PAIR_ENCODING




# Cosine similarity
cosine_similarity = lambda a, b: dot(a, b) / ((norm(a) * norm(b)))


#----------------------------------------------------------------------------

# Resize image
def load_and_resize_image(img_path, model_image_size):
	"""Reads the image and resizes it.

		Parameters:
		-----------
			img_path (str): fullpath to the image where it is located.
			model_image_size (tuple): the dimension (width, height) of image which goes to model. 
									  Note: here that pil-images have first dimension width and second height 

		Returns:
		--------
			image_data (numpy.ndarray): the resized image in shape (H x W x C)
	"""

	image = Image.open(img_path)
	if image.mode == 'RGBA':
		# https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
		background = Image.new("RGB", image.size, (255, 255, 255))
		background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
		image = background
	
	resized_image = image.resize(model_image_size, Image.BICUBIC) # NOTE:  (width, height).image.resize(model_image_size, Image.BICUBIC)
	image_data = np.array(resized_image, dtype='float32') #  this converts to (height x width x channel)
	
	# true_img_size = image.size
	return image_data


#----------------------------------------------------------------------------

# Load the pretrained model
def get_pretrained_face_cropper():
	mtcnn = MTCNN()
	return mtcnn


#----------------------------------------------------------------------------

def get_pretrained_feature_extractor():
	"""Downloads pre-trained network for feature extraction  
	"""
	resnet = InceptionResnetV1(pretrained='vggface2').eval()
	resnet.classify = False # This will allow to only return feature and not  the output of logit layer

	return resnet


#----------------------------------------------------------------------------

def get_feature_extractor_info():
	"""Return tuple of pretrained feature extractor and its best-input image size for the extractor"""
	return get_pretrained_feature_extractor(), K_MODEL_IMAGE_SIZE

#----------------------------------------------------------------------------

def extract_features_with_nomalization(img_path, model_size, data_normalizer, features_extractor_model, type_tensor=False):
	"""Resizes the data according to model-size, normalizes it and then
	   extract feature using InceptionResnetV1	 
	"""
	img = load_and_resize_image(img_path, model_size)
	img_norml = data_normalizer(img) 
	img_tensor = torch.tensor(img_norml)
	img_tensor = img_tensor.permute(2, 0, 1)

	img_feature = features_extractor_model(img_tensor.unsqueeze(0).float())
	if type_tensor:
		return img_feature.detach()
	else:
		return img_feature.detach().numpy()


def extract_features_after_cropping_face(img_path, face_cropper_model, features_extractor_model, type_tensor=False):
	"""First crop the face using <MTCNN> module and then
	   Extract feature using InceptionResnetV1	 
	"""
	print('[Extractor:img_path] = ', img_path)
	img = Image.open(img_path)

	img_cropped = face_cropper_model(img)
	if img_cropped is None:
		pdb.set_trace()

	img_feature = features_extractor_model(img_cropped.unsqueeze(0))

	if type_tensor:
		return img_feature.detach()
	else:
		return img_feature.detach().numpy()


#----------------------------------------------------------------------------

def extract_features(data_tensor, feature_extractor_model, type_tensor=True):
	""" Extracte features 
	"""
	return feature_extractor_model(data_tensor.unsqueeze(0).detach().squeeze()) # 1D return 


#----------------------------------------------------------------------------

def compute_diff_vector(img1_feature, img2_feature, type_chi=True):
	""" [Deep Face Recognition: A Survey](https://arxiv.org/abs/1804.06655)
		To use chi distribution on pair of images. 
	"""
	if type_chi:
		return ((img1_feature - img2_feature)**2) / (img1_feature + img2_feature)
	else:
		return img1_feature - img2_feature




















