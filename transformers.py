# -*- coding: utf-8 -*-


import pdb
import os

import torch
import torchvision.transforms as torch_transformer


import numpy as np
import skimage.transform as sk_transformer




class NormalizeImageData(object):
	"""Normalize the image data per channel for quicker and better training 
	   TODO: Normalization value should computed locally for now it on Imagenet data 	
	"""
	# [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] is computed from imagenet
	def __init__(self, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
		self.means = np.array(means)
		self.stds = np.array(stds)
	
	def __call__(self, X):
		""" Normalize the image data

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				
			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray with shifted colors
		"""

		assert len(X.shape) == 3, '[Assertion Error]: Normalization is performed for each channel so input must be only one image data only'
		# img_tensor = torch.tensor(img_ndarray)
		# img_tensor = img_tensor.contiguous()
		X = X / 255

		# If normalization is done idividually 
		# mean_list = img_tensor.view(img_tensor.shape[0], -1).mean(dim=1)
		# std_list = img_tensor.view(img_tensor.shape[0], -1).std(dim=1)
		
		# normalizer = torch_transformer.Normalize(mean=self.means, std=self.stds)
		X = (X - self.means) / self.stds 

		return X

