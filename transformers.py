# -*- coding: utf-8 -*-


import os
import pdb
import random

import numpy as np
import skimage.transform as sk_transformer
from skimage import color
from skimage.util import random_noise



#----------------------------------------------------------------------------

class RandomNoiseImage(object):
	"""Randomly noising image where input array is numpy.ndarray and of form H x W x C"""
	def __init__(self):
		super(RandomNoiseImage, self).__init__()
		self.modes = ['gaussian', 's&p', 'poisson']
		self.mode_kwargs = {
			'gaussian': {'mean': 0.1, 'var': 0.02},
			's&p': {'amount': 0.02, 'salt_vs_pepper': 0.5}, # {'amount': What fraction of pixel should be salt and peppered,  'salt_vs_pepper': what fraction is for salting and (1-salt_vs_pepper) for peppering) 
			'poisson': {} }

	@staticmethod
	def transform(X: np.ndarray):
		loc_transformer = RandomNoiseImage()
		return loc_transformer(X) 

	def __call__(self, X: np.ndarray):
		mode = random.choice(self.modes)
		mode_kwargs = self.mode_kwargs[mode]

		noised_img = random_noise(X, mode=mode, **mode_kwargs)

		# Convert back the intensity from float value to <uint8>
		X = (255 * noised_img).astype(np.uint8) 

		return X

		


#----------------------------------------------------------------------------

class MirrorImage(object):
	"""Horizontal flip of image where the input array is numpy.ndarray and of form H x W x C"""

	def __init__(self):
		super(MirrorImage, self).__init__()
		
	
	@staticmethod
	def transform(X: np.ndarray):
		return np.flip(X, axis=1)


	def __call__(self, X: np.ndarray):
		X = np.flip(X, axis=1).copy() # because width is at second dimension i.e. axis=1

		return X

#----------------------------------------------------------------------------

class InvertVerticallyImage(object):
	"""Vertical flip of image where the input array is numpy.ndarray and of form H x W x C

	"""
	def __init__(self):
		super(InvertVerticallyImage, self).__init__()

	
	@staticmethod
	def transform(X: np.ndarray):
		return np.flip(X, axis=0)


	def __call__(self, X: np.ndarray):
		"""Invert image vertically and object coordinates are transformed while constructing target
		"""

		X = np.flip(X, axis=0).copy() # because width is at second dimension i.e. axis=1
		return X


#----------------------------------------------------------------------------

class MirrorAndInvertVerticallyImage(object):
	"""Mirror/Horizontal and Vertical flip of image where the input array is numpy.ndarray and of form H x W x C """
	
	def __init__(self):
		super(MirrorAndInvertVerticallyImage, self).__init__()

	
	@staticmethod
	def transform(X):
		return np.flip(np.flip(X, axis=1), axis=0)


	def __call__(self, X: np.ndarray):
		"""Take mirror image and also invert vetically and object coordinates are transformed while constructing target """
		
		X = np.flip(np.flip(X, axis=1), axis=0).copy() # because width is at second dimension i.e. axis=1
		return X


#----------------------------------------------------------------------------

class Rotate90Image(object):
	"""Rotate image by 90 degree in clockwise (axes=(1,0)) the input array is numpy.ndarray and of form H x W x C
	    
	   Note: For now it only works if image is square else after rotation the image size and coordinated need to be recalibrated
	"""

	def __init__(self):
		super(Rotate90Image, self).__init__()
		

	@staticmethod
	def transform(X:np.ndarray):
		return np.rot90(X, k=1, axes=(1, 0)) 


	def __call__(self, X: np.ndarray):
		"""Rotate image by 270 degree and object coordinates are transformed while constructing target"""

		X = np.rot90(X, k=1, axes=(1, 0)).copy() # because width is at second dimension i.e. axis=1
		return X
		

#----------------------------------------------------------------------------

class Rotate270Image(object):
	"""Rotate image by 270 degree in clockwise (axes=(1,0)) the input array is numpy.ndarray and of form H x W x C
	    
	   Note: For now it only works if image is square else after rotation the image size and coordinated need to be recalibrated
	"""
	def __init__(self):
		super(Rotate270Image, self).__init__()
		
	
	@staticmethod
	def transform(X: np.ndarray):
		return np.rot90(X, k=3, axes=(1, 0)) 


	def __call__(self, X: np.ndarray):
		"""Rotate image by 270 degree and object coordinates are transformed while constructing target"""

		X = np.rot90(X, k=3, axes=(1, 0)).copy() # because width is at second dimension i.e. axis=1
		return X
	

#----------------------------------------------------------------------------

class RandomColorShifter(object):
	"""Shift the color of image by taking average of randomly selected two of the channels (from RGB) and replacing with it, 
	   and nullify i.e. replace with zero the remaining channel
	   For example averaging Red (R) and Blue (B) and nullifying the Green (G) will make the image look purply 
	"""

	def __init__(self):
		super(RandomColorShifter, self).__init__()
		

	@staticmethod
	def transform(X: np.ndarray):
		"""Shift the RGB gradient of image by randomly selecting one of RGB channel to set it zero and
		   average of other two.					
		"""
		
		X.setflags(write=1)

		color_indices = {0, 1, 2}
		sub_index = np.random.randint(3)

		add_indices = list(color_indices - {sub_index})

		added_color_grd = np.sum(X[:, :, add_indices], axis=2) / 2
		added_color_grd = added_color_grd.astype(int)

		for id in add_indices:
			X[:, :, id] = added_color_grd

		# X = X - X[:, :, sub_index:sub_index+1]
		X[:, :, sub_index] = 0
		X[X < 0] = 0

		return X
		
	def __call__(self, X: np.ndarray):
		""" Shift the RGB gradient of image 		
		"""
		return RandomColorShifter.transform(X)


#----------------------------------------------------------------------------

class ChangeHueImage(object):
	"""Randomly change hue of the image"""

	def __init__(self, low=-0.5, high=0.5):
		super(ChangeHueImage, self).__init__()
		self.low = low
		self.high = high
		
	@staticmethod
	def transform(X: np.ndarray, low=-0.5, high=0.5):
		""" Change the hue of image which is selection of RGB gradient meaning which part will dominate i.e. from R(0), G(0.33), B(0.67) and then back to red	
		"""

		# pdb.set_trace()
		X = X.astype(np.float32)
		hue = np.random.uniform(low, high, 1).round(2)[0]
		# print('hue = ', hue)
		hsv = color.rgb2hsv(X)
		hsv[:, :, 0] += hue
		hsv[:, :, 0] = hsv[:, :, 0].clip(min=0, max=1)
		# hsv[:, :, 1] = 1  # Turn up the saturation; we want the color to pop!
		X = color.hsv2rgb(hsv)
		X = X.astype(np.uint8)
		return X


	def __call__(self, X, y):
		return ChangeHueImage.transform(X, self.low, self.high), y


#----------------------------------------------------------------------------

class ChangeSaturation(object):
	"""Randomly change saturation of the image"""

	def __init__(self, low=-0.3, high=0.3):
		super(ChangeSaturation, self).__init__()
		self.low = low
		self.high = high

	@staticmethod
	def transform(X: np.ndarray, low=-0.3, high=0.3):
		""" Change the saturation of image color which is intensity of color meaning 0: Black, 1: Full intensity of that color	
		"""

		X = X.astype(np.float32)

		saturation = np.random.uniform(low, high, 1).round(2)[0]

		hsv = color.rgb2hsv(X)
		# hsv[:, :, 0] = 1
		hsv[:, :, 1] += saturation  
		hsv[:, :, 1] = hsv[:, :, 1].clip(min=0, max=1)

		X = color.hsv2rgb(hsv)
		X = X.astype(np.uint8)
		return X


	def __call__(self, X: np.ndarray):
		return ChangeSaturation.transform(X, self.low, self.high)


#----------------------------------------------------------------------------

class ChangeLuminescenceImage(object):
	"""docstring for ChangeLuminescenceImage"""
	def __init__(self, low=-20, high=90):
		super(ChangeLuminescenceImage, self).__init__()
		self.low = low
		self.high = high

	@staticmethod
	def transform(X: np.ndarray, low=-20, high=90):
		""" Change the saturation of image brightness which is how bright is of colors in the image 0: Black, 1: towards white and 0.5 is original color will be retained i.e. no brightness added or removed
		"""
		X = X.astype(np.float32)

		luminescence = np.random.randint(low, high)
		hsv = color.rgb2hsv(X)
		hsv[:, :, 2] += luminescence
		hsv[:, :, 2] = hsv[:, :, 2].clip(min=60, max=250)  # Normally it also between 0, 1 but becaue of <X = X.astype(np.float32)> it gets between 0 and 255

		X = color.hsv2rgb(hsv)
		X = X.astype(np.uint8)
		return X


	def __call__(self, X: np.ndarray):
		return ChangeLuminescenceImage.transform(X, self.low, self.high)


#----------------------------------------------------------------------------

class ChangeHSLImage(object):
	"""docstring for ChangeHSLImage"""
	def __init__(self):
		super(ChangeHSLImage, self).__init__()

	@staticmethod
	def transform(X: np.ndarray):
		""" Randomly manipulate Hue(H), Saturation(S), Luminescence(L) of image. See change above to check how individually them impact the image 
		"""

		# pdb.set_trace()
		X = X.astype(np.float32)
		hue = np.random.uniform(-0.2, 0.6, 1).round(2)[0]
		saturation = np.random.uniform(-0.2, 0.2, 1).round(2)[0]
		luminescence = np.random.randint(-20, 90)

		# print('hue = {}, saturation= {},  luminescence = '.format(hue , saturation, luminescence))
		
		hsv = color.rgb2hsv(X)
		hsv[:, :, 0] += hue
		hsv[:, :, 1] += saturation
		hsv[:, :, 2] += luminescence

		hsv[:, :, 0:2] = hsv[:, :, 0:2].clip(min=0, max=1)  
		hsv[:, :, 2] = hsv[:, :, 2].clip(min=60, max=250)  

		X = color.hsv2rgb(hsv)
		X = X.astype(np.uint8)
		return X


	def __call__(self, X, y):
		return ChangeLuminescenceImage.transform(X), y


#----------------------------------------------------------------------------

class NormalizeImageData(object):
	"""Normalize the image data per channel for quicker and better training (H x W x C)
	   TODO: Normalization value should computed locally for now it on Imagenet data 	
	"""
	# [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] is computed from imagenet
	def __init__(self, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
		self.means = np.array(means)
		self.stds = np.array(stds)
	
	def __call__(self, X: np.ndarray):

		assert len(X.shape) == 3, '[Assertion Error]: Normalization is performed for each channel so input must be only one image data only'
		# img_tensor = torch.tensor(img_ndarray)
		# img_tensor = img_tensor.contiguous()
		X = X / 255

		# If normalization is done idividually but not a good idea 
		# mean_list = img_tensor.view(img_tensor.shape[0], -1).mean(dim=1)
		# std_list = img_tensor.view(img_tensor.shape[0], -1).std(dim=1)
		# normalizer = torch_transformer.Normalize(mean=self.means, std=self.stds)

		X = (X - self.means) / self.stds 

		return X





























