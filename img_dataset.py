# -*- coding: utf-8 -*-

import os
import pdb
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import NormalizeImageData

import util 
import fr_util as futil 


K_INDEX_COL = 'rel_path'


#----------------------------------------------------------------------------

class ImageDataset(Dataset):
	"""docstring for ImageDataset"""
	
	def __init__(self, feature_path, pairs_imgnames_relpath_dict, transformers=None, use_transformer_flag=False, type_chi=True):
		super(ImageDataset, self).__init__()		
		
		self.df_features = pd.read_csv(feature_path)
		self.df_features = self.df_features.set_index(K_INDEX_COL)
		
		self.image_pair_rel_path_dict = util.PickleHandler.extract_from_pickle(pairs_imgnames_relpath_dict)
		self.rel_path_keys = list(self.image_pair_rel_path_dict.keys())
		random.shuffle(self.rel_path_keys)
		random.shuffle(self.rel_path_keys)

		self.transformers = transformers
		# self.transformers_len = len(transformers) if transformers is not None else 0
		self.use_transformer_flag = use_transformer_flag 
		if transformers is None:
			self.use_transformer_flag = False

		self.features_extractor_model, self.features_extractor_input_model_size = futil.get_feature_extractor_info()

		self.normalizer = NormalizeImageData()
		self.type_chi = type_chi


	def __len__(self):
		return len(self.rel_path_keys)


	def get_transformed_features(self, img_rel_path):
		"""Extracts features using extractor"""
		img_path = util.get_full_imgpath_given_relpath(img_rel_path)
		trasform_func = random.choice(self.transformers)
		# print('\ntrasform_func = ', trasform_func)

		img = futil.load_and_resize_image(img_path, self.features_extractor_input_model_size)
		img = trasform_func(img)

		img_norml = self.normalizer(img) 

		img_tensor = torch.tensor(img_norml)

		img_tensor = img_tensor.permute(2, 0, 1)
		img_feature = self.features_extractor_model(img_tensor.unsqueeze(0).float())

		img_feature =  img_feature.detach().numpy()
		# pdb.set_trace()
		return img_feature


	def __getitem__(self, index):
		

		rel_path1, rel_path2 = self.rel_path_keys[index] # rel_path_keys has tuple as dict
		label = self.image_pair_rel_path_dict[(rel_path1, rel_path2)]

		transforming_chance1 = np.random.randint(2) 
		transforming_chance2 = np.random.randint(2)

	
		if self.use_transformer_flag and transforming_chance1:
			feature1 = self.get_transformed_features(rel_path1, )
		else:
			feature1 = self.df_features.loc[rel_path1].values.reshape(-1, 512)[0, :]
			# print('[Loaded] feature1.dtype = ', feature1.dtype)


		if self.use_transformer_flag and transforming_chance2:
			feature2 = self.get_transformed_features(rel_path2)
		else:
			feature2= self.df_features.loc[rel_path2].values.reshape(-1, 512)[0, :]
			# print('[Loaded] feature2.dtype = ', feature2.dtype)



		feature = futil.compute_diff_vector(feature1.reshape(-1), feature2.reshape(-1), type_chi=self.type_chi)
		# print('\n\nfeature.dtype = ', feature.dtype)
		return feature.astype('float32'), np.array(label).reshape(-1)













