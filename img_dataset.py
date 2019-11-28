# -*- coding: utf-8 -*-

import os
import pdb
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import util 
import fr_util as futil

K_INDEX_COL = 'rel_path'

class ImageDataset(Dataset):
	"""docstring for ImageDataset"""
	def __init__(self, feature_path, pairs_imgnames_relpath_dict, type_chi=True):
		super(ImageDataset, self).__init__()		
		self.df_features = pd.read_csv(feature_path)
		self.df_features = self.df_features.set_index(K_INDEX_COL)
		self.image_pair_rel_path_dict = util.PickleHandler.extract_from_pickle(pairs_imgnames_relpath_dict)
		self.rel_path_keys = list(self.image_pair_rel_path_dict.keys())
		random.shuffle(self.rel_path_keys)
		random.shuffle(self.rel_path_keys)

		self.type_chi = type_chi


	def __len__(self):
		return len(self.rel_path_keys)


	def __getitem__(self, index):
		
		# TODO: Data augmentation
		rel_path1, rel_path2 = self.rel_path_keys[index]
		# pdb.set_trace()
		label = self.image_pair_rel_path_dict[(rel_path1, rel_path2)]
		# pdb.set_trace()
		feature1 = self.df_features.loc[rel_path1].values.reshape(-1, 512)[0, :]
		feature2 = self.df_features.loc[rel_path2].values.reshape(-1, 512)[0, :]

		feature = futil.compute_diff_vector(feature1.reshape(-1), feature2.reshape(-1), type_chi=self.type_chi)

		# feature = feature.reshape(-1)
		return feature, np.array(label).reshape(-1)













