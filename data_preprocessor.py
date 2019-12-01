# -*- coding: utf-8 -*-


""" ©Prem Prakash
	Data preprocessor
"""

import os
import sys
import pdb

import random
from copy import deepcopy


import pickle
import pandas as pd
import numpy as np


from transformers import NormalizeImageData
import util
import fr_util as futil

#----------------------------------------------------------------------------

class FeatureExtractor(object):
	"""Extracts features for images and saves them in CSV along with division of train/dev/test sets
	   For pair of same and different images
	"""
	def __init__(self, train_pair_paths, test_pair_paths):
		super(FeatureExtractor, self).__init__()
		self.train_same_pairspath, self.train_diff_pairspath = train_pair_paths
		self.test_same_pairspath, self.test_diff_pairspath = test_pair_paths

		self.data_normalizer = NormalizeImageData(means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]) # TODO: # Replace with local calculation of mean and stds
		self.features_extractor_model =  futil.get_pretrained_feature_extractor()
		self.face_cropper_model = futil.get_pretrained_face_cropper()
		self.model_img_size = futil.get_model_img_size()

	def get_extracted_features(self, rel_path, is_cropped=False):

		if is_cropped:
			features = futil.extract_features_after_cropping_face(
				img_path=util.get_full_imgpath_given_relpath(rel_path), 
				face_cropper_model=self.face_cropper_model, 
				features_extractor_model=self.features_extractor_model, 
				type_tensor=False)
		else:
			features = futil.extract_features_with_nomalization(
							img_path=util.get_full_imgpath_given_relpath(rel_path), 
							model_size=self.model_img_size, 
							data_normalizer=self.data_normalizer, 
							features_extractor_model=self.features_extractor_model, 
							type_tensor=False
						)

		return features


	def construct_same_features(self, df_same, feature_len=512, break_index=-1, is_cropped=False):
		local_pair_dicts = {}
		df = pd.DataFrame(columns=['f' + str(i) for i in range(feature_len)] + ['rel_path'])
		df = df.set_index('rel_path')

		# dir_name,img1,img2
		for index, row in df_same.iterrows():

			# pdb.set_trace()
			if index % 20 == 0:
				print(f'[Same] [i/n] = {index}/{df_same.shape[0]}')
			if index == break_index:
				break
			rel_path1 = util.get_relative_imgpath(row.dir_name, row.img1)
			rel_path2 = util.get_relative_imgpath(row.dir_name, row.img2)

			key = (rel_path1, rel_path2)
			if key in local_pair_dicts.keys():
				print(f'[Same key] = {key}: why the ... key already exist')
			local_pair_dicts[key] = futil.get_pair_encoding()['same'] # {'same': 0, 'diff': 1}

			img1_feature = self.get_extracted_features(rel_path1, is_cropped)
			img2_feature = self.get_extracted_features(rel_path2, is_cropped)

			
			df.loc[rel_path1] = list(img1_feature.reshape(-1))
			df.loc[rel_path2] = list(img2_feature.reshape(-1))
			
		
		return df, local_pair_dicts


	# dir_name1,img1,dir_name2,img2
	def construct_diff_features(self, df_diff, feature_len=512, break_index=-1, is_cropped=False):
		local_pair_dicts = {}
		df = pd.DataFrame( columns=['f' + str(i) for i in range(feature_len)] + ['rel_path'])
		df = df.set_index('rel_path')

		
		# dir_name,img1,img2
		for index, row in df_diff.iterrows():
			# pdb.set_trace()
			if index % 20 == 0:
				print(f'[Diff] [i/n] = {index}/{df_diff.shape[0]}')
			if index == break_index:
				break

			rel_path1 = util.get_relative_imgpath(row.dir_name1, row.img1)
			rel_path2 = util.get_relative_imgpath(row.dir_name2, row.img2)

			key = (rel_path1, rel_path2)
			if key in local_pair_dicts.keys():
				print(f'[Diff key] = {key}: why the ... key already exist')
				
			local_pair_dicts[key] = futil.get_pair_encoding()['diff'] # {'same': 0, 'diff': 1}


			img1_feature = self.get_extracted_features(rel_path1, is_cropped)
			img2_feature = self.get_extracted_features(rel_path2, is_cropped)

			df.loc[rel_path1] = list(img1_feature.reshape(-1))
			df.loc[rel_path2] = list(img2_feature.reshape(-1))


		return df, local_pair_dicts


	def construct_and_save(self, df_same, df_diff, fpath, dict_path, ftype):
		break_index = -1
		print(f'\n\n[{ftype}]')
		df_fs_same, pairs_dict_same = self.construct_same_features(df_same, break_index=break_index)
		df_fs_diff, pairs_dict_diff = self.construct_diff_features(df_diff, break_index=break_index)
		
		# pdb.set_trace()
		# TODO: Handle repeatation of row
		df = df_fs_same.append(df_fs_diff) # ignore_index=False becuse you want to keep index while appending
		df = df.reset_index()
		df = df.sample(frac=1).reset_index(drop=True)
		pairs_dict = {**pairs_dict_same, **pairs_dict_diff}

		df.to_csv(fpath, index=False)
		util.PickleHandler.dump_in_pickle(pairs_dict, dict_path)



	def tabulate_image_extracted_features(self, is_cropped=False, ext=''):

		break_index = -1

		
		df_train_same_temp, df_train_diff_temp = pd.read_csv(self.train_same_pairspath), pd.read_csv(self.train_diff_pairspath)
		# For validation shuffle and splice
		df_train_same_temp = df_train_same_temp.sample(frac=1).reset_index(drop=True)
		df_train_diff_temp = df_train_diff_temp.sample(frac=1).reset_index(drop=True)

		if (df_train_same_temp.shape[0]) > 1024:
			train_len  = 1024
		else:
			train_len = int(df_train_val.shape[0] * 0.9)

		df_train_same, df_train_diff = df_train_same_temp.iloc[:train_len], df_train_diff_temp.iloc[:train_len]
		
		df_val_same, df_val_diff = df_train_same_temp.iloc[train_len:], df_train_diff_temp.iloc[train_len:]
		df_val_same, df_val_diff = df_val_same.reset_index(), df_val_diff.reset_index()
		
		df_test_same, df_test_diff = pd.read_csv(self.test_same_pairspath), pd.read_csv(self.test_diff_pairspath)

		# pdb.set_trace()
		self.construct_and_save(df_train_same, df_train_diff, util.get_train_feature_path(ext), util.get_train_pairs_imgnames_relpath_dict(ext), ftype='Train')
		self.construct_and_save(df_val_same, df_val_diff, util.get_val_feature_path(ext), util.get_val_pairs_imgnames_relpath_dict(ext), ftype='Val')
		self.construct_and_save(df_test_same, df_test_diff, util.get_test_feature_path(ext), util.get_test_pairs_imgnames_relpath_dict(ext), ftype='Test')
		


#----------------------------------------------------------------------------

if __name__ == '__main__':
	print('[Preprocessor module] ')
	train_pair_paths = util.get_train_pair_paths()
	test_pair_paths = util.get_test_pair_paths()

	fext = FeatureExtractor(train_pair_paths, test_pair_paths)
	fext.tabulate_image_extracted_features()

	print('[Complete] Feature extraction Complete')	


#----------------------------------------------------------------------------

