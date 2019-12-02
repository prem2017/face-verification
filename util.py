# -*- coding: utf-8 -*-

# ©Prem Prakash
# General util methods


import os
import sys
import pdb
import random
import logging

import pickle
import pandas as pd
import numpy as np
from PIL import Image
from skimage.util import random_noise

import torch

# For device agnostic 
get_training_device =  lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#---------------------------------------------------------------------------- Constants

K_SEARCH_LR  = False
get_search_lr_flag = lambda: K_SEARCH_LR

K_USE_PRETRAINED = True
get_use_pretrained_flag = lambda: K_USE_PRETRAINED 


#----------------------------------------------------------------------------  Batches

K_TRAIN_BATCH_SIZE = 64
get_train_batch_size = lambda: K_TRAIN_BATCH_SIZE

K_VALIDATION_BATCH_SIZE = 16
get_val_batch_size = lambda: K_VALIDATION_BATCH_SIZE

K_TEST_BATCH_SIZE = 64
get_test_batch_size = lambda: K_TEST_BATCH_SIZE


#---------------------------------------------------------------------------- Data

K_PROJECT_DIR =  os.path.dirname(os.path.abspath(__file__)) # os.path.dirname(os.getcwd())
get_project_dir = lambda: K_PROJECT_DIR


K_DATA_DIR = os.path.join(K_PROJECT_DIR, 'data')
get_root_data_dir = lambda: K_DATA_DIR

K_IMAGE_DIR_NAME = 'images'
get_image_dir = lambda: os.path.join(get_root_data_dir(), K_IMAGE_DIR_NAME)


#----------------------------------------------------------------------------

def construct_image_suffix(img_num, max_len=4):
	"""
		Examples:
			input: (4, 4) => output: '0004'
			input: (10, 4) => output: '0010'  
	"""
	last_suffix = str(img_num)
	img_suffix = ''.join(['0'] * (max_len - len(last_suffix))) + last_suffix
	return img_suffix


#----------------------------------------------------------------------------

def get_full_imgpath(dirname, img_num, ext='.jpg'):
	img_dir = os.path.join(get_image_dir(), dirname)
	full_img_name = dirname + '_' + construct_image_suffix(img_num) + ext
	return os.path.join(img_dir, full_img_name)


#----------------------------------------------------------------------------

def get_relative_imgpath(dirname, img_num, ext='.jpg'):
	full_img_name = dirname + '_' + construct_image_suffix(img_num) + ext
	return os.path.join(dirname, full_img_name)


#----------------------------------------------------------------------------

def get_full_imgpath_given_relpath(rel_path):
	return os.path.join(get_image_dir(), rel_path) # rel_path: 'dirname/image_name.jpg'


#---------------------------------------------------------------------------- T/V/T Dir Datapath

get_train_datapath = lambda: os.path.join(K_DATA_DIR, 'train')
get_val_datapath = lambda: os.path.join(K_DATA_DIR, 'val')
get_test_datapath = lambda: os.path.join(K_DATA_DIR, 'test')

# CSV files for train and test pars
# get_train_pairs_path = lambda: {'same': os.path.join(get_train_datapath(), 'same_pair_train.csv'),
# 								'diff': }
# get_test_pairs_path = lambda: {'same': os.path.join(get_test_datapath(), 'same_pair_test.csv'),
# 								'diff': os.path.join(get_test_datapath(), 'different_pair_test.csv')}

#----------------------------------------------------------------------------

get_train_pair_paths = lambda: (os.path.join(get_train_datapath(), 'same_pair_train.csv'), os.path.join(get_train_datapath(), 'different_pair_train.csv'))
get_test_pair_paths = lambda: (os.path.join(get_test_datapath(), 'same_pair_test.csv'), os.path.join(get_test_datapath(), 'different_pair_test.csv'))


#----------------------------------------------------------------------------

# Simple extracted featue path of cropped extracted feature path # 
get_train_feature_path = lambda ext='': os.path.join(get_train_datapath(), f'train_features{ext}.csv') 
get_val_feature_path = lambda ext='': os.path.join(get_val_datapath(), f'val_features{ext}.csv') 
get_test_feature_path = lambda ext='': os.path.join(get_test_datapath(), f'test_features{ext}.csv')


#---------------------------------------------------------------------------- Pickle Paths

# Dictionary store for t/v/t in the form of  {(rel_path, rel_path): label}
get_all_pairs_imgnames_relpath_dict = lambda ext='': os.path.join(get_root_data_dir(), f'all_pairs_imgnames_relpath_dict{ext}.pkl') 
get_train_pairs_imgnames_relpath_dict = lambda ext='': os.path.join(get_train_datapath(), f'train_pairs_imgnames_relpath_dict{ext}.pkl') 
get_val_pairs_imgnames_relpath_dict = lambda ext='': os.path.join(get_val_datapath(), f'val_pairs_imgnames_relpath_dict{ext}.pkl') 
get_test_pairs_imgnames_relpath_dict = lambda ext='': os.path.join(get_test_datapath(), f'test_pairs_imgnames_relpath_dict{ext}.pkl') 


#----------------------------------------------------------------------------  Models and Results

get_models_dir = lambda: os.path.join(K_PROJECT_DIR, 'models')

get_results_dir = lambda arg='': os.path.join(K_PROJECT_DIR, 'results' , arg)

K_TRAINED_MODELNAME = 'face_recog.model'


#----------------------------------------------------------------------------

def set_trained_model_name(ext_cmt=''):
	global K_TRAINED_MODELNAME
	K_TRAINED_MODELNAME = f"face_recog{ext_cmt}.model"
	return K_TRAINED_MODELNAME


#----------------------------------------------------------------------------

get_trained_model_name = lambda: K_TRAINED_MODELNAME





#---------------------------------------------------------------------------- Logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
def reset_logger(filename='train_output.log'):
	logger.handlers = []
	filepath = os.path.join(get_results_dir(), filename)
	logger.addHandler(logging.FileHandler(filepath, 'w'))


def add_logger(filename):
	filepath = os.path.join(get_results_dir(), filename)
	logger.addHandler(logging.FileHandler(filepath, 'w'))


def setup_logger(filename='output.log'):
	filepath = os.path.join(get_results_dir(), filename)
	logger.addHandler(logging.FileHandler(filepath, 'a'))




#---------------------------------------------------------------------------- Pickle py objects

class PickleHandler(object):    
	@staticmethod
	def dump_in_pickle(py_obj, filepath):
		"""Dumps the python object in pickle
			
			Parameters:
			-----------
				py_obj (object): the python object to be pickled.
				filepath (str): fullpath where object will be saved.
			
			Returns:
			--------
				None
		"""
		with open(filepath, 'wb') as pfile:
			pickle.dump(py_obj, pfile)
	
	
	
	@staticmethod
	def extract_from_pickle(filepath):
		"""Extracts python object from pickle
			
			Parameters:
			-----------
				filepath (str): fullpath where object is pickled
			
			Returns:
			--------
				py_obj (object): python object extracted from pickle
		"""
		with open(filepath, 'rb') as pfile:
			py_obj = pickle.load(pfile)
			return py_obj    


#----------------------------------------------------------------------------

import collections
def pretty(d, indent=0):
	""" Pretty printing of dictionary """
	ret_str = ''
	for key, value in d.items():

		if isinstance(value, collections.Mapping):
			ret_str = ret_str + '\n' + '\t' * indent + str(key) + '\n'
			ret_str = ret_str + pretty(value, indent + 1)
		else:
			ret_str = ret_str + '\n' + '\t' * indent + str(key) + '\t' * (indent + 1) + ' => ' + str(value) + '\n'

	return ret_str


#----------------------------------------------------------------------------

def salt_image(img_path):

	imp = Image.open(img_path)
	mode_imp = imp.mode
	im = np.array(imp)
	# pdb.set_trace()
	modes = ['gaussian', 's&p', 'poisson']
	kwargs = {
		'gaussian': {'mean': 0.1, 'var': 0.02},
		's&p': {'amount': 0.02, 'salt_vs_pepper': 0.5}, # {'amount': What fraction of pixel should be salt and peppered,  'salt_vs_pepper': what fraction is for salting and (1-salt_vs_pepper) for peppering) 
		'poisson': {}
	}

	mode = random.choice(modes)
	print('[mode] = ', mode)

	pdb.set_trace()
	noise_img = random_noise(im, mode=mode, **kwargs[mode])

	noise_img = (255 * noise_img).astype(np.uint8)

	imp_noised = Image.fromarray(noise_img, mode_imp)

	imp_noised.save(img_path[:-3] + '_noised.png')


#----------------------------------------------------------------------------  Test utitlity

if __name__ == '__main__':
	print('Util is the main module')
	print(K_PROJECT_DIR)
	print(get_full_imgpath('Pervez_Musharraf', 1))
	print(get_full_imgpath('Pervez_Musharraf', 2))
	print(get_full_imgpath('Pervez_Musharraf', 11))
	print(get_full_imgpath('Pervez_Musharraf', 112))

	print(get_relative_imgpath('Pervez_Musharraf', 11))
	print(get_relative_imgpath('Pervez_Musharraf', 112))


	print(os.path.join(get_root_data_dir(), 'images', get_relative_imgpath('Pervez_Musharraf', 11)))


#----------------------------------------------------------------------------



