# -*- coding: utf-8 -*-

# ©Prem Prakash
# Predictor module


import pdb
import os
import sys
from copy import deepcopy
import argparse

from matplotlib.pyplot import imshow
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve # (y_true, y_score)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import FaceNet, SiameseFaceNet
from img_dataset import ImageDataset, ImageDatasetPT
from transformers import NormalizeImageData

import util 
from util import logger
import fr_util as futil

device = util.get_training_device()


#----------------------------------------------------------------------------

def compute_label_and_prob(pred: torch.Tensor, th=0.5):
	"""Computes probability  and label/class with given threshold."""
	pred_prob = pred.sigmoid()
	label = deepcopy(pred_prob)
	label[label >= th] = 1
	label[label < th] = 0
	return label,  pred_prob


#----------------------------------------------------------------------------

# val_report, f1_checker, roc = gen_conf_and_cls_report(y_true_all, y_pred_all)
def gen_metric_report(ytrue: torch.Tensor, ypred: torch.Tensor):
	"""Generated report from the predcition such as classification-report, confusion-matrix, F1-score, ROC """

	ytrue = ytrue.contiguous().view(-1) # [..., 0:1]
	ypred = ypred.contiguous().view(-1)

	ypred_label, ypred_prob = compute_label_and_prob(ypred)

	report = {}
	report['clf_report'] = classification_report(y_true=ytrue.cpu().numpy(), y_pred=ypred_label.cpu().numpy())
	f1_checker = report['f1_score'] = f1_score(y_true=ytrue.cpu().numpy(), y_pred=ypred_label.cpu().numpy())
	report['conf_mat'] = confusion_matrix(y_true=ytrue.cpu().numpy(), y_pred=ypred_label.cpu().numpy())
	auc = report['auc'] = roc_auc_score(y_true=ytrue.cpu().numpy(), y_score=ypred_prob.cpu().numpy())
	
	# For plotting graph
	report['roc'] = roc_curve(y_true=ytrue.cpu().numpy(), y_score=ypred_prob.cpu().numpy())


	return report, f1_checker, auc


#----------------------------------------------------------------------------

def load_trained_model(model_fname, use_batchnorm):
	"""Loads the pretrained model for the given model name."""

	model_path = os.path.join(util.get_models_dir(), model_fname)
	model = {}

	if util.get_use_pretrained_flag():
		model = SiameseFaceNet(use_batchnorm=use_batchnorm)
	else:
		model = FaceNet(use_batchnorm=use_batchnorm) # # None)

	saved_state_dict = torch.load(model_path, map_location= lambda storage, loc: storage)

	# Available dict
	# raw_model_dict = model.state_dict()
	

	model.load_state_dict(saved_state_dict)
	model = model.eval()

	return model


#----------------------------------------------------------------------------

def extract_img_features(img_path, model_img_size, data_normalizer, features_extractor_model, type_tensor=True):
	"""Call feature extractor method to get image representation """

	features = futil.extract_features_with_nomalization(
							img_path=img_path, 
							model_size=model_img_size, 
							data_normalizer=data_normalizer, 
							features_extractor_model=features_extractor_model, 
							type_tensor=type_tensor
						)
	return features


#----------------------------------------------------------------------------

def compute_input_feature(img_path1: str, img_path2: str):
	"""Using pair of image computes a feature representaion on the them for example chi-distributed feature"""
	model_img_size = futil.get_model_img_size()
	data_normalizer = NormalizeImageData(means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]) # TODO: local
	feature_extractor = futil.get_pretrained_feature_extractor()

	img1_features = extract_img_features(img_path1, model_img_size, data_normalizer, feature_extractor, True)
	img2_features = extract_img_features(img_path2, model_img_size, data_normalizer, feature_extractor, True)

	return futil.compute_diff_vector(img1_features.reshape(-1), img2_features.reshape(-1))


#----------------------------------------------------------------------------
	
def load_image_data(img_path):
	model_img_size = futil.get_model_img_size()
	data_normalizer = NormalizeImageData(means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]) # TODO: local

	img = futil.load_and_resize_image(img_path, model_img_size)

	img_norml = data_normalizer(img) 
	img_norml_ch_first = np.transpose(img_norml, (2, 0, 1))

	return torch.tensor(img_norml_ch_first).unsqueeze(0)



#----------------------------------------------------------------------------

def predict_on_test(model, img_path1, img_path2, th=0.5):
	""" """
		
	with torch.no_grad():
		if util.get_use_pretrained_flag():
			x = load_image_data(img_path1), load_image_data(img_path2)
			y = model(x)
		else:
			x = compute_input_feature(img_path1, img_path2)
			y = model(x.reshape(1, -1).float())

	label, prob = compute_label_and_prob(y, th)

	return label, prob


#----------------------------------------------------------------------------

def get_arguments_parser(img1_datapath, img2_datapath):
	"""Argument parser for predition"""
	description = 	'Provide arguments for fullpath to pair of images to receive prob. of sameness score'
					
	parser = argparse.ArgumentParser(description=description)


	parser.add_argument('-i1', '--img1', type=str, default=img1_datapath, 
		help='Provide full path to two image locations.', required=True)

	parser.add_argument('-i2', '--img2', type=str, default=img2_datapath, 
		help='Provide full path to two image locations.', required=True)


	return parser


#----------------------------------------------------------------------------

def main():

	util.reset_logger('predictor_output.log')

	# First set the model_name and load 
	util.set_trained_model_name(ext_cmt='on_ext_features')
	base_model_fname = util.get_trained_model_name()


	use_batchnorm = False
	th = 0.5

	ex =  '_minval' # 
	model_fnames = [base_model_fname + ex for ex in ['_maxf1']] #  ['', '_maxauc', '_maxf1', '_minval', '_mintrain'] 

	img1_datapath = None # util.get_full_imgpath('Ann_Veneman', 5) 
	img2_datapath = None # util.get_full_imgpath('Ann_Veneman', 11)
	arg_parser = get_arguments_parser(img1_datapath, img2_datapath)
	arg_parser = arg_parser.parse_args()
	img1_datapath = arg_parser.img1 
	img2_datapath = arg_parser.img2 

	msg = '[Args]: \nimg1_datapath = {}, \nimg2_datapath = {}'.format(img1_datapath, img2_datapath)
	logger.info(msg)

	models = [load_trained_model(model_fname, use_batchnorm) for model_fname in model_fnames]
	output = {}
	for i, model in enumerate(models):
		label, prob = predict_on_test(model, img1_datapath, img2_datapath, th=0.5)
		prob = '{0:.3f}'.format(prob.item())
		
		output['threshold'] = th
		output['prob'] = prob
		if label == 0:
			output['class'] = 'Same'
			# print(f"\n[Th = {th}] Both the input images are **same** with [prob] = {prob}")
		else:
			output['class'] = 'Diff'
			# print(f"\n[Th = {th}] Images seem to be **different** with [prob] = {prob}")
	print('\n\n######################################################\n')
	print(output)
	print('\n######################################################\n\n')
	return output
		

#----------------------------------------------------------------------------

if __name__ == '__main__':
	print('[Run Test]')
	main()











