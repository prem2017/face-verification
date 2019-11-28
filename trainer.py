# -*- coding: utf-8 -*-


import os
import math
import pdb
import time
from datetime import datetime




import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR



from models import FaceNet
from img_dataset import ImageDataset
from predictor import gen_metric_report

import util
from util import logger
import fr_util as futil

device = util.get_training_device()

criterion = nn.BCEWithLogitsLoss() # Numericlly more stable



class Optimizer(object):
	"""Different optimizer of optimize learning process than vanilla greadient descent """
	def __init__(self):
		super(Optimizer, self).__init__()
		
		
	@staticmethod
	def rmsprop_optimizer(params, lr=1e-3, weight_decay=1e-6):
		return optim.RMSprop(params=params, lr=lr, alpha=0.99, eps=1e-6, centered=True, weight_decay=weight_decay)


	@staticmethod
	def adam_optimizer(params, lr=1e-3, weight_decay=1e-6):
		return optim.Adam(params=params, lr=lr, weight_decay=weight_decay)

	@staticmethod
	def sgd_optimizer(params, lr=1e-6, weight_decay=1e-6, momentum=0.9):
		return optim.SGD(params=params, lr=lr, weight_decay=weight_decay, momentum=momentum)




def train_network(dataloader, model, loss_function, optimizer, start_lr, end_lr, num_epochs=90, sanity_check=False):
	"""Trains the network and saves for different checkpoints such as minimum train/val loss, f1-score, AUC etc. different performance metrics

		Parameters:
		-----------
			dataloader (dict): {key (str):  Value(torch.utils.data.DataLoader)} training and validation dataloader to respective purposes
			model (nn.Module): models to traine the face-recognition
			loss_function (torch.nn.Module): Module to mesure loss between target and model-output
			optimizer (Optimizer): Non vanilla gradient descent method to optimize learning and descent direction
			start_lr (float): For one cycle training the start learning rate
			end_lr (float): the end learning must be greater than start learning rate
			num_epochs (int): number of epochs the one cycle is 
			sanity_check (bool): if the training is perfomed to check the sanity of the model. i.e. to anaswer 'is model is able to overfit for small amount of data?'

		Returns:
		--------
			None: perfoms the required task of training

	"""

	model = model.train()
	logger_msg = '\nDataLoader = %s' \
				 '\nModel = %s' \
				 '\nLossFucntion = %s' \
				 '\nOptimizer = %s' \
				 '\nStartLR = %s, EndLR = %s' \
				 '\nNumEpochs = %s' % (dataloader, model, loss_function, optimizer, start_lr, end_lr, num_epochs)

	logger.info(logger_msg), print(logger_msg)

	# [https://arxiv.org/abs/1803.09820]
	# This is used to find optimal learning-rate which can be used in one-cycle training policy
	# [LR]TODO: for finding optimal learning rate
	if util.get_search_lr_flag():
		lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=list(np.arange(2, 24, 2)), gamma=10, last_epoch=-1)
		

	def get_lr():
		lr = []

		for param_group in optimizer.param_groups:
			lr.append(np.round(param_group['lr'], 11))
		return lr

	def set_lr(lr):

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	# Loss storation
	current_epoch_batchwise_loss = []
	avg_epoch_loss_container = []  # Stores loss for each epoch averged over
	all_epoch_batchwise_loss = []
	avg_val_loss_container = []
	val_report_container = []
	f1_checker_container = []
	val_auc_container = []
	test_auc_container = {}
	test_f1_container = {}



	if util.get_search_lr_flag():
		extra_epochs = 4
	else:
		extra_epochs = 20
	total_epochs = num_epochs + extra_epochs

	# One cycle setting of Learning Rate
	num_steps_upndown = 10
	further_lowering_factor = 10
	further_lowering_factor_steps = 4

	def one_cycle_lr_setter(current_epoch):
		if current_epoch <= num_epochs:
			assert end_lr > start_lr, '[EndLR] should be greater than [StartLR]'
			lr_inc_rate = np.round((end_lr - start_lr) / (num_steps_upndown), 9)
			lr_inc_epoch_step_len = max(num_epochs / (2 * num_steps_upndown), 1)

			steps_completed = current_epoch / lr_inc_epoch_step_len
			print('[Steps Completed] = ', steps_completed)
			if steps_completed <= num_steps_upndown:
				current_lr = start_lr + (steps_completed * lr_inc_rate)
			else:
				current_lr = end_lr - ((steps_completed - num_steps_upndown) * lr_inc_rate)
			set_lr(current_lr)
		else:
			current_lr = start_lr / (
						further_lowering_factor ** ((current_epoch - num_epochs) // further_lowering_factor_steps))
			set_lr(current_lr)

	if sanity_check:
		train_dataloader = next(iter(dataloader['train']))
		train_dataloader = [train_dataloader] * 32
	else:
		train_dataloader = dataloader['train']

	for epoch in range(total_epochs):
		msg = '\n\n\n[Epoch] = %s' % (epoch + 1)
		print(msg)
		start_time = time.time()
		start_datetime = datetime.now()
		
		for i, (x, y) in enumerate(train_dataloader): # 
			loss = 0
			# pdb.set_trace()


			x = x.to(device=device, dtype=torch.float)
			y = y.to(device=device, dtype=torch.float)
			
			# TODO: early breaker
			# if i == 2:
			# 	print('[Break] by force for validation check')
			# 	break

			
			optimizer.zero_grad()
			output = model(x) # 
			loss = loss_function(output, y)
			loss.backward()
			optimizer.step()


			current_epoch_batchwise_loss.append(loss.item())
			all_epoch_batchwise_loss.append(loss.item())

			batch_run_msg = '\nEpoch: [%s/%s], Step: [%s/%s], InitialLR: %s, CurrentLR: %s, Loss: %s' \
							% (epoch + 1, total_epochs, i + 1, len(train_dataloader), start_lr, get_lr(), loss.item())
			print(batch_run_msg)
		#------------------ End of an Epoch ------------------ 
		
		# store average loss
		avg_epoch_loss = np.round(sum(current_epoch_batchwise_loss) / (i + 1.0), 6)
		current_epoch_batchwise_loss = []
		avg_epoch_loss_container.append(avg_epoch_loss)
		
		if not (util.get_search_lr_flag() or sanity_check):
			val_loss, val_report, f1_checker, auc = cal_loss_and_metric(model, dataloader['val'], loss_function, epoch+1)
			val_report['roc'] = 'Removed'
		test_test_data = False
		if not (util.get_search_lr_flag() or sanity_check):
			avg_val_loss_container.append(val_loss)
			val_report_container.append(val_report)  # ['epoch_' + str(epoch)] = val_report
			f1_checker_container.append(f1_checker)	
			val_auc_container.append(auc)

			if np.round(val_loss, 4) <= np.round(min(avg_val_loss_container), 4):
				model = save_model(model, extra_extension='_minval') # + '_epoch_' + str(epoch))

			if np.round(auc, 4) >= np.round(max(val_auc_container), 4):
				model = save_model(model, extra_extension='_maxauc') # + '_epoch_' + str(epoch))
				test_test_data = True

			if np.round(f1_checker, 4) >= np.round(max(f1_checker_container), 4):
				model = save_model(model, extra_extension='_maxf1') # + '_epoch_' + str(epoch))
				test_test_data = True


		if avg_epoch_loss <= min(avg_epoch_loss_container):
			model = save_model(model, extra_extension='_mintrain')


		
		# Logger msg
		msg = '\n\n\n\n\nEpoch: [%s/%s], InitialLR: %s, CurrentLR= %s \n' \
			  '\n\n[Train] Average Epoch-wise Loss = %s \n' \
			  '\n\n********************************************************** [Validation]' \
			  '\n\n[Validation] Average Epoch-wise loss = %s \n' \
			  '\n\n[Validation] Report () = %s \n'\
			  '\n\n[Validation] F-Report = %s\n'\
			  %(epoch+1, total_epochs, start_lr, get_lr(), avg_epoch_loss_container, avg_val_loss_container, None if not val_report_container else util.pretty(val_report_container[-1]), f1_checker_container)
		logger.info(msg); print(msg)

		if not (util.get_search_lr_flag() or sanity_check) or test_test_data:
			test_loss, test_report, test_f1_checker, test_auc = cal_loss_and_metric(model, dataloader['test'], loss_function, epoch+1, model_type='test_set')
			test_report['roc'] = 'Removed'
			test_auc_container[epoch+1] = "{0:.3f}".format(round(test_auc, 4)) 
			test_f1_container[epoch+1] = "{0:.3f}".format(round(test_f1_checker, 4))
			msg = '\n\n\n\n**********************************************************[Test]\n '\
				  '[Test] Report = {}' \
				  '\n\n[Test] fscore = {}' \
				  '\n\n[Test] AUC dict = {}' \
				  '\n\n[Test] F1-dict= {}'.format(util.pretty(test_report), test_f1_checker, test_auc_container, test_f1_container)
			logger.info(msg); print(msg)

		
		if avg_epoch_loss < 1e-6 or get_lr()[0] < 1e-11 or get_lr()[0] >= 10:
			msg = '\n\nAvg. Loss = {} or Current LR = {} thus stopping training'.format(avg_epoch_loss, get_lr())
			logger.info(msg)
			print(msg)
			break
			
		
		# [LR]TODO:
		if util.get_search_lr_flag():
			lr_scheduler.step(epoch+1) # TODO: Only for estimating good learning rate
		else:
			one_cycle_lr_setter(epoch + 1)

		end_time = time.time()
		end_datetime = datetime.now()
		msg = '\n\n[Time] taken for epoch({}) time = {}, datetime = {} \n\n'.format(epoch+1, end_time - start_time, end_datetime - start_datetime)
		logger.info(msg); print(msg)

	# ----------------- End of training process -----------------

	msg = '\n\n[Epoch Loss] = {}'.format(avg_epoch_loss_container)
	logger.info(msg); print(msg)

	
	# [LR]TODO: change for lr finder
	if util.get_search_lr_flag():
		losses = avg_epoch_loss_container
		plot_file_name = 'training_epoch_loss_for_lr_finder.png'
		title = 'Training Epoch Loss'
	else:
		losses = {'train': avg_epoch_loss_container, 'val': avg_val_loss_container}
		plot_file_name = 'training_vs_val_avg_epoch_loss.png'
		title= 'Training vs Validation Epoch Loss'
	plot_loss(losses=losses,
			plot_file_name=plot_file_name,
			title=title)
	plot_loss(losses=all_epoch_batchwise_loss, plot_file_name='training_batchwise.png', title='Training Batchwise Loss',
			xlabel='#Batchwise')
		

	# Save the model		
	model = save_model(model)




def cal_loss_and_metric(model: torch.nn.Module, 
						dataloader: torch.utils.data.DataLoader, 
						loss_func: torch.nn.Module, 
						epoch_iter=0, 
						model_type='val_set'):
	"""Computes loss on val/test data and return a prepared metric report on that"""
	
	model = model.eval()
	
	loss = 0
	y_pred_all = None
	y_true_all = None
	with torch.no_grad():
		for i, (x, y) in enumerate(dataloader): # last one is true image size
			# print('[Val/Test] i = ', i)
			# loss = 0
			
			x = x.to(device=device, dtype=torch.float)
			y = y.to(device=device, dtype=torch.float)

			y_pred = model(x)
			loss = loss_func(y_pred, y)

			loss += loss.item()

			y_pred_all = y_pred if y_pred_all is None else torch.cat((y_pred_all, y_pred), dim=0)
			y_true_all = y if y_true_all is None else torch.cat((y_true_all, y), dim=0)

	report, f1_checker, auc = gen_metric_report(y_true_all, y_pred_all)
	model = model.train()
	loss =  np.round(loss / (i + 1.0), 6)
	
	plot_roc_curves_binclass(report, epoch_iter, model_type=model_type)
	
	return loss.item(), report, f1_checker, auc  # loss, score, model




# Plot training loss
def plot_loss(losses, plot_file_name='training_loss.png', title='Training Loss', xlabel='Epochs'):
	fig = plt.figure()
	label_key = {'train': 'Training Loss', 'val': 'Validation Loss'}
	if isinstance(losses, dict):
		for k, v in losses.items():
			plt.plot(range(1, len(v)), v[1:], '-*', markersize=3, lw=1, alpha=0.6, label=label_key[k])	
	else:
		plt.plot(range(1, len(losses)+1), losses, '-*', markersize=3, lw=1, alpha=0.6)
	
	
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('BCE Loss')
	plt.legend(loc='upper right')
	full_path = os.path.join(util.get_results_dir(), plot_file_name)
	fig.tight_layout()  # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
	fig.savefig(full_path)
	plt.close(fig)  # clo


def plot_roc_curves_binclass(roc_data, epoch_iter, model_type='val_set'):
	"""Plots ROC curve after each iteration """
	fpr, tpr, ths = roc_data['roc']

	tpr_fpr = tpr - fpr
	optimal_idx = np.argmax(tpr_fpr)
	max_tpr_fpr = tpr_fpr[optimal_idx]

	optimal_fpr, optimal_tpr, optimal_threshold = fpr[optimal_idx], tpr[optimal_idx], ths[optimal_idx]

	msg = '\n\n[%s] on ROC best (tpr-fpr) = %f, optimal-threshold = %f\n\n' %(model_type, max_tpr_fpr, optimal_threshold)
	logger.info(msg); print(msg)
	auc_val = roc_data['auc']


	fig = plt.figure()
	plt.plot(fpr, tpr, lw=2, color='red', label='ROC Curve (area = %0.2f)\nMax(tpr-fpr) = %.2f\nThs=%.9f' %(auc_val, max_tpr_fpr, optimal_threshold))
	plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
	plt.plot(optimal_fpr, optimal_tpr, color='green', marker='*', markersize=9)

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])

	plt.xlabel('False Positive Rate (1-Specificity)')
	plt.ylabel('True Positive Rate (Recall)')
	plt.title('ROC Curve')
	plt.legend(loc='lower right')

	base_path = util.get_results_dir(model_type)
	full_path = os.path.join(base_path, str(epoch_iter) + '_' + model_type + '.png')
	fig.tight_layout()
	plt.savefig(full_path)
	plt.close(fig)


def save_model(model, extra_extension=""):
	msg = '[Save] model extra_extension = {}'.format(extra_extension)
	logger.info(msg); print(msg)

	model_path = os.path.join(util.get_models_dir(), util.get_trained_model_name()) + extra_extension
	if next(model.parameters()).is_cuda:
		model = model.cpu().float()

	model_dict = model.state_dict()
	torch.save(model_dict, model_path)
	
	models_dict = model.to(device)
	return models_dict






# Pre-requisite setup for training process
def train_model(train_data_info, val_data_info, test_data_info, sanity_check=False):
	"""Setup all the pre-requisites for complete training of the model 

		Parameters:
		-----------
			train_data_info (dict): Arguments needed to setup train dataset such datapath etc.
			val_data_info (dict): Arguments needed to setup val dataset such datapath etc.
			test_data_info (dict): Arguments needed to check performance on test set
			sanity_check (bool): pass the boolean to the method <train_network> to indicate if it is sanity check or full training

		Returns:
		--------
			None: Only works as setup for the training of the model
	"""
	msg = '\n\n[Train] data info = {}\n\n[Validation] data info = {}\n\n[SanityCheck] = {}'.format(train_data_info, val_data_info, sanity_check)
	logger.info(msg), print(msg)
	
	train_params = {}
	# [LR]
	if util.get_search_lr_flag() :
		start_lr, end_lr, epochs = 1e-6, 10, 20 # 
	else:
		start_lr, end_lr, epochs = 5e-5, 2e-4, 70 # 5e-5, 2e-4, 70   3e-3, 6e-3, 70  2e-3, 9e-3, 70 # 1e-3, 5e-3, 50 # 7e-3, 11e-3, 70
	train_params['start_lr'] = start_lr = start_lr
	train_params['end_lr'] = end_lr
	train_params['num_epochs'] = epochs


	use_batchnorm = False # TODO: batchnorm
	dropout = 0.0
	if sanity_check or util.get_search_lr_flag():
		weight_decay = 0
		dropout = 0
	else:
		weight_decay = 1e-3# 1e-6
		dropout = 0.4 # 0.5 # might not be needed
	


	dataset = {}
	dataset['train'] = ImageDataset(**train_data_info)
	dataset['val'] = ImageDataset(**val_data_info)
	dataset['test'] = ImageDataset(**test_data_info)


	dataloader = {}
	# pdb.set_trace()
	dataloader['train'] = DataLoader(dataset=dataset['train'], batch_size=util.get_train_batch_size(), shuffle=True)
	dataloader['val'] = DataLoader(dataset=dataset['val'], batch_size=util.get_val_batch_size())
	dataloader['test'] = DataLoader(dataset=dataset['test'], batch_size=util.get_test_batch_size())

	train_params['dataloader'] = dataloader

	net_args = {}
	net_args['in_features'] = 512
	net_args['out_features'] = 1
	net_args['nonlinearity_function'] = None
	net_args['dropout'] = dropout
	net_args['use_batchnorm'] = use_batchnorm
	model =  FaceNet(**net_args)

	train_params['model'] = model = model.to(device)

	loss_function = criterion
	train_params['loss_function'] = loss_function.to(device)

	optimizer = Optimizer.sgd_optimizer(params=model.parameters(), lr=start_lr, weight_decay=weight_decay, momentum=0.9)
	# optimizer = Optimizer.adam_optimizer(params=model.parameters(), lr=start_lr, weight_decay=weight_decay)

	train_params['optimizer'] = optimizer
	train_params['sanity_check'] = sanity_check
	

	# Train the network
	train_network(**train_params)



def main(sanity_check=False):

	# Sanity check of the model for learning
	# TODO: put the sanity check in util or make some setup file
	sanity_check = sanity_check
	util.set_trained_model_name(ext_cmt='on_ext_features')


	train_data_info = {'feature_path': util.get_train_feature_path(), 'pairs_imgnames_relpath_dict': util.get_train_pairs_imgnames_relpath_dict()}
	val_data_info = {'feature_path': util.get_val_feature_path(), 'pairs_imgnames_relpath_dict': util.get_val_pairs_imgnames_relpath_dict()}
	test_data_info = {'feature_path': util.get_test_feature_path(), 'pairs_imgnames_relpath_dict': util.get_test_pairs_imgnames_relpath_dict()}



	msg = '[Datapath] \nTrain = {}, \nValidation = {}'.format(train_data_info, val_data_info)
	logger.info(msg); print(msg)
	train_model(train_data_info=train_data_info, val_data_info=val_data_info, test_data_info=test_data_info, sanity_check=sanity_check)
	



if __name__ == '__main__':
	print('Trainer')
	torch.manual_seed(999)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(999)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	util.reset_logger()
	print('Pid = ', os.getpid())

	main(sanity_check=False)

	msg = '\n\n********************** Training Complete **********************\n\n'
	logger.info(msg); print(msg)

	