# -*- coding: utf-8 -*-

# ©Prem Prakash
# Different models


import pdb
import os

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import fr_util as futil 
import util


#----------------------------------------------------------------------------

device = util.get_training_device()



#----------------------------------------------------------------------------

class PretrainedInceptiongResnet(nn.Module):
	"""docstring for PretrainedInceptiongResnet"""

	@staticmethod
	def get_pretrained_facenet_resenet():
		resnet = InceptionResnetV1(pretrained='vggface2').train()

		# https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
		# newmodel = torch.nn.Sequential(*(list(model.children())[:-6])) # one block less i.e. removed self.block8 = Block8(noReLU=True) 
		resnet_trimmed = nn.Sequential(*(list(resnet.children())[:-3]))  # output is [b, 1792, 1, 1]
		output_dim = 1792 # can change if layers are reduced

		## Set a layer as [non-trainable](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html):	
		for param in resnet_trimmed.parameters():
			param.requires_grad = False

		return resnet_trimmed, 1792


	def __init__(self, needed_output_dim=512, nonlinearity_function=None, dropout=0.0,  use_batchnorm=False, eps=1e-6):
		super(PretrainedInceptiongResnet, self).__init__()
		self.nonlinearity_function =  nonlinearity_function if nonlinearity_function is not None else nn.ReLU()
		
		self.pretrained_model, self.pretrained_output_dim = PretrainedInceptiongResnet.get_pretrained_facenet_resenet()
		
		self.needed_output_dim = needed_output_dim

		self.needed_output_fc = nn.Linear(self.pretrained_output_dim, self.needed_output_dim)



	def forward(self, X):
		X = self.pretrained_model(X)
		X = X.view(X.shape[0], -1)
		# Batchnorm
		X = self.nonlinearity_function(X)
		out = self.needed_output_fc(X)
		# Batchnorm
		X = self.nonlinearity_function(X)
		# dropout

		return out


#----------------------------------------------------------------------------

class SiameseFaceNet(nn.Module):
	"""docstring for SiameseFaceNet"""

	def __init__(self, in_features=512, out_features=1, nonlinearity_function=None, dropout=0.0, use_batchnorm=False, eps=1e-6):
		super(SiameseFaceNet, self).__init__()
		self.nonlinearity_function =  nonlinearity_function if nonlinearity_function is not None else nn.ReLU()
		self.eps = eps

		self.pretrained_model = PretrainedInceptiongResnet(needed_output_dim=in_features, nonlinearity_function=self.nonlinearity_function, dropout=dropout, use_batchnorm=use_batchnorm, eps=eps)

		hidden_features = 1024
		if use_batchnorm:
			self.fc_layers = nn.Sequential(
					nn.Linear(in_features=in_features, out_features=hidden_features),
					nn.BatchNorm1d(hidden_features, eps=self.eps),
					self.nonlinearity_function,
					nn.Dropout(dropout, inplace=True),

					nn.Linear(in_features=in_features, out_features=hidden_features // 2),
					nn.BatchNorm1d(hidden_features // 2, eps=self.eps),
					self.nonlinearity_function,
					nn.Dropout(dropout, inplace=True),

					nn.Linear(in_features=hidden_features // 2, out_features=out_features))
		else:
			self.fc_layers = nn.Sequential(
					nn.Linear(in_features=in_features, out_features=hidden_features),
					self.nonlinearity_function,
					nn.Dropout(dropout, inplace=True),

					nn.Linear(in_features=hidden_features, out_features=hidden_features // 2),
					self.nonlinearity_function,
					nn.Dropout(dropout, inplace=True),

					nn.Linear(in_features=hidden_features // 2, out_features=out_features))			


	def forward(self, X):
		X1, X2 = [temp_X.to(device=device, dtype=torch.float32) for temp_X in X]

		X1 = self.pretrained_model(X1)
		X2 = self.pretrained_model(X2)

		X = futil.compute_diff_vector(X1, X2)

		out = self.fc_layers(X)

		return out


#----------------------------------------------------------------------------

class FaceNet(nn.Module):
	"""docstring for FaceNet"""


	def __init__(self, in_features=512, out_features=1, nonlinearity_function=None, dropout=0.0, use_batchnorm=False):
		super(FaceNet, self).__init__()
		self.nonlinearity_function =  nonlinearity_function if nonlinearity_function is not None else nn.ReLU()
		

		hidden_features = 1024
		if use_batchnorm:
			self.fc_layers = nn.Sequential(
					nn.Linear(in_features=in_features, out_features=hidden_features),
					nn.BatchNorm1d(hidden_features, eps=1e-6),
					self.nonlinearity_function,
					nn.Dropout(dropout, inplace=True),

					nn.Linear(in_features=hidden_features, out_features=hidden_features // 2),
					nn.BatchNorm1d(hidden_features // 2, eps=1e-6),
					self.nonlinearity_function,
					nn.Dropout(dropout, inplace=True),


					nn.Linear(in_features=hidden_features // 2, out_features=(hidden_features // 2**2)),
					nn.BatchNorm1d((hidden_features // 2**2), eps=1e-6),
					self.nonlinearity_function,
					nn.Dropout(dropout, inplace=True),

					nn.Linear(in_features=(hidden_features // 2**2), out_features=out_features)) 
		else:
			self.fc_layers = nn.Sequential(
				nn.Linear(in_features=in_features, out_features=hidden_features),
				self.nonlinearity_function,
				nn.Dropout(dropout, inplace=True),

				nn.Linear(in_features=hidden_features , out_features=hidden_features // 2),
				self.nonlinearity_function,
				nn.Dropout(dropout, inplace=True),

				nn.Linear(in_features=hidden_features // 2, out_features=hidden_features // 2**2),
				self.nonlinearity_function,
				nn.Dropout(dropout, inplace=True),


				nn.Linear(in_features=hidden_features // 2 **2, out_features=out_features)) 

	
	def forward(self, X):
		# pdb.set_trace()
		out = self.fc_layers(X)
		return out


		

















