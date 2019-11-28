# -*- coding: utf-8 -*-


import pdb
import os

import torch
import torch.nn as nn


class FaceNet(nn.Module):
	"""docstring for FaceNet"""


	def __init__(self, in_features=512, out_features=1, nonlinearity_function=None, dropout=0.0, use_batchnorm=False):
		super(FaceNet, self).__init__()
		self.nonlinearity_function =  nonlinearity_function if nonlinearity_function is not None else nn.ReLU()
		
		input_dimension = in_features
		output_dimension = out_features
		hidden_dimensions = 1024
		if use_batchnorm:
			self.fc_layers = nn.Sequential(
					nn.Linear(in_features=input_dimension, out_features=hidden_dimensions),
					nn.BatchNorm1d(hidden_dimensions, eps=1e-6),
					self.nonlinearity_function,
					nn.Dropout(dropout, inplace=True),

					nn.Linear(in_features=hidden_dimensions, out_features=hidden_dimensions // 2),
					nn.BatchNorm1d(hidden_dimensions // 2, eps=1e-6),
					self.nonlinearity_function,
					nn.Dropout(dropout, inplace=True),


					nn.Linear(in_features=hidden_dimensions // 2, out_features=(hidden_dimensions // 2**2)),
					nn.BatchNorm1d((hidden_dimensions // 2**2), eps=1e-6),
					self.nonlinearity_function,
					nn.Dropout(dropout, inplace=True),

					nn.Linear(in_features=(hidden_dimensions // 2**2), out_features=(hidden_dimensions // 2**3)),
					nn.BatchNorm1d((hidden_dimensions // 2**3), eps=1e-6),
					self.nonlinearity_function,
					nn.Dropout(dropout, inplace=True),

					nn.Linear(in_features=(hidden_dimensions // 2**3), out_features=output_dimension)
				) 
		else:
			self.fc_layers = nn.Sequential(
				nn.Linear(in_features=input_dimension, out_features=hidden_dimensions),
				self.nonlinearity_function,
				nn.Dropout(dropout, inplace=True),

				nn.Linear(in_features=hidden_dimensions , out_features=hidden_dimensions // 2),
				self.nonlinearity_function,
				nn.Dropout(dropout, inplace=True),

				nn.Linear(in_features=hidden_dimensions // 2, out_features=hidden_dimensions // 2**2),
				self.nonlinearity_function,
				nn.Dropout(dropout, inplace=True),


				nn.Linear(in_features=hidden_dimensions // 2 **2, out_features=output_dimension),

			) 


	def forward(self, X):
		# pdb.set_trace()
		out = self.fc_layers(X)
		return out



