# -*- coding:utf-8 -*-

""" ©Prem Prakash
	Unittest for different module
"""



import os
import pdb
import sys
import unittest

import util

import predictor

#----------------------------------------------------------------------------

class TestIneferrer(unittest.TestCase):
	"""docstring for TestUtils"""
	def __init__(self, *args, **kwargs):
		super(TestUtils, self).__init__(*args, **kwargs)
		
		
	
	def test_predictor(self):
		print('\n\n')
		dimg1 = util.get_full_imgpath('Al_Cardenas', 1)
		dimg2 = util.get_full_imgpath('Mary_Landrieu', 3)
		util.set_trained_model_name(ext_cmt='on_ext_features')
		base_model_fname = util.get_trained_model_name()
		use_batchnorm = False
		th = 0.5
		output = predictor.main(base_model_fname, dimg1, dimg2, use_batchnorm, th)
		print(output)