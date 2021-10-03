import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Layers import *

class SDF:
	def __init__(self, checkpoint=None):
		self._build()
		self._best = float('inf')
		# load pre-trained
		if checkpoint is not None:
			print("Loading pre-trained model: " + checkpoint)
			self.load(checkpoint)
	
	def _build(self):
		self._mlp = [
			FourierFullyConnected((3, 128), name='ff0'),
			FullyConnected((256, 256), act=tf.nn.relu, name='fc0'),
			FullyConnected((256, 256), act=tf.nn.relu, name='fc1'),
			FullyConnected((256, 256), act=tf.nn.relu, name='fc2'),
			FullyConnected((256, 256), act=tf.nn.relu, name='fc3'),
			FullyConnected((256, 256), act=tf.nn.relu, name='fc4'),
			FullyConnected((256, 256), act=tf.nn.relu, name='fc5'),
			FullyConnected((256, 256), act=tf.nn.relu, name='fc6'),
			FullyConnected((256, 256), act=tf.nn.relu, name='fc7'),
			FullyConnected((256, 1), name='fc8')
		]
		self._layers = self._mlp

	def gather(self):
		vars = []
		for l in self._layers:
			vars += l.gather()
		return vars
	
	def load(self, checkpoint):
		# checkpoint: path to pre-trained model
		# list vars
		vars = self.gather()
		# load vars values
		if not checkpoint.endswith('.npy'): checkpoint += '.npy'
		values = np.load(checkpoint, allow_pickle=True)[()]
		# assign
		_vars = set([v.name for v in vars])
		_vars_chck = set(values.keys()) - {'best'}
		# vars in checkpoint but not in model
		_diff = sorted(list(_vars_chck - _vars))
		if len(_diff):
			print("Model missing vars:")
			for v in _diff: print("\t" + v)
		# vars in model but not in checkpoint
		_diff = sorted(list(_vars - _vars_chck))
		if len(_diff):
			print("Checkpoint missing vars:")
			for v in _diff: print("\t" + v)
		# assign values to vars
		for v in vars: 
			try: v.assign(values[v.name])
			except:
				if v.name not in values: continue
				else: 
					print("Mismatch in variable shape:")
					print("\t" + v.name)
		if 'best' in values: self._best = values['best']
	
	def save(self, checkpoint):
		# checkpoint: path to pre-trained model
		print("\tSaving checkpoint: " + checkpoint)
		# get vars values
		values = {v.name: v.numpy() for v in self.gather()}
		if self._best is not 0: values['best'] = self._best
		# save weights
		if not checkpoint.endswith('.npy'): checkpoint += '.npy'
		np.save(checkpoint, values)
		
	def __call__(self, P):
		X = P
		for l in self._mlp:
			X = l(X)
		return tf.squeeze(X)