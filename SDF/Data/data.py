import os
import sys
from random import shuffle, choice
from math import floor
from scipy import sparse

from time import time
import tensorflow as tf
import tensorflow_graphics as tfg
import pymesh

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from util import *
from IO import *

class Data:	
	def __init__(self, object, steps, mode='train', n_points=5000, thr=.01):
		"""
		Args:
		- poses: path to .npy file with poses
		- shape: SMPL shape parameters for the subject
		- gender: 0 = female, 1 = male
		- batch_size: batch size
		- shuffle: shuffle
		"""
		self._mode = mode # 'train' or 'test'
		self._n_samples = steps * n_points
		self._n_points = n_points
		self._thr = thr
		# compute points on the fly
		if object.endswith('.obj'):
			# Read OBJ
			self.V, self.F = readOBJ(object)[:2]
			self.F = quads2tris(self.F)
			self.mesh = pymesh.form_mesh(self.V, self.F)
			self.mesh.add_attribute('face_area')
			self.face_area = self.mesh.get_attribute('face_area').copy()
			self.face_area /= np.sum(self.face_area, axis=-1)
			# TF Dataset
			ds = tf.data.Dataset.from_tensor_slices(list(range(steps)))
			ds = ds.map(self.tf_map, num_parallel_calls=1)
			ds = ds.batch(batch_size=1)
			self._iterator = ds
		# pre-computed points
		elif object.endswith('.npy'):
			# Read data points (x, y, z, d)
			D = np.load(object)
			self._n_samples = D.shape[0]
			# TF Dataset
			ds = tf.data.Dataset.from_tensor_slices((D[:,:3], D[:,3]))
			ds = ds.batch(batch_size=n_points)
			if mode == 'train':
				ds = ds.shuffle(n_points)
			self._iterator = ds
		
	def _next(self, pose):
		""" Generate query points """
		N0 = int(4 * self._n_points / 5)
		N1 = self._n_points - N0
		P = self._sample_points(N0) + np.random.uniform(-self._thr, self._thr, size=(N0, 3))
		_P = np.random.uniform(-1, 1, size=(N1, 3))
		_P[:,0] *= (1.0 / 5.0)
		_P[:,1] *= (1.0 / 10.0)
		_P[:,2] *= (1.0 / 6.66666)
		_P[:,2] += 0.05
		P = np.concatenate((P, _P), axis=0)
		y = self._get_dist(P)
		return P, y
	
	def _sample_points(self, N=None):
		if N is None: N = self._n_points
		# choose indices
		idx = np.random.choice(self.F.shape[0], size=(N,), p=self.face_area)
		# Barycentric sampling
		W = np.random.uniform(size=(N, 3))
		W /= np.sum(W, axis=-1, keepdims=True)
		# Compute points
		triangles = self.V[self.F[idx]]
		return np.einsum('ijk,ij->ik', triangles, W)

	def _get_dist(self, P):
		d, f, p = pymesh.distance_to_mesh(self.mesh, P)
		d = np.sqrt(d)
		sign = np.sign(-2 * pymesh.compute_winding_number(self.mesh, P) + 1)
		return sign * d
		
	def tf_map(self, pose):
		return tf.py_function(func=self._next, inp=[pose], Tout=[tf.float32, tf.float32])

""" Pre-compute data points """
if __name__ == '__main__':
	object = '../Objects/stanford_dragon.obj'
	N = 5000000 # total number of points to pre-compute
	n_points = 100000 # compute per batches
	steps = int(N / n_points)
	data = Data(object, steps, n_points=n_points, thr=.005)
	# store data points as N x 4 (xyzd)
	D = np.zeros((N, 4), np.float32)
	i = 0
	for P, d in data._iterator:
		print(i, '/', steps)
		P, d = P[0].numpy(), d[0].numpy()
		s = i * n_points
		e = (i + 1) * n_points
		D[s:e, :3] = P
		D[s:e, 3] = d
		i += 1
	np.save('../Objects/stanford_dragon.npy', D)