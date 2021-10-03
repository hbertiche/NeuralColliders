import sys
import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg

sys.path.append('SDF/Model')
# Bounding Boxes
bunnyBB = np.float32([
				[-0.06961248, 0.051060814],
				[-0.08818923, 0.06750977],
				[-0.00036688228, 0.15396672]
			])
dragonBB = np.float32([
				[-.108324, .0965662],
				[-.0412075, .050414],
				[.0, .14446905]
			])
				
class Collider:
	def __init__(self, object, thr=.001, loc=(.0,.0,.0), v=(.0, .0, .0), rot=(.0, .0, .0), w=(.0, .0, .0)):
		# this implementation supports 2 objects only, but can be easily scaled
		assert object in {'bunny', 'dragon'}, 'Wrong object: ' + str(object)
		if object == 'bunny':
			from SDFBunny import SDF
			self.sdf = SDF('SDF/checkpoints/bunny.npy')
			self.BB = bunnyBB
		elif object == 'dragon':
			from SDFDragon import SDF
			self.sdf = SDF('SDF/checkpoints/dragon.npy')
			self.BB = dragonBB
		if self.BB is not None:
			self.BB[:,0] -= thr
			self.BB[:,1] += thr
		self.loc = tf.Variable(loc, dtype=tf.float32)
		self.v = tf.constant(v, tf.float32)
		self.rot = tf.Variable(rot, dtype=tf.float32)
		self.w = tf.constant(w, tf.float32)
	
	def update(self, dt):
		self.loc.assign(self.loc + dt * self.v)
		self.rot.assign(self.rot + dt * self.w)

	def compute_rot(self):
		x, y, z = self.rot[0], self.rot[1], self.rot[2]
		c_x, s_x = tf.math.cos(x), tf.math.sin(x)
		c_y, s_y = tf.math.cos(y), tf.math.sin(y)
		c_z, s_z = tf.math.cos(z), tf.math.sin(z)
		Rx = tf.stack([[1, 0, 0],
					   [0, c_x, -s_x],
					   [0, s_x, c_x]], axis=0)
		Ry = tf.stack([[c_y, 0, s_y],
					   [0, 1, 0],
					   [-s_y, 0, c_y]], axis=0)
		Rz = tf.stack([[c_z, -s_z, 0],
					   [s_z, c_z, 0],
					   [0, 0, 1]], axis=0)
		R = tf.linalg.matmul(Rz, tf.linalg.matmul(Ry, Rx))
		return R, tf.linalg.inv(R)
			
	def check_BB(self, x):
		if self.BB is None: return tf.constant(True, tf.bool)
		bb = self.BB
		p_bb = tf.math.greater(x[:,:,0], bb[0,0])
		p_bb = tf.math.logical_and(p_bb, tf.math.less(x[:,:,0], bb[0, 1]))
		p_bb = tf.math.logical_and(p_bb, tf.math.greater(x[:,:,1], bb[1, 0]))
		p_bb = tf.math.logical_and(p_bb, tf.math.less(x[:,:,1], bb[1, 1]))
		p_bb = tf.math.logical_and(p_bb, tf.math.greater(x[:,:,2], bb[2, 0]))
		p_bb = tf.math.logical_and(p_bb, tf.math.less(x[:,:,2], bb[2, 1]))
		return tf.math.reduce_any(p_bb)
	
	def collide_sdf(self, X):
		with tf.GradientTape() as tape:
			tape.watch(X)
			d = self.sdf(tf.reshape(X, (-1, 3)))
			d = tf.reshape(d, (*X.shape[:2], 1))
		grads = tape.gradient(d, X)
		grads /= tf.norm(grads + 1e-9, axis=-1, keepdims=True)
		return d, grads
		
	def no_collide(self, X):
		return 1e3 * tf.ones((*X.shape[:2], 1), tf.float32), tf.zeros(X.shape, tf.float32)
	
	def collide(self, x):
		X = x - self.loc[None,None]
		R, R_inv = self.compute_rot()
		X = tf.einsum('ab,cdb->cda', R_inv, X)
		p = self.check_BB(X)
		true_fn = lambda: self.collide_sdf(X)
		false_fn = lambda: self.no_collide(X)
		d, N = tf.cond(p, true_fn, false_fn)
		N = tf.einsum('ab,cdb->cda', R, N)
		return d, N