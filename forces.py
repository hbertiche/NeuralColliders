import os
import sys
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

# aux
# springs
pad_v0 = tf.constant([[1,0],[0,0],[0,0]])
pad_v1 = tf.constant([[0,1],[0,0],[0,0]])
pad_h0 = tf.constant([[0,0],[1,0],[0,0]])
pad_h1 = tf.constant([[0,0],[0,1],[0,0]])
# bend
pad_bv0 = tf.constant([[2,0],[0,0],[0,0]])
pad_bv1 = tf.constant([[0,2],[0,0],[0,0]])
pad_bh0 = tf.constant([[0,0],[2,0],[0,0]])
pad_bh1 = tf.constant([[0,0],[0,2],[0,0]])
# shear
pad_s0 = tf.constant([[1,0],[1,0],[0,0]])
pad_s1 = tf.constant([[0,1],[0,1],[0,0]])
pad_s2 = tf.constant([[0,1],[1,0],[0,0]])
pad_s3 = tf.constant([[1,0],[0,1],[0,0]])

@tf.function
def spring_forces(x, k_tension, size):
	# axis 0
	x0 = x[:-1]
	x1 = x[1:]
	d = x1 - x0
	r = tf.norm(d, axis=2, keepdims=True)
	f = k_tension * d * (r - size) / r
	f0 = -tf.pad(f, pad_v0)
	f1 = tf.pad(f, pad_v1)
	F = f0 + f1
	# axis 1
	x0 = x[:,:-1]
	x1 = x[:,1:]
	d = x1 - x0
	r = tf.norm(d, axis=2, keepdims=True)
	f = k_tension * d * (r - size)
	f0 = -tf.pad(f, pad_h0)
	f1 = tf.pad(f, pad_h1)
	F += f0 + f1
	return F

@tf.function
def bend_forces(x, k_bend, size):
	# axis 0
	x0 = x[:-2]
	x1 = x[2:]
	d = x1 - x0
	r = tf.norm(d, axis=2, keepdims=True)
	f = k_bend * d * (r - 2 * size) / r
	f0 = -tf.pad(f, pad_bv0)
	f1 = tf.pad(f, pad_bv1)
	F = f0 + f1
	# axis 1
	x0 = x[:,:-2]
	x1 = x[:,2:]
	d = x1 - x0
	r = tf.norm(d, axis=2, keepdims=True)
	f = k_bend * d * (r - 2 * size) / r
	f0 = -tf.pad(f, pad_bh0)
	f1 = tf.pad(f, pad_bh1)
	F += f0 + f1
	return F
	
@tf.function
def shear_forces(x, k_shear, size):
	# 'axis' 0
	x0 = x[:-1,:-1] # up left
	x1 = x[1:,1:] # low right
	d = x1 - x0
	r = tf.norm(d, axis=2, keepdims=True)
	f = k_shear * d * (r - 1.41421356237 * size) / r
	f0 = -tf.pad(f, pad_s0)
	f1 = tf.pad(f, pad_s1)
	F = f0 + f1
	# 'axis' 1
	x0 = x[1:,:-1] # low left
	x1 = x[:-1, 1:] # up right
	d = x1 - x0
	r = tf.norm(d, axis=2, keepdims=True)
	f = k_shear * d * (r - 1.41421356237 * size) / r
	f0 = -tf.pad(f, pad_s2)
	f1 = tf.pad(f, pad_s3)
	F += f0 + f1
	return F

@tf.function
def dampen_forces(v, k_dampen):
	# axis 0
	v0 = v[:-1]
	v1 = v[1:]
	f = (v1 - v0) * k_dampen
	f0 = -tf.pad(f, pad_v0)
	f1 = tf.pad(f, pad_v1)
	F = f0 + f1
	# axis 1
	v0 = v[:,:-1]
	v1 = v[:,1:]
	f = (v1 - v0) * k_dampen
	f0 = -tf.pad(f, pad_h0)
	f1 = tf.pad(f, pad_h1)
	F += f0 + f1
	# air drag
	F += -k_dampen * v
	return F

@tf.function
def self_collision_forces(x, self_collision_k, self_collision_thr):
	grid_size = x.shape[:2]
	# reshape to Nx3
	x = tf.reshape(x, (-1,3))
	# compute 'self' distances
	d = x[None] - x[:,None]
	r = tf.norm(d, axis=-1) + 1e6 * tf.eye(x.shape[0])
	# get correspondences and minimum distances
	i = tf.cast(tf.argmin(r, axis=1), tf.int32)
	i = tf.stack((tf.range(x.shape[0]), i), axis=-1)
	r = tf.gather_nd(r, i)[:,None]
	d = tf.gather_nd(d, i)
	# get collided vertices
	p = tf.cast(tf.math.less(r, self_collision_thr), tf.float32)
	f = -p * self_collision_k * d * (self_collision_thr - r) / (r + 1e-7)
	f = tf.reshape(f, (*grid_size, 3))
	return f
	
# this function is for self_collision_mode = 'tree'
@tf.function
def self_collision_forces2(x, r, idx, self_collision_k, self_collision_thr):
	grid_size = x.shape[:2]
	x = tf.reshape(x, (-1, 3))
	d = x - tf.gather(x, idx, axis=0)
	p = tf.cast(tf.math.less(r, self_collision_thr), tf.float32)
	f = p * self_collision_k * d * (self_collision_thr - r) / (r + 1e-7)
	f = tf.reshape(f, (*grid_size, 3))
	return f