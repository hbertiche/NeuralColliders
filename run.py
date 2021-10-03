import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg

from IO import readOBJ, writeOBJ, writePC2Frames
from time import time

from util import *
from forces import *
from collider import Collider

from scipy.spatial import cKDTree

"""
This is a TensorFlow implementation of cloth simulation to test neural SDF for collision handling.
"""

# parse args
# gpu_id : GPU/s to use
# name   : results will be stored in 'result/' folder under the chosen name
gpu_id, name = sys.argv[1], sys.argv[2]

# check to avoid overwritting previous simulation with same name
if os.path.isfile('results/' + name + '.pc2'):
	print("File already exists, delete previous to run or use a different name")
	sys.exit()
	
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# time-related parameters
dt = 1e-4					# simulation time step
T = 5						# total simulation time
FPS = 50.0					# output animation FPS
io_steps = int((1/FPS) / dt)# save 1 frame every 'io_steps' steps

# constants
G = np.array([0, 0, -9.81], np.float32) # gravity

# simulation params
steps = int(T / dt)	# number of steps to simulate
k_tension = 15.0	# tension springs strength
k_bend = 15.0		# bending springs strength
k_shear = 15.0		# shear springs strength
k_dampen = 1e-4		# dampen forces strength
# self
self_collision = False			# use self-collision?
self_collision_mode = 'tree'	# 'tree': cKDTree, 'brute': GPU-based all vs. all
assert self_collision_mode in {'tree', 'brute'}, "Wrong self-collision mode. Must be 'tree' or 'brute'"
self_collision_thr = .001		# distance for self-collision detection
self_collision_k = 15.0			# self-collision repel forces strength

# Cloth
density = 1.0						# density (kg/m2)
grid_size_x, grid_size_y = 200, 200	# n. verts in X and Y axis
size = 0.00125						# spring length
mass = size ** 2					# vertex mass
pos = 0.0, 0.015, .25				# starting location (center)
pin = np.ones(						# pin vertices. 0 = pinned, 1 = not pinned
	(grid_size_x, grid_size_y, 1), 
	np.float32
	)
# with this, two corners will be pinned
pin[0, 0, 0] = 0
pin[0, -1, 0] = 0

# make cloth
x = size * np.mgrid[:grid_size_x, :grid_size_y].transpose([1,2,0])
x -= x.mean(axis=(0,1))
x = np.concatenate((x, np.zeros((*x.shape[:2],1))), axis=-1)
x += np.array(pos)[None,None]
# Save as OBJ
writeOBJ('results/' + name + '.obj', x.reshape((-1,3)), grid_topology_to_faces(grid_size_x, grid_size_y))
# Init TF vars
x = tf.Variable(x, name='pos', dtype=tf.float32)
x_prev = tf.Variable(x, name='pos', dtype=tf.float32)
v = tf.Variable(np.zeros(x.shape), name='vel', dtype=tf.float32)

# Nueral Collider
collider_thr = .001
collider = Collider('bunny', # collider object 'bunny' or 'dragon'
	thr=collider_thr, 		  # collider margin
	loc=(.0, .0, .0), 	  # collider initial location
	v=(.0, .0, .0),			  # collider velocity
	rot=(.0, .0, .0),  # collider initial orientation
	w=(.0, .0, .0)			  # collider angular velocity
)

@tf.function
def compute_forces(x):
	F = []
	
	# Spring
	f = spring_forces(x, k_tension, size)
	F += [f]
	
	# Bend
	f = bend_forces(x, k_bend, size)
	F += [f]
	
	# Shear
	f = shear_forces(x, k_shear, size)
	F += [f]
	
	# Dampening
	f = dampen_forces(v, k_dampen)
	F += [f]
	
	# Sum forces
	F = tf.reduce_sum(F, axis=0)
	return F
	
@tf.function
def update(x, F):
	# Update
	a = F * (1.0 / mass) + G[None, None]
	dx = x - x_prev + a * dt**2
	dx *= pin
	v.assign((x + dx - x_prev) * (1.0 / (2.0 * dt)))
	x_prev.assign(x)
	x.assign(x + dx)	
	
@tf.function
def compute_collisions(x):
	collider.update(dt)
	d, N = collider.collide(x)
	p = tf.cast(tf.math.less(d, collider_thr), tf.float32)
	dx = p * (collider_thr - d) * N
	x.assign(x + dx)
	
@tf.function
def compute_ground_collisions(x):
	p = tf.minimum(0.0, x[:,:,2])[:,:,None]
	dx = tf.concat((tf.zeros((*x.shape[:2], 2), tf.float32), -p), axis=-1)
	x.assign(x + dx)

# for 'tree' self-collision mode
def compute_self_collisions(x):
	tree = cKDTree(x.reshape((-1, 3)))
	r, idx = tree.query(x.reshape((-1, 3)), k=2, n_jobs=-1) # query 2-NN, reject 1st (self)
	return r[:,1][:,None].astype(np.float32), idx[:,1]

# 'tf.function' decorator can be put here if self-collision is 'brute' or not activated
# @tf.function
def step():
	F = compute_forces(x)
	# Self-collision
	if self_collision:
		# TF
		if self_collision_mode == 'brute':
			f = self_collision_forces(x, self_collision_k, self_collision_thr)
		elif self_collision_mode == 'tree':
			r, idx = compute_self_collisions(x.numpy())
			f = self_collision_forces2(x, r, idx, self_collision_k, self_collision_thr)
		F += f
	update(x, F)
	
	# collisions
	compute_collisions(x)
	
	# ground collision
	compute_ground_collisions(x)

""" SIMULATION """
start = time()
for i in range(steps):
	step()		
	if i % io_steps == 0: 
		ellapsed = time() - start
		print((i + 1) // io_steps, '/', steps // io_steps)
		writePC2Frames('results/' + name + '.pc2', x.numpy().reshape((1, -1, 3)))
print('Time per step: ', (time() - start)/steps)
print('Time per frame: ', (time() - start)/(steps // io_steps))
print('Total time: ', time() - start)