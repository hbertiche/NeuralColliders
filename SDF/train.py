import os
import sys
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from time import time
from datetime import timedelta
from math import floor, ceil

from Data.data import Data
from Model.SDF import SDF

from util import model_summary
from Losses import *

@tf.function
def train_step(points, y):
	""" Train step """
	with tf.GradientTape() as tape:
		pred = model(
					points
				)
		# Losses & Metrics
		loss = L1(y, pred)
		err = L1(y, pred)
	""" Backprop """
	grads = tape.gradient(loss, tgts)
	optimizer.apply_gradients(zip(grads, tgts))
	return loss, err
	
@tf.function
def test_step(points, y):
	pred = model(points)
	err = L1(y, pred)
	return err

""" ARGS """
# gpu_id: GPU slot to run model
# name: name under which model checkpoints will be saved
# checkpoint: pre-trained model (must be in ./checkpoints/ folder)
gpu_id = sys.argv[1] # mandatory
name = sys.argv[2]   # mandatory
checkpoint = None
if len(sys.argv) > 3:
	checkpoint = 'checkpoints/' + sys.argv[3]

""" GPU """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" Log """
stdout_steps = 100
if name == 'test': stdout_steps = 1

""" TRAIN PARAMS """
tr_steps = 1000
val_steps = 100
num_epochs = 1000 # if fine-tuning, fewer epochs
substeps = None # usually None

""" PARAMS """
n_points = 5000 # points to sample per sample
thr = .01 # sample query points within a 'thr' margin wrt. object surface

""" MODEL """
print("Building model...")
model = SDF(checkpoint)
if checkpoint is not None and checkpoint.split('/')[1] != name: model._best = float('inf')
tgts = model.gather() # model weights
tgts = [t for t in tgts if t.trainable]
model_summary(tgts)
print("*"*25)
print("Model Best: ", model._best)
print("*"*25)
optimizer = tf.optimizers.Adam()

""" DATA """
print("Reading data...")
npy = 'Objects/stanford_dragon.npy'
obj = 'Objects/stanford_dragon.obj'
tr_data = Data(npy, steps=tr_steps, n_points=n_points, thr=thr)
te_data = Data(obj, steps=val_steps, n_points=n_points, thr=0)

tr_steps = floor(tr_data._n_samples / n_points)
for epoch in range(num_epochs):
	print("")
	print("Epoch " + str(epoch + 1))
	print("--------------------------")
	""" TRAIN """
	print("Training...")
	step = 0
	total_time = 0
	metrics = [0] * 2 # loss, err
	start = time()
	for points, y in tr_data._iterator:
		points = tf.squeeze(points)
		y = tf.squeeze(y)
		""" Train step """
		loss, err = train_step(points, y)
		""" Progress """
		metrics[0] += loss.numpy() 
		metrics[1] += 1000 * err.numpy()
		total_time = time() - start
		ETA = (tr_steps - step - 1) * (total_time / (1+step))
		if (step + 1) % stdout_steps == 0:
			sys.stdout.write('\r\tStep: ' + str(step+1) + '/' + str(tr_steps) + ' ... '
					+ 'Loss: {:.5f}'.format(metrics[0] / (1+step)) 
					+ ' - '
					+ 'Err: {:.2f}'.format(metrics[1] / (1+step))
					+ ' ... ETA: ' + str(timedelta(seconds=ETA)))
			sys.stdout.flush()
		step += 1
		if substeps is not None and substeps == step: break
	""" Epoch results """
	metrics = [m / (step + 1) for m in metrics]
	print("")
	print("Total loss: {:.5f}".format(metrics[0]))
	print("Total err: {:.2f}".format(metrics[1]))
	print("Total time: " + str(timedelta(seconds=total_time)))
	print("")

	""" TEST SURFACE """
	print("Testing...")
	""" Forward pass """
	err = test_step(te_data.V, 0)
	""" Epoch results """
	print("")
	print("Total err: {:.5f}".format(err))
	print("")
	""" Save checkpoint """
	if err.numpy() < model._best: 
		model._best = err.numpy()
		""" Save checkpoint """
		model.save('checkpoints/' + name)
	print("")
	print("BEST: ", model._best)
	print("")
