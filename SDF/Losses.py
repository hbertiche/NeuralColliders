import tensorflow as tf

def L1(y, p):
	return tf.reduce_mean(tf.abs(y - p))

def L2(y, p):
	return tf.reduce_sum((y - p) ** 2)