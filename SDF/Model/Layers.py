import tensorflow as tf
import tensorflow_graphics as tfg

class FullyConnected:
	def __init__(self, shape, act=None, name='fc'):
		self.w = tf.Variable(tf.initializers.glorot_normal()(shape), name=name+'_w')
		self.b = tf.Variable(tf.zeros(shape[-1], dtype=tf.float32), name=name+'_b')
		self.act = act or (lambda x: x)

	def gather(self):
		return [self.w, self.b]

	def __call__(self, X):
		X = tf.einsum('ab,bc->ac', X, self.w) + self.b
		X = self.act(X)
		return X
		
class FourierFullyConnected:
	def __init__(self, shape, stddev=2**5, name='ff'):
		self.w = tf.Variable(
							tf.random_normal_initializer(stddev=stddev)(shape), 
							trainable=True, 
							name=name+'_w'
						)
		
	def gather(self):
		return [self.w] if self.w.trainable else []
	
	def __call__(self, X):
		X = tf.einsum('ab,bc->ac', X, self.w)
		return tf.concat((tf.math.cos(X), tf.math.sin(X)), -1)