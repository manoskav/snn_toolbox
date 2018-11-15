import tensorflow as tf
import keras
import keras.backend as K


def myce(y_true, y_pred):

	# return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))#
	_epsilon = tf.convert_to_tensor(K.epsilon(), y_true.dtype.base_dtype)
	yt = (y_true + 1)/2
	yp = (y_pred + 1)/2
	yt = tf.clip_by_value(yt, _epsilon, 1 - _epsilon)
	yp = tf.clip_by_value(yp, _epsilon, 1 - _epsilon)

	ce = -K.mean(yt*K.log(yp) + (1-yt)*K.log(1-yp))
	return ce


def myaccuracy(y_true, y_pred):

	return K.mean(K.equal(y_true, K.sign(y_pred)))
