import tensorflow as tf
import numpy as np
from tensorflow import keras as K
class categorical_focal_loss:                             
    '''
    Softmax version of focal loss.

           m
      FL = sum  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    '''
    def __init__(self, gamma=0., alpha=.25):
        self._gamma = gamma
        self._alpha = alpha
        self.__name__ = 'categorical_focal_loss'
        
    def __int_shape(self, x):
        return tf.keras.backend.int_shape(x) if self.backend == 'tensorflow' else tf.keras.backend.shape(x)    
    def  __call__(self, y_true, y_pred):        
        '''
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        '''

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        print(y_pred)
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        # loss = tf.keras.backend.pow(1 - tf.math.softmax(y_pred), self._gamma) * tf.keras.backend.log(tf.math.softmax(y_pred))
        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        # loss = self._alpha * tf.keras.backend.pow(1 - y_pred, self._gamma) * tf.keras.backend.log(y_pred)
        loss = self._alpha * tf.keras.backend.pow(1 - y_pred, self._gamma) * cross_entropy
        # print(self._alpha)
        # Sum the losses in mini_batch
        # loss = -loss
        return tf.keras.backend.sum(loss, axis=-1)
