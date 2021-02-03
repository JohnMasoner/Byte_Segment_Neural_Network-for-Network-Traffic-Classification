# import time
import sys
import os
# import keras
import input_data
import numpy as np
import tensorflow as tf
import focal_loss
from tensorflow import keras

from attention import Attention
# train config
DATA_DIR = 'Data' # data directory 
CLASS_NUM = 10 # number of classes
dim = 32 # number of dimensions
epochs_num = 20 # number of epochs
segments_N = 5
dict = {}

sess = tf.compat.v1.InteractiveSession()

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', DATA_DIR, 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# normalization: make the value to [norm_min, norm_max]
def normalization(value, norm_max, norm_min):
    value_max = np.max(value)
    value_min = np.min(value)
    normal_k = (norm_max - norm_min) / (value_max - value_min)
    normal_value = np.trunc(norm_min + normal_k * (value - value_min))
    return normal_value

# Fill the missing values
def append_equal(first_value,value):
    if value.shape(0) < first_value.shape(0):
        value = np.append(value,[0])
    return value

# Segments Generator
def segments_gen(value, segments_N):
    value = np.array_split(value, segments_N, axis = 1)
    return value

tarin_labels = mnist.train._labels
train_value = normalization(mnist.train._images, 255, 0)
test_labels = mnist.test._labels
test_images = normalization(mnist.test._images, 255, 0)

def Attention_Model(inputs, feature_cnt, dim):
    h_block = int(feature_cnt*dim/32/2)
    inputs = keras.layers.Flatten()(inputs)
    while(h_block >= 1):
        h_dim = h_block * 32
        inputs = keras.layers.Dense(h_dim,activation='selu',use_bias=True)(inputs)
        h_block = int(h_block/2)
    inputs = keras.layers.Dense(feature_cnt,activation='softmax',name='attention')(inputs)
    return inputs

# def Attention(hidden_states):
#     hidden_size = 1
#     # Inside dense layer
#     #              hidden_states            dot               W            =>           score_first_part
#     # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
#     # W is the trainable weight matrix of attention Luong's multiplicative style score
#     score_first_part = keras.layers.Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
#     #            score_first_part           dot        last_hidden_state     => attention_weights
#     # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
#     # h_t = keras.layers.Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
#     h_t = keras.layers.Lambda(lambda z: keras.backend.expand_dims(z, axis=-1))(hidden_states)
#     score = keras.layers.dot([score_first_part, h_t], [2, 1], name='attention_score')
#     attention_weights = keras.layers.Activation('softmax', name='attention_weight')(score)
#     # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
#     context_vector = keras.layers.dot([hidden_states, attention_weights], [1, 1], name='context_vector')
#     pre_activation = keras.layers.concatenate([context_vector, h_t], name='attention_output')
#     attention_vector = keras.layers.Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
#     return attention_vector

# create the model
def The_Model(value, feature_cnt, Model_name, dim):
    keras.backend.clear_session()
    inputs = keras.layers.Input(shape=(feature_cnt,), dtype='float32') # 157
    # at_inputs = inputs.reshape(inputs.shape[0],feature_cnt)
    print('--------',inputs.shape)
    if Model_name == 'LSTM':
        model_output = keras.layers.Dropout(0.5)(inputs)
        model_output = keras.layers.Embedding(value.shape[0], 256, input_length = value.shape[1])(model_output)
        model_output = keras.layers.Bidirectional(keras.layers.LSTM(128))(model_output)
        # model_output = Attention_Model(model_output, CLASS_NUM, dim)s
        # model_output = keras.layers.Lambda(lambda z: keras.backend.expand_dims(z, axis=-1))(model_output)
        model_output = Attention(name='atten')(model_output)
        # model_output = tf.cast(model_output, tf.float32) # 10
        # model_output = keras.layers.multiply([model_output, at_inputs])
        # model_output = keras.layers.Embedding(value.shape[0], dim, input_length = value.shape[1])(model_output)
    elif Model_name == 'GRU':
        model_output = keras.layers.Embedding(value.shape[0], 256, input_length = value.shape[1])(inputs)
        model_output = keras.layers.Bidirectional(keras.layers.GRU(128))(model_output)
        model_output = Attention_Model(model_output, CLASS_NUM, dim)
        model_output = keras.layers.Lambda(lambda z: keras.backend.expand_dims(z, axis=-1))(model_output)
        # model_output = Attention(name='atten')(model_output)
    else:
        raise ValueError("The model does not exist")
    return inputs, model_output


inputs, model_output = The_Model(train_value, 10, 'GRU', 64)
model = keras.Model(inputs=inputs,outputs=model_output,name='model12')
sgd = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss=[focal_loss.categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=sgd)
model.summary()

print(segments_gen(train_value, segments_N)[0].shape, tarin_labels.shape)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True, 
                 write_grads=True,
                 write_images=True,
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)

checkpointer = keras.callbacks.ModelCheckpoint('model',
                                   verbose=1, save_weights_only=False, period=1)
model.fit(segments_gen(train_value, segments_N), tarin_labels,
          batch_size=128,
          epochs=epochs_num,
          validation_data=(test_images, test_labels),
          callbacks=[checkpointer,tensorboard_callback])