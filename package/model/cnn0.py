'''
接受一个三维张量
'''
import tensorflow as tf
class CNN0(tf.keras.layers.Layer):
    def __init__(self):
        super(CNN0, self).__init__()

    def call(self, inputs, *args, **kwargs):
        pass