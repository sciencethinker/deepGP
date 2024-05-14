import package.train.compile as comp
import tensorflow as tf
import numpy as np
import package.model.fnn as fn


TRAIN_COR_ALPHA = 0.4
VAR_COR_GAMMA = 0.6

@tf.function
def average_(a, b, alpha=0.4, gamma=0.6):
    assert a != None and b != None, 'one of a or b must unequal None!'
    if a == None: res = b
    if b == None: res = a

    mean = tf.cast(alpha * a + gamma * b, dtype=tf.float32)
    abs = tf.cast(tf.abs(a - b), dtype=tf.float32)
    res = mean - abs
    return res
class MonitorCor():
    def __init__(self):
        self.score = tf.constant([-np.inf],dtype=tf.float32)
    def monitor_cor_average(self,log):
        is_save = False
        tcor = log['corralation']
        vcor = log['val_corralation']
        score = average_(tcor,vcor,TRAIN_COR_ALPHA,VAR_COR_GAMMA)
        if score > self.score:
            self.score = score
            is_save = True
        return is_save
    def __call__(self, log,*args, **kwargs):
        return self.monitor_cor_average(log)






filepath = '../'
ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,monitor=MonitorCor(),save_best_only=True,
                                          save_weights_only=True,save_freq='epoch')

cor_metric = comp.Corralation()

size = 1024
x = tf.cast(tf.random.uniform((size,4673),maxval=2,minval=0,dtype=tf.int32),dtype=tf.float32)
y = tf.random.uniform((size,1))

x_val = tf.cast(tf.random.uniform((int(size*0.4), 4673), maxval=2, minval=0, dtype=tf.int32), dtype=tf.float32)
y_val = tf.random.uniform((int(size*0.4), 1))

batch = 32
layer_arrange = [512,None,1024]

fres1 = fn.FNN_res1(layer_arrange,activation='relu',dropout_rate=0.5,single_block_num=3,
                 last_dense_units=1,last_dens_act=None)

save_check = '../tmp.ckpt'


fres1.compile(loss=tf.keras.losses.MeanSquaredError(),metrics=cor_metric)
his = fres1.fit(x,y,batch,10,validation_data=(x_val,y_val),)
fres1.summary()

fres1.save_weights('../')
fres2 = fn.FNN_res1(layer_arrange,activation='relu',dropout_rate=0.5,single_block_num=3,
                 last_dense_units=1,last_dens_act=None)


