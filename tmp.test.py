import package.train.compile as comp
import tensorflow as tf
import numpy as np
import package.model.fnn as fn
import package.train.callbacks as callback

cor_metric = comp.Corralation()

size = 512
x = tf.cast(tf.random.uniform((size,4673),maxval=2,minval=0,dtype=tf.int32),dtype=tf.float32)
y = tf.random.uniform((size,1))

x_val = tf.cast(tf.random.uniform((int(size*0.4), 4673), maxval=2, minval=0, dtype=tf.int32), dtype=tf.float32)
y_val = tf.random.uniform((int(size*0.4), 1))

batch = 32
layer_arrange = [512,None,1024]

fres1 = fn.FNN_res1(layer_arrange,activation='relu',dropout_rate=0.5,single_block_num=3,
                 last_dense_units=1,last_dens_act=None)

save_check = '../tmp/save/test' #目录名称+文件前缀名称


fres1.compile(loss=tf.keras.losses.MeanSquaredError(),metrics=cor_metric)
ckpt = callback.CkptSaveByMeanCor(save_check,None,'val_corralation')
his = fres1.fit(x,y,batch,50,validation_data=(x_val,y_val),callbacks=[ckpt],validation_batch_size=512)
fres1.summary()

fres2 = fn.FNN_res1(layer_arrange,activation='relu',dropout_rate=0.5,single_block_num=3,
                 last_dense_units=1,last_dens_act=None)
fres2.load_weights(save_check)



import package.utils.staticProc as static
def pred_r(true, pred, h=1):
    import math
    pred_mean = tf.reduce_mean(pred)
    true_mean = tf.reduce_mean(true)
    f1 = tf.reduce_sum((pred - pred_mean) * (true - true_mean))
    f2 = tf.sqrt(tf.reduce_sum((pred - pred_mean) ** 2) * tf.reduce_sum((true - true_mean) ** 2))
    if f2 == 0:
        r = 0.
    else:
        cor = f1 / f2
        r = float(cor / math.sqrt(h))
    return r
static.corr_tf(fres2(x),y)
static.corr_tf(fres2(x_val),y_val)


print(tf.reduce_mean(fres2(x)-y)**2)
print(tf.reduce_mean((fres2(x_val)-y_val)**2))
print('------cor -> ')
print(pred_r(fres2(x),y))
print(pred_r(fres2(x_val),y_val))

# #单纯的save_weights方式,使用index文件进行load
# fres1.save_weights(save_check)
# fres2 = fn.FNN_res1(layer_arrange,activation='relu',dropout_rate=0.5,single_block_num=3,
#                  last_dense_units=1,last_dens_act=None)
# fres2.load_weights(save_check)
#
# fres2(x) == fres1(x) #True

# #checkpoint保存方式
# ckpt = tf.train.Checkpoint(fres1) #保存文件前缀为teset-i，与保存文件字符串存在出入
# ckpt.save(save_check)
# #ckpt.restore() -> 只能使得fres1进行load

#save 方式
# fres1.save(save_check)
# tf.keras.models.load_model(save_check)

import tensorflow as tf
strategy = tf.distribute.MirroredStrategy()
strategy.num_replicas_in_sync
tf.nn.compute_average_loss()
tf.keras.models.Model.fit