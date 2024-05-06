'''
full connection model
'''
import platform

import tensorflow as tf
class ResFullBlock(tf.keras.layers.Layer):
    def __init__(self,units_list,activation,dropout_rate):
        super(ResFullBlock, self).__init__()
        assert units_list[0] == units_list[-1],\
            'dimension of output must equal dimension of input,but d_in{0}:d_out{1}'.format(units_list[0],units_list[-1])
        self.denses = [tf.keras.layers.Dense(units,activation) for units in units_list ]
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.layerNor = tf.keras.layers.LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for i,dense in enumerate(self.denses):
            inputs = dense(inputs)
        inputs = self.dropout(inputs)
        #res add
        inputs += x
        #scale
        inputs = self.layerNor(inputs)
        return inputs

class FullTransBlock(tf.keras.layers.Layer):
    def __init__(self,units,activation):
        super(FullTransBlock, self).__init__()
        self.dense = tf.keras.layers.Dense(units,activation)
    def call(self, inputs, *args, **kwargs):
        out = self.dense(inputs)
        return out



class FNN_res0(tf.keras.Model):
    def __init__(self,block_units_list,block_act,blocks_num,
                 last_units,last_action):
        super(FNN_res0, self).__init__()
        self.blocks = [ResFullBlock(block_units_list,block_act)
                                          for _ in range(blocks_num)]
        self.fl = tf.keras.layers.Dense(last_units,last_action)

    def call(self, inputs, training=None, mask=None):
        for i,block in enumerate(self.blocks):
            inputs = block(inputs)
        fl = self.fl(inputs)
        return fl

class FNN_res1(tf.keras.Model):
    def __init__(self,blocks_arrange = None,activation = None,dropout_rate = 0.5,single_block_num = 3,
                 last_dense_units = 1,last_dens_act = None):
        '''

        :param blocks_arrange: [5096,None,] list by int or None;当为None时表示不建立FullTransBlock，仅建立ResBlock
        :param activation:str or tf's Activation instance  全局除最后层外的激活方式
        :param dropout_rate: int 全局dropout rate
        :param single_block_num:int 单个resblock内部的层数
        :param last_dense_units:int 最后层的神经元数
        :param last_dens_act:str or tf's Activation instance  最后层的激活方式
        '''

        super(FNN_res1, self).__init__()
        self.denses = []
        cur_units = blocks_arrange[0]
        for i,units in enumerate(blocks_arrange):
            #len(block_arrange) = res_blocks' num
            #units为None表示不建立FullTransBlock,两个同维度的resBlock连接在一起
            if units != None:
                cur_units = units
                trans_block = FullTransBlock(cur_units,activation)
                self.denses.append(trans_block)
            block_units_list = [cur_units for _ in range(single_block_num)]
            res_block = ResFullBlock(block_units_list,activation,dropout_rate=dropout_rate)
            self.denses.append(res_block)

        self.lastDense = tf.keras.layers.Dense(last_dense_units,last_dens_act)

    def call(self, inputs, training=None, mask=None):
        for i,block in enumerate(self.denses):
            inputs = block(inputs)
        out = self.lastDense(inputs)
        return out






if __name__ == '__main__':
    #:test0:ResFullBlock.call
    print('\n:test0:FNN_res0.call\n')
    size = 1024
    x = tf.cast(tf.random.uniform((size,46731),maxval=2,minval=0,dtype=tf.int32),dtype=tf.float32)
    y = tf.random.uniform((size,1))
    units_list = [5096,5096,5096,5096]
    trans_f = FullTransBlock(units=5096,activation='relu')
    res_f = ResFullBlock(units_list=units_list,activation='relu',dropout_rate=0.2)
    res = trans_f(x)
    res = res_f(res)
    print('res:\n:{}'.format(res))

    for i,var in enumerate(res_f.trainable_variables):
        print(i,var)

    #:test0:FNN_res0.call && train
    # print('\n:test0:FNN_res0.call && train\n')
    # size = 1024
    # x = tf.cast(tf.random.uniform((size,46731),maxval=2,minval=0,dtype=tf.int32),dtype=tf.float32)
    #
    # y = tf.random.uniform((size,1))
    # units_list = [5096,5096,5096,5096]
    # res_f = FNN_res0(units_list,'relu',blocks_num=8,last_units=1,last_action=None)
    # res_f.compile(loss=tf.keras.losses.MeanSquaredError())
    # res_f.fit(x,y,batch_size=32,epochs=10)

    #:test1:FNN_res1.call && train
    print('\n:test1:FNN_res1.call && train\n')
    size = 1024
    x = tf.cast(tf.random.uniform((size,46731),maxval=2,minval=0,dtype=tf.int32),dtype=tf.float32)
    y = tf.random.uniform((size,1))

    if platform.system() == 'Windows':
        batch = 32
        layer_arrange = [5,None,5]
    else:
        layer_arrange = [5120,*[None for _ in range(3)],4096,*[None for _ in range(3)],2048,*[None for _ in range(3)],
                         1024,*[None for _ in range(3)],512,*[None for _ in range(3)],256,*[None for _ in range(3)],
                         128,*[None for _ in range(3)]]
        batch = 512
    fres1 = FNN_res1(layer_arrange,activation='relu',dropout_rate=0.5,single_block_num=3,
                     last_dense_units=1,last_dens_act=None)
    fres1.compile(loss=tf.keras.losses.MeanSquaredError())
    fres1.fit(x,y,batch,10)
    fres1.summary()










