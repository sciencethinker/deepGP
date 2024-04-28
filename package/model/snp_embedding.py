'''
1.SnpEmbedding
    编码基因分型数据的抽象父类,期望将一个基本的基因分型tensor编码为一种特定结构的tensor

'''
import numpy as np

import tensorflow as tf

class SnpEmbedding:
    def __init__(self):
        pass

    def embeding(self,x,*args,**kwargs):
        '''

        :param x:
        :return:
        '''
        return x

    def __call__(self,x,*args, **kwargs):
        '''

        :param x:
        :param args:
        :param kwargs:
        :return:
        '''
        return self.embeding(x,*args,**kwargs)

class Snp2Vec(SnpEmbedding):
    def __init__(self,depth):
        '''
        :param map:
        '''
        self.depth = depth

    def embeding(self,x,onehot_dtype=tf.float32):
        x = tf.cast(x,dtype=tf.int32)
        res = tf.one_hot(x,depth=self.depth,dtype=onehot_dtype)
        return res

    def __call__(self,x,*args, **kwargs):
        x = self.embeding(x)
        return x

    def add_coloumn(self,x,add_elem = -1):
        '''
        在x指定位置构建标志符
        :param x: 2D Tensor shape = (n,length)
        :param add_elem:-1
        :return:
        '''
        try:
            num,length = x.shape
            shape = (num,1)
        except ValueError as ve:
            num,length,d_model = x.shape
            shape = (num,1,d_model)
        except Exception as e :raise Exception("Snp2Vec-add_coloumn:shape\'s error x.shape={},hope a 2D or 3D tensor".format(x.shape))

        dtype = x.dtype
        if add_elem == 0:
            add_coloum = tf.zeros(shape=shape, dtype=dtype)  # 生成num,1的元素权威add_elem的tensor
        else:
            add_coloum = add_elem * tf.ones(shape=shape,dtype=dtype) #生成num,1的元素权威add_elem的tensor
        x = tf.concat((add_coloum,x),axis=1)
        return x

    def random_pick(self,x,pick_num,min_snp,max_snp,seed=42,axis = 1):
        '''
        :param x:2D or 3D Tensor
        :param seed: random seed
        :return:
        '''
        tf.random.set_seed(seed=seed)
        index = tf.range(min_snp,max_snp)
        index = tf.random.shuffle(index)[0:pick_num]
        index = tf.sort(index) #按照顺序排序
        x = tf.gather(x,index,axis=axis) #选取snp序列维度
        tf.random.set_seed(None)#取消全局设置
        return x




if __name__ == '__main__':
    #:test1:Snp2Vec
    print(':test1:Snp2vec')
    te = tf.constant([[1,2,3,4,5,6,],[2,3,4,5,1,2,]])
    te = Snp2Vec(depth=7).add_coloumn(te,0)
    s2v = Snp2Vec(6)
    res = s2v(te)
    print('result\n{}'.format(res))

    #:test2:Snp2Vec.random_pick
    print('\n:test2:Snp2Vec.random_pick\n')
    x = tf.random.uniform((2,20),maxval=20,dtype=tf.int32)
    x = Snp2Vec(depth=20).random_pick(x,10,5,20)
    print('res\n{}'.format(x))



