'''
1.SnpEmbedding
    编码基因分型数据的抽象父类,期望将一个基本的基因分型tensor编码为一种特定结构的tensor

'''
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

    def embeding(self,x):
        x = tf.cast(x,dtype=tf.int32)
        res = tf.one_hot(x,depth=self.depth)
        return res

    def __call__(self,x,*args, **kwargs):
        return self.embeding(x)


if __name__ == '__main__':
    #:test1:Snp2Vec
    print(':test1:Snp2vec')
    te = tf.constant([[1,2,3,4,5,6],[2,3,4,5,1,2]])
    s2v = Snp2Vec(6)
    res = s2v(te)
    print('result\n{}'.format(res))






