'''
原作者模型框架
'''

import tensorflow as tf
'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ model @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
#局部链接层
class LocalLinear(tf.keras.Model):
    def __init__(self,in_features,kernel_size,local_feature,stride,bias=True,init_seed = 10):
        super(LocalLinear, self).__init__()
        #hou many snp for one sample
        self.in_feature = in_features
        #for one local kernel size
        self.kernel_size = kernel_size
        #how many out nodes for one local kernel'output
        self.local_feature = local_feature
        #stride
        self.stride = stride
        #padding
        self.padding = kernel_size-1
        #how many licese for one sample
        self.fold_num = (self.in_feature + self.padding -self.kernel_size)//stride + 1

        #init W and b
        initializer = tf.keras.initializers.GlorotUniform(seed=init_seed)
        self.W = tf.Variable(initializer(shape=(self.fold_num,self.kernel_size,self.local_feature)))
        self.b = tf.Variable(tf.zeros([self.fold_num,self.local_feature])) if bias else None

    def call(self,X):
        '''
        :param X: snp_matrix:X.shape = [samples,snps_num] = [n,p]
        test X:tf.random.normal([1000,100])
        :return: AL
        '''

        #对张量X的最后一维的右侧进行0填充
        # pad_dimension = [[0, self.padding if i == int(tf.rank(X)) - 1 else 0] for i in range(int(tf.rank(X)))]
        pad_dimension = [[0,0],[0,self.padding]]
        X = tf.pad(X,pad_dimension,mode="CONSTANT")

        #对张量X(n,p+pad)->(n,1,p+pad,1)进行重复切割形成新张量X(n,fold_num,kernel_size,1) 左1用于填充 右1表示snp位点的表示向量长度
        X = tf.reshape(X,[-1,1,X.shape[1],1])#X:(n,p+pad)->(n,1,p+pad,1)
        '''
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        X = tf.image.extract_patches(X,sizes=[1,1,self.kernel_size,1],
                                     strides=[1,1,1,1],
                                     rates=[1,1,1,1],
                                     padding='VALID') #X.shape = [n,1,fold_num,kernel_size]
        X = tf.reshape(X,[-1,self.fold_num,self.kernel_size])

        #前向传播 X(n,fold_num,1,kernel_size) @ W(fold_num,kernel_size,local_feature) + b(fold_num,local_feature)
        X = tf.reshape(X,[-1,X.shape[1],1,X.shape[2]]) #X.shape = [n,fold_num,1,kernel_size]
        #多维矩阵乘法
        #AL.shape = [n,fold_num,1,1]
        AL = tf.matmul(X,self.W)
        AL = tf.reshape(AL,[-1,AL.shape[1],1])+self.b #AL.shape = [n,fold_num,1] b.shape = [fold_num,1]
        AL = tf.reshape(AL,[-1,AL.shape[1]]) #AL.shape = [n,fold_num]
        return AL

#整个神经网络前向传播过程
class DeepGblup(tf.keras.Model):
    def __init__(self,**kwargs):
        ymean = kwargs['ymean']
        snp_num = kwargs['snp_num']
        super(DeepGblup, self).__init__()
        self.mean = ymean
        '''
        ################################################
        ############### 定义LCL常数CONSTANT #############
        ################################################
        '''
        LCL1 = {'kernel_size':5,'local_feature':1,'stride':1}
        LCL2 = {'kernel_size':3,'local_feature':1,'stride':1}


        self.c1 = LocalLinear(in_features=snp_num,kernel_size=LCL1['kernel_size'],local_feature=LCL1['local_feature'],stride=LCL1['stride'])
        self.c2 = tf.keras.layers.LayerNormalization()
        self.c3 = tf.keras.layers.Activation(tf.keras.activations.gelu)
        self.c4 = LocalLinear(in_features=snp_num,kernel_size=LCL2['kernel_size'],local_feature=LCL2['local_feature'],stride=LCL2['stride'])

        '''activation Gelu !!!!!!!!!!'''

        self.decoder = tf.keras.layers.Dense(1, activation=None)

    def call(self, X):
        x = self.c1(X)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x += X
        al = self.decoder(x)

        return al + self.mean

