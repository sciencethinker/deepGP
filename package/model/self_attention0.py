'''
通用 self-attention模块
Self_attention:继承自tf.keras.layers.Layer
'''
import tensorflow as tf
import numpy as np
import package.model.snp_embedding as snp_emb
import package.model.cnn0 as cnn_0
K=tf.keras.backend

class MultiSelf_attention0(tf.keras.layers.Layer):
    '''
    ps 多头操作是构建三个隐藏层为units*head的W，将Q,K,V分离后进行注意力操作，之后在-1维合并为units*head的高维向量，
       之后经过dense层降维至d_model
    scale dot self attention
    input (n,m,d) tensor has demension of d
    out  (n,m,d)  tensor has demension of d
    train parameters:
                     Wq:(d,units)
                     Wk:(d,units)
                     Wv:(d,units)
    '''
    def __init__(self,units,multi_head=8,use_bais=True,initializer = None):
        '''

        :param units: one head's units ->int
        :param multi_head: how many heads ->int
        :param use_bais:
        :param initializer:
        '''
        super(MultiSelf_attention0, self).__init__()
        self.units = units
        self.multi_head = int(multi_head)
        self.use_bais = use_bais
        self.initializer = initializer

        #check out multi_head if or not >0
        if multi_head <=0:raise Exception('multi_head must be a intenger and > 0! your multi_head is {}'.format(multi_head))

    def build(self, input_shape):
        #vector dimension -> em.[1 0 0 ]
        d = input_shape[-1]
        #总共单元数为期望维度与多头之积，通过一个W实现多头机制
        units_total = self.units*self.multi_head
        self.w_q = self.add_weight(name='w_q',shape=(d,units_total),initializer=self.initializer)
        self.w_k = self.add_weight(name='w_k',shape=(d,units_total),initializer=self.initializer)
        self.w_v = self.add_weight(name='w_v',shape=(d,units_total),initializer=self.initializer)
        self.w_0 = self.add_weight(name='w_0',shape=(units_total,self.units),initializer=self.initializer)

        if self.use_bais:
            self.b_q = self.add_weight(name='b_q',shape=(units_total,))
            self.b_k = self.add_weight(name='b_k',shape=(units_total,))
            self.b_v = self.add_weight(name='b_v',shape=(units_total,))
            self.b_0 = self.add_weight(name='b_0',shape=(self.units,))


    def call(self, q,k,v, *args, **kwargs):
        #make q k v shape (n,m,d*h) h:multi head num
        if self.use_bais:
            q = K.bias_add(K.dot(q,self.w_q),self.b_q)
            k = K.bias_add(K.dot(k,self.w_k),self.b_k)
            v = K.bias_add(K.dot(v,self.w_v),self.b_v)
        else:
            q = K.dot(q,self.w_q)
            k = K.dot(k,self.w_k)
            v = K.dot(v,self.w_v)
        #linear projction

        q = K.relu(q)
        k = K.relu(k)
        v = K.relu(v)

        d_k = self.units #dimension of q and k

        #重构q k v，使其成为(n*h,m,units),用于后续点积使用  在最后维度分开头并合并至0维度
        q_ = tf.concat(tf.split(q,self.multi_head,axis=-1),0)
        k_ = tf.concat(tf.split(k,self.multi_head,axis=-1),0)
        v_ = tf.concat(tf.split(v,self.multi_head,axis=-1),0)


        #q (n,m,units) dot  k.T (n,d,units) -> (by permunate_dimension) e->linear e
        a = K.batch_dot(q_,K.permute_dimensions(k_,(0,2,1)))
        #scale
        a /= d_k**0.5
        #softmax
        a = K.softmax(a,axis=-1)
        #dropout

        #matmul dot(A,V)
        out = K.batch_dot(a,v_)    # n*h,m,units

        #split & concat : [n*h,m,units] -> [n,m,units]
        out = tf.concat(tf.split(out,self.multi_head,axis=0),-1)  #在0维度上分开之前合并的头,并拼接在最终维度

        out = K.dot(out,self.w_0) + self.b_0 if self.use_bais else K.dot(out,self.w_0)
        # out = K.relu(out)
        return out

    def attention_map(self,x):
        '''
        get a attention map
        :param x:
        :return: tensor
        '''
        pass


class Position_encoding(tf.keras.layers.Layer):
    '''
    positional encoding:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

    how to tansfer?
    pos = Position_encoding(d_model,maxlen);pos(tensor) --return--> tensor shape=[maxlen,d_model]
    create a instance ,then use instance to return a position matrix

    '''
    def __init__(self,d_model,maxlen,constant=10000):
        '''
        :param d_model:
        :param maxlen:
        :param constant:
        '''
        super(Position_encoding, self).__init__()
        #初始化位置编码矩阵
        self.d_model = d_model
        self.constant = float(constant)
        self.encoding = np.array([[pos/np.power(self.constant,(i-i%2)/d_model) for i in range(d_model)]
                                  for pos in range(maxlen)])
        self.encoding[:,0::2] = tf.cos(self.encoding[:,0::2])
        self.encoding[:,1::2] = tf.sin(self.encoding[:,1::2])
        self.encoding = tf.convert_to_tensor(self.encoding)

    def call(self, inputs, *args, **kwargs):
        '''

        :param inputs:
        :param ismask:
        :param args:
        :param kwargs:
        :return:
        '''
        if len(inputs.shape) != 3:raise Exception('Position Encoding --- input shape has wrong!need 3d tensor:(batch_size,length,d_model) but input.shape={}'.format(inputs.shape))
        batch_size,length,d_model = inputs.shape #n,length,dimension
        #检查维度是否相同
        if d_model != self.encoding.shape[-1]:
            raise Exception('Position Encoding dimension dosen\'t match:d_input(x):{0},d_encoding:{1}'.format(d_model,self.encoding.shape[-1]))
        out = self.encoding[:length,:]
        return out

class LayerNorm(tf.keras.layers.Layer):
    '''
    when Layer_norm?
    在进行残差相加操作后对tensor进行Layer Normaliztion

    How Layer_norm?
    对每个样本的每个维度进行标准化:
    tensor:(n,m,d)
    mean = mean(x_i) i blg [1,d]
    var = var(x_i)
    o = (x-mean)/(var+eps)**0.5
    o = gama*o + beta

    '''
    def __init__(self,d_model,eps = 1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = self.add_weight(name='gamma',shape=[d_model],initializer='ones',trainable=True)
        self.beta = self.add_weight(name='beta',shape=[d_model],initializer='zeros',trainable=True)
        self.eps = eps

    def call(self, inputs, *args, **kwargs):
        mean,var = tf.nn.moments(inputs,axes=-1,keepdims=True)
        out = (inputs-mean) / tf.sqrt(var+self.eps)
        out = self.gamma * out + self.beta
        return out

class FullLayer(tf.keras.layers.Layer):
    '''
    含有1隐藏层的全连接网络模型
    '''
    def __init__(self,units_list,activation_list=['relu',None]):
        super(FullLayer, self).__init__()
        self.dense0 = tf.keras.layers.Dense(units=units_list[0],activation=activation_list[0])
        self.dense1 = tf.keras.layers.Dense(units=units_list[1],activation=activation_list[1])

    def call(self, inputs, *args, **kwargs):
        al = self.dense0(inputs)
        al = self.dense1(al)

        return al

class EncoderLayer0(tf.keras.layers.Layer):
    '''
    inculde:
    single muti-self-attention
    dropout
    residual add & layer normalization
    full connection layer
    dropout
    '''
    def __init__(self,d_model,attention_units,multi_head,use_bais,full_units,full_act=['relu',None],
                 dropout_rates = [0.2,0.2],
                 attention_initializer=None,):
        '''
        :param d_model: dimension of inputs
        :param attention_units: w_q,k,v的隐藏单元数
        :param multi_head: attenion头数量
        :param use_bais: multi-attention是否使用偏差b_q,k,v
        :param full_units: list [a,b] len(full_units) = 2  注意力模块全连接层列表
        :param full_act:
        :param dropout_rates:
        :param attention_initializer:
        '''
        super(EncoderLayer0, self).__init__()
        self.attenion = MultiSelf_attention0(units=attention_units,multi_head=multi_head,
                                             use_bais=use_bais,initializer=attention_initializer)
        self.norm1 = LayerNorm(d_model) #ps:需要分开构建layerNorm，因为层内存在训练参数
        self.dropout1 = tf.keras.layers.Dropout(dropout_rates[0])
        self.ffn = FullLayer(full_units,activation_list=full_act)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rates[1])


    def call(self, inputs, *args, **kwargs):
        x_orig = inputs
        x = self.attenion(inputs,inputs,inputs)
        # dropout
        x = self.dropout1(x)
        #residual add
        x += x_orig
        #normalization
        x = self.norm1(x)

        #full connection
        x = self.ffn(x)
        #dropout
        x = self.dropout2(x)
        #residual add
        x += x_orig
        x = self.norm2(x)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self,maxlen,d_model,
                 attention_units,multi_head,use_bais,
                 full_units,full_act=['relu',None],
                 dropout_rates = [0.2,0.2],
                 attention_initializer=None,
                 pos_CONSTANT=10000,
                 bocks_num=8):
        super(Encoder, self).__init__()
        self.pos_enc = Position_encoding(d_model,maxlen,pos_CONSTANT)
        self.blocks = [EncoderLayer0(d_model,attention_units,multi_head,
                                     use_bais,full_units,full_act,
                                     dropout_rates,attention_initializer) for _ in range(bocks_num)]

    def call(self, inputs, *args, **kwargs):
        x_dtype = inputs.dtype
        pos = tf.cast(self.pos_enc(inputs),x_dtype)
        x = inputs + pos
        for block in self.blocks:
            x = block(x)
        return x

class SNPAtten0(tf.keras.Model):
    def __init__(self,maxlen,d_model,
                 fp_units,fp_acts,fp_drop,
                 attention_units,multi_head,use_bais,
                 full_units,full_act=['relu',None],
                 full_dropout_rates = [0.2,0.2],
                 attention_initializer=None,
                 pos_CONSTANT=10000,
                 bocks_num = 8,):
        '''

        :param maxlen: int -- for position encoding
        :param d_model:dimension of each sequence
        :param fp_units:
        :param fp_acts:
        :param attention_units: int
        :param multi_head:int -- how many heads your Attention model has
        :param use_bais:
        :param full_units:
        :param full_act:
        :param dropout_rates:
        :param attention_initializer:
        :param pos_CONSTANT:
        :param bocks_num:
        :param snp_depth:
        '''
        super(SNPAtten0, self).__init__()
        self.snp2vec = snp_emb.Snp2Vec(depth=d_model)
        self.decoders = Encoder(maxlen,d_model,
                                attention_units,multi_head,use_bais,
                                full_units,full_act,full_dropout_rates,
                                attention_initializer,
                                pos_CONSTANT,
                                bocks_num)

        #last layer ——> full prediction (abbr."fp")
        self.ffn_pre = FullLayer(units_list=fp_units,activation_list=fp_acts)
        self.fp0 = tf.keras.layers.Dense(units=fp_units[0],activation=fp_acts[0],name='fp0')
        self.fp1 = tf.keras.layers.Dense(units=fp_units[1],activation=fp_acts[1],name='fp1')
        self.fp_drop1 = tf.keras.layers.Dropout(fp_drop,name='fp_drop1')
        self.fp2 = tf.keras.layers.Dense(units=fp_units[2],activation=fp_acts[2],name='fp2')
        self.fpAL = tf.keras.layers.Dense(units=fp_units[-1],activation=fp_acts[-1],name='fpAL')

    def call(self,x):
        x = self.snp2vec(x)
        x = self.decoders(x)

        x_pre = self.ffn_pre(x[:,0,:])
        y = self.fp0(x_pre)
        y = self.fp1(y)
        y = self.fp_drop1(y)
        y = self.fp2(y)
        y = self.fpAL(y)
        return y

class ChrAtten0(tf.keras.Model):
    def __init__(self,conv_param_list
                 ,snp2chr_list,chr_emb_units,
                 maxlen,
                 fp_units,fp_acts,fp_drop,
                 atten_units,multi_head,use_bais,
                 full_units,full_act=['relu',None],
                 full_dropout_rates = [0.2,0.2],
                 attention_initializer=None,
                 pos_CONSTANT=10000,
                 blocks_num = 8):
        '''

        :param snp2chr_list:
        :param chr_emb_units:
        :param maxlen:
        :param fp_units:
        :param fp_acts:
        :param fp_drop:
        :param atten_units:
        :param multi_head: int -- how many heads your Attention model has;
        :param use_bais:
        :param full_units:
        :param full_act:
        :param full_dropout_rates:
        :param attention_initializer:
        :param pos_CONSTANT:
        :param bocks_num:
        '''
        super(ChrAtten0, self).__init__()
        #全局基因组卷积层，并使用一层全连接层转换为与编码向量shape相同的向量
        self.global_conv = [cnn_0.VggBlock(*param) for param in conv_param_list]
        self.global_conv.append(tf.keras.layers.Flatten())
        self.global_conv.append(tf.keras.layers.Dense(chr_emb_units))


        self.emb = snp_emb.ChrEmbed(snp2chr_list,chr_emb_units)
        # self.pos_enc = Position_encoding(chr_emb_units,maxlen,pos_CONSTANT)
        self.decoders = Encoder(maxlen,chr_emb_units,
                                atten_units,multi_head,use_bais,
                                full_units,full_act,full_dropout_rates,
                                attention_initializer,
                                pos_CONSTANT,
                                blocks_num)
        #last attention layer
        self.last_atten = MultiSelf_attention0(chr_emb_units,multi_head)

        #last layer ——> full prediction (abbr."fp")
        self.ffn_pre = FullLayer(units_list=fp_units,activation_list=fp_acts)
        self.fp0 = tf.keras.layers.Dense(units=fp_units[0],activation=fp_acts[0],name='fp0')
        self.fp1 = tf.keras.layers.Dense(units=fp_units[1],activation=fp_acts[1],name='fp1')
        self.fp_drop1 = tf.keras.layers.Dropout(fp_drop,name='fp_drop1')
        self.fp2 = tf.keras.layers.Dense(units=fp_units[2],activation=fp_acts[2],name='fp2')
        self.fpAL = tf.keras.layers.Dense(units=fp_units[-1],activation=fp_acts[-1],name='fpAL')

    def call(self, inputs, training=None, mask=None):
        '''
        charAtten forward
        :param inputs: n,m
        :param training:
        :param mask:
        :return:
        '''
        #全局卷积编码
        n,length = inputs.shape
        x_conv = tf.expand_dims(inputs,axis=-1) #在最后一维增加1维，成为3Dtensor
        for conv in self.global_conv:
            x_conv = conv(x_conv)
        n,d_model = x_conv.shape
        x_conv = tf.expand_dims(x_conv,axis=1) #将二维tensor x_conv转换为序列长度为1,编码长度为d_model的三维tensor

        #染色体向量编码
        x = self.emb(inputs)

        x = tf.concat((x,x_conv),axis=1) #axis = 1 ，在序列维度进行拼接

        x = self.decoders(x)
        x_pre = self.ffn_pre(x[:,0,:])
        x_pre = self.fp0(x_pre)
        x_pre = self.fp1(x_pre)
        x_pre = self.fp_drop1(x_pre)
        x_pre = self.fp2(x_pre)
        x_pre = self.fpAL(x_pre)

        return x_pre


class ChrAtten1(tf.keras.Model):
    def __init__(self):
        super(ChrAtten1, self).__init__()







if __name__ == "__main__":
    # #:test1:Multiself_Attention0()(tensor)
    # print('\n:test1:Multiself_Attention0()(tensor)\nresult:')
    # te = tf.random.uniform((2,5,3))
    # n,m,d = te.shape
    # model = MultiSelf_attention0(d)
    # result = model(te,te,te)
    # print(result)
    #
    # #:test2:Positional_encoding
    # print('\n:test2:Positional_encoding\nresult:')
    # maxlen,d_model = 20,6
    # te = tf.constant([[[i for i in range(d_model)] for pos in range(maxlen-3)]for n in range(2)],dtype=tf.float32)
    # pos_em = Position_encoding(d_model,maxlen)
    # result = pos_em(te)
    # print("position encoding matrix:\n{}".format(result))
    # print("Position_encoding'variables:{}".format(pos_em.variables))
    #
    # #:test3:Layer_norm
    # print("\n:test3:Layer_norm\n")
    # te = tf.constant([[[1,2,3,4,5],[6,7,8,9,10]],
    #                   [[2,3,4,5,6],[7,8,9,10,11]]],dtype=tf.float32)
    # print(te)
    # lay_nor = LayerNorm(te.shape[-1])
    # result = lay_nor(te)
    # print('result:\n{}'.format(result))
    #
    # #:test4:FullLayer
    # print('\n:test4:FullLayer\n')
    # te = tf.random.uniform((2,4,6),minval=0,maxval=10)
    # full_layer = FullLayer((10,te.shape[-1]))
    # result = full_layer(te)
    # print('result:\n{0}\nfull_layer.variables:\n{1}'.format(result,full_layer.variables))
    #
    # te = tf.ones((2,5,10))
    # te_l = tf.keras.layers.Dropout(0.5)
    # tf.keras.layers.Dropout(0.5)(te)
    # print(te_l(te))
    #
    # #:test5:EncoderLayer0
    # print("\n:test5:EncoderLayer0\n")
    # te = tf.random.uniform((2,6,10),0,10)
    # enc = EncoderLayer0(te.shape[-1],te.shape[-1],8,True,full_units=[int(te.shape[-1]*1.5),te.shape[-1]])
    # result = enc(te)
    # print('result\n{0}\nvariables\n{1}'.format(result,enc.variables))
    # for i,val in enumerate(enc.variables):
    #     print('###{0}--@@@@@@@@@@@@@@@@@@@@ {1} @@@@@@@@@@@@@@@@@@@@\n{2}\n'.format(i,val.name,val),)
    #
    # #:test6:Encoder
    # print('\n:test6:Encoder\n')
    # te = tf.random.uniform((2,6,10),0,10)
    # encs = Encoder(bocks_num=8,maxlen=te.shape[1],d_model=te.shape[-1],attention_units=te.shape[-1],
    #                multi_head=8,use_bais=True,full_units=[int(te.shape[-1]*2),te.shape[-1]])
    # result = encs(te)
    # print('result\n{}'.format(result))
    # for i,val in enumerate(encs.variables):
    #     print('{0}---{1}'.format(i,val.name))
    #
    # #:test7:SNPAtten
    # te = tf.random.uniform(shape=(2,6),maxval=3,minval=0,dtype=tf.int32)
    # d_model = 3
    # snp_model = SNPAtten0(maxlen=te.shape[1]+1,d_model=d_model,
    #                       fp_units=[d_model*3,d_model*2,d_model,1],fp_acts=['relu','relu','relu',None],fp_drop=0.2,
    #                       attention_units=d_model,multi_head=8,use_bais=True,
    #                       full_units=[d_model*2,d_model])
    # snp_model.snp2vec(te)
    # res = snp_model(te)
    # print('res\n{}'.format(res))

    # :test8:SNPAtten---train
    print('\n:test8:SNPAtten---train\n')
    te = tf.random.uniform(shape=(100,50),maxval=4,minval=0,dtype=tf.int32)
    te_y = tf.random.uniform(shape=(100,1))
    #data_process
    d_model = 5
    te = snp_emb.Snp2Vec(depth=d_model).add_coloumn(te, )
    snp_emb.Snp2Vec(depth=d_model).embeding(te)
    snp_num = te.shape[1]
    ####################### done #######################

    snp_model = SNPAtten0(maxlen=snp_num,d_model=d_model,
                          fp_units=[d_model * 3, d_model * 2, d_model, 1], fp_acts=['relu', 'relu', 'relu', None],
                          fp_drop=0.2,
                          attention_units=d_model, multi_head=8, use_bais=True,
                          full_units=[d_model * 2, d_model])
    snp_model.compile(loss=tf.keras.losses.MeanSquaredError())

    history = snp_model.fit(x=te,y=te_y,batch_size=32,epochs=10)
    snp_model.summary()

    # #:test9:ChrAtten0---train
    # print('\n#:test9:ChrAtten0---train\n')
    # sample_num = 2
    # maxlen = 21
    # chr_s = 512
    # te_x = tf.random.uniform((sample_num,46731),maxval=2,minval=0,)
    # snpnum_list = [8769, 3484, 2442, 2153, 2262, 1811, 2583, 2176, 2168, 2368,
    #                1241, 1383, 1021, 2643, 2468, 2159, 1372, 1094, 981,  2153]
    # te_y = tf.random.uniform((sample_num, 1))
    # ############################################################################
    # # te_x = tf.random.uniform((sample_num,1000))
    # # te_y = tf.random.uniform((sample_num,1))
    # # snpnum_list = [250,250,250,200,50]
    # # maxlen = 6
    # # chr_s = 32
    #
    #
    # conv_list = [[64,3,1,],[128,3,1],[128,3,1],[128,3,1],[128,3,1],[64,3,1,]]
    # chr_model = ChrAtten0(conv_param_list=conv_list,maxlen = maxlen,snp2chr_list=snpnum_list,chr_emb_units=chr_s,
    #                       fp_units=[chr_s,int(chr_s*0.8),int(chr_s*0.5),1], fp_acts=['relu', 'relu', 'relu', None],fp_drop=0.2,
    #                       atten_units=chr_s,multi_head=8,use_bais=True,
    #                       full_units=[int(chr_s*1.5),int(chr_s)])
    # chr_model(te_x)
    # chr_model.compile(loss=tf.keras.losses.MeanSquaredError())
    # history = chr_model.fit(x=te_x,y=te_y,batch_size=32,epochs=10)
    #
    # chr_model.summary()








