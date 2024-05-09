'''
接受一个三维张量,

'''
import tensorflow as tf



class VggBlock(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides,padding='same',activation='relu',dropout_rate=0.2,if_dropout=True):
        super(VggBlock, self).__init__()
        self.c0_0 = tf.keras.layers.Conv1D(filters,kernel_size,strides,padding)
        self.b0_0 = tf.keras.layers.BatchNormalization()
        self.a0_0 = tf.keras.layers.Activation(activation)
        self.c0_1 = tf.keras.layers.Conv1D(filters,kernel_size,strides,padding)
        self.b0_1 = tf.keras.layers.BatchNormalization()
        self.a0_1 = tf.keras.layers.Activation(activation)
        self.p0 = tf.keras.layers.MaxPool1D(pool_size=2,strides=1,padding='same')
        self.d0 = tf.keras.layers.Dropout(dropout_rate)
        self.if_dropout = if_dropout
    def call(self, inputs, *args, **kwargs):
        x = self.c0_0(inputs)
        x = self.b0_0(x)
        x = self.a0_0(x)
        x = self.c0_1(x)
        x = self.b0_1(x)
        x = self.a0_1(x)
        x = self.p0(x)
        if self.if_dropout:
            x = self.d0(x)
        return x

class VGG0(tf.keras.models.Model):
    def __init__(self,conv_param_list,dropout_dense_rate,out_units,out_act=None):
        super(VGG0, self).__init__()
        self.convblocks = [VggBlock(*param) for param in conv_param_list]
        self.flatten = tf.keras.layers.Flatten()

        self.dense0 = tf.keras.layers.Dense(2048,activation='relu')
        self.drop0 = tf.keras.layers.Dropout(dropout_dense_rate)
        self.dense1 = tf.keras.layers.Dense(2048,activation='relu')
        self.drop1 = tf.keras.layers.Dropout(dropout_dense_rate)
        self.dense2 = tf.keras.layers.Dense(1024,activation='relu')
        self.drop2 = tf.keras.layers.Dropout(dropout_dense_rate)
        self.denseLast = tf.keras.layers.Dense(out_units,out_act)

    def call(self, inputs, training=None, mask=None):
        #conv
        for convBlock in self.convblocks:
            inputs = convBlock(inputs)
        #flatten
        x = self.flatten(inputs)

        #MLP
        x = self.dense0(x)
        x = self.drop0(x)
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        x = self.denseLast(x)

        return x


class VGG(tf.keras.models.Model):
    def __init__(self,out_units,out_act,dropout_rate=0.2,dropout_dense_rate=0.2):
        super(VGG, self).__init__()
        self.c0_0 = tf.keras.layers.Conv1D(filters=64,strides=1,padding='same')
        self.b0_0 = tf.keras.layers.BatchNormalization()
        self.a0_0 = tf.keras.layers.Activation('relu')
        self.c0_1 = tf.keras.layers.Conv1D(filters=64,strides=1,padding='same')
        self.b0_1 = tf.keras.layers.BatchNormalization()
        self.a0_1 = tf.keras.layers.Activation('relu')
        self.p0 = tf.keras.layers.MaxPool1D(pool_size=2,strides=1,padding='same')
        self.d0 = tf.keras.layers.Dropout(dropout_rate)

        self.c1_0 = tf.keras.layers.Conv1D(filters=128,strides=1,padding='same')
        self.b1_0 = tf.keras.layers.BatchNormalization()
        self.a1_0 = tf.keras.layers.Activation('relu')
        self.c1_1 = tf.keras.layers.Conv1D(filters=128,strides=1,padding='same')
        self.b1_1 = tf.keras.layers.BatchNormalization()
        self.a1_1 = tf.keras.layers.Activation('relu')
        self.p1 = tf.keras.layers.MaxPool1D(pool_size=2,strides=1)
        self.d1 = tf.keras.layers.Dropout(dropout_rate)

        self.c2_0 = tf.keras.layers.Conv1D(filters=256,strides=1,padding='same')
        self.b2_0 = tf.keras.layers.BatchNormalization()
        self.a2_0 = tf.keras.layers.Activation('relu')
        self.c2_1 = tf.keras.layers.Conv1D(filters=256,strides=1,padding='same')
        self.b2_1 = tf.keras.layers.BatchNormalization()
        self.a2_1 = tf.keras.layers.Activation('relu')
        self.p2 = tf.keras.layers.MaxPool1D(pool_size=2,strides=1)
        self.d2 = tf.keras.layers.Dropout(dropout_rate)

        self.c3_0 = tf.keras.layers.Conv1D(filters=512,strides=1,padding='same')
        self.b3_0 = tf.keras.layers.BatchNormalization()
        self.a3_0 = tf.keras.layers.Activation('relu')
        self.c3_1 = tf.keras.layers.Conv1D(filters=512,strides=1,padding='same')
        self.b3_1 = tf.keras.layers.BatchNormalization()
        self.a3_1 = tf.keras.layers.Activation('relu')
        self.p3 = tf.keras.layers.MaxPool1D(pool_size=2,strides=1)
        self.d3 = tf.keras.layers.Dropout(dropout_rate)

        self.c4_0 = tf.keras.layers.Conv1D(filters=512,strides=1,padding='same')
        self.b4_0 = tf.keras.layers.BatchNormalization()
        self.a4_0 = tf.keras.layers.Activation('relu')
        self.c4_1 = tf.keras.layers.Conv1D(filters=512,strides=1,padding='same')
        self.b4_1 = tf.keras.layers.BatchNormalization()
        self.a4_1 = tf.keras.layers.Activation('relu')
        self.p4 = tf.keras.layers.MaxPool1D(pool_size=2,strides=1)
        self.d4 = tf.keras.layers.Dropout(dropout_rate)

        self.flatten = tf.keras.layers.Flatten()

        self.dense0 = tf.keras.layers.Dense(2048,activation='relu')
        self.d5 = tf.keras.layers.Dropout(dropout_dense_rate)
        self.dense1 = tf.keras.layers.Dense(2048,activation='relu')
        self.d6 = tf.keras.layers.Dropout(dropout_dense_rate)
        self.dense2 = tf.keras.layers.Dense(1024,activation='relu')
        self.d7 = tf.keras.layers.Dropout(dropout_dense_rate)
        self.dense3 = tf.keras.layers.Dense(out_units,out_act)

    def call(self, inputs, *args, **kwargs):
        pass




class ResBlcok(tf.keras.layers.Layer):
    def __init__(self,convLayer,filters):
        super(ResBlcok, self).__init__()
        #继承自tf.keras.layers.Layer的实例，input shape = (n,m,c) -- output shape = (n,m,filters)
        self.conv = convLayer
        self.filter1D = tf.keras.layers.Conv1D(filters=filters,strides=1,padding='same')
        self.lay_nor = tf.keras.layers.LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        y = self.conv(inputs)
        y += self.filter1D(x)
        y = self.lay_nor(y)
        return  y





if __name__ == "__main__":
    #:test
    x = tf.cast(tf.random.uniform((2,1024,1),maxval=10,minval=0,dtype=tf.int32),dtype=tf.float32)
    y = tf.random.uniform((2,1))

    conv_block = [[64,3,1,],[128,3,1,],[256,3,1,'same'],[512,3,1,],[512,3,1,]]
    vgg16 = VGG0(conv_block,dropout_dense_rate=0.2,out_units=1)
    vgg16.compile(loss=tf.keras.losses.MeanSquaredError())
    vgg16.fit(x,y,batch_size=32,epochs=1)
    vgg16.summary()
    for i,var in enumerate(vgg16.layers[0].trainable_variables):
        print(i,var.shape)


