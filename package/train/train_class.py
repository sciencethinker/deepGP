'''
全面重构train_struct 函数!，将其整合为类
'''
import os

import tensorflow as tf
import numpy as np
import time
import package.train.callbacks as callbacks
import package.train.compile as compile


class Train:
    def __init__(self):
        #默认设置

        self.lr = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metric_list = [compile.Corralation()]
        self.callback_list = []

        self.ckpt_path = None
        self.Model = None #class
        self.data_train = None
        self.data_val = None
        self.data_all = None

    def set_model(self,parameters):
        assert self.Model != None,"set model ERROR!  \nself.Model must be a class !"
        model = self.Model(*parameters)
        return model

    def go_train(self,param,ckpt_path,batch,batch_val,epoch):

        model = self.set_model(param)
        model.compile(optimizer=self.optimizer(self.lr),
                  loss=self.loss,
                  metrics=self.metric_list)

        if os.path.exists(ckpt_path+'.index'):
            model.load_weights(ckpt_path)
            print('{0}\n{0}\n{0}'.format('*********************** load model *****************************'))
        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
        history = model.fit()


    def __call__(self,if_cross, *args, **kwargs):
        if if_cross:
            self.cross_validation(*args,**kwargs)
        else:
            self.go_train(*args,**kwargs)


    def make_corss_data(self):
        pass

    def cross_validation(self,*args,**kwargs):
        pass

    def reset_compile(self,*args,**kwargs):
        pass

    def add_compile(self,compile,):
        pass

    def reset_callBacks(self,*args,**kwargs):
        pass




