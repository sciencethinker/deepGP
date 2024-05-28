'''
全面重构train_struct 函数!，将其整合为类
'''
import os

import tensorflow as tf
import numpy as np
import time
import package.train.callbacks as callbacks
import package.train.compile as compile
import package.out_process.predictFuc as pred
LR = 0.0001
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR)
LOSS = tf.keras.losses.MeanSquaredError()
METRICS = [compile.Corralation(),]
CALLBACK_LIST = None
class Train:
    def __init__(self):
        #默认设置

        self.lr = LR
        self.optimizer = OPTIMIZER
        self.loss = LOSS
        self.metric_list = METRICS
        self.callback_list = CALLBACK_LIST

        self.ckpt_path = None
        self.Model = None #class


    def set_model(self,parameters):
        assert self.Model != None,"set model ERROR!  \nself.Model must be a class  !"
        model = self.Model(*parameters)
        return model

    def _train_tf_fit(self,model,data_train,data_val,epoch,batch,lr):
        '''compile'''
        metrics = self.metric_list
        model.compile(optimizer=self.optimizer,
                  loss=self.loss,
                  metrics=self.metric_list)

        if os.path.exists(self.ckpt_path+'.index'):
            print('{0}\n{0}\n{0}'.format('*********************** load model *****************************'))
            model.load_weights(self.ckpt_path)

        model.fit(epoch,callbacks = self.callback_list)


    def _train_tf(self):
        pass


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
        self.callback_list.append(compile) #compile is a instance

    def reset_callBacks(self,*args,**kwargs):
        pass




