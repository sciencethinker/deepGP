'''
全面重构train_struct 函数!，将其整合为类
'''
import os
import sys

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
CALLBACK_LIST = [callbacks.CkptCorSaveSchedule()]
class Train:
    '''
    Train instance useage:
        #create a Train instance
    trainer = Train()
        #set Model
    trainer.set_Model(Model) #get a Model class
        #set all data
    trainer.set_data(data_all) #a couple or list with two Tensors(x,y)
        #you can go to cross validation after set Model and data
    trainer.corss_validation(param_model,ckpt_head,fold_num,range_fold,epoch,batch_t,batch_v,
                         if_pred,if_saveHis,save_history_head='out/trainHistory/model',
                         *args,**kwargs)

    '''
    def __init__(self):
        #默认设置
        #compile
        self.optimizer = OPTIMIZER
        self.loss = LOSS
        self.metric_list = METRICS
        #callback
        self.callback_list = CALLBACK_LIST

        #Model
        self.ckpt_path = None
        self.Model = None #class

        #data
        self.data = None # [data,labels]
        #save path

    def set_Model(self,Model):
        '''
        设置Model类,在训练前必须选取合适的Model
        '''
        self.Model = Model

    def model_init(self,parameters):
        '''
        对self.Model初始化，在训练时进行
        :param parameters:
        :return:
        '''
        assert self.Model != None,"set model ERROR!  \nself.Model must be a class  !"
        model = self.Model(**parameters)
        return model

    def set_data(self,data):
        '''
        用于设置data
        :param data:
        :return:
        '''
        self.data = data

    def cross_validation(self,param_model,ckpt_head,fold_num,range_fold,epoch,batch_t,batch_v,
                         if_pred,if_saveHis,if_fit=True,save_history_head='out/trainHistory/model',save_log='out/log/model',
                         *callback_args,**callback_kwargs):
        '''
        交叉验证API
        :param param_model:
        :param ckpt_head:
        :param fold_num:
        :param range_fold:
        :param epoch:
        :param batch_t:
        :param batch_v:
        :param if_pred:是否在每次交叉验证后进行预测
        :param if_saveHis:是否在保存训练历史
        :param if_fit:是否使用fit框架进行训练
        :param save_history_head:
        :param save_log:
        :param args:任意参数，用于callbacks重构
        :param kwargs:任意参数，用于callbacks重构
        :return:
        '''

        histories = {}
        #创建数据集生成器
        data_gen = self.make_cross_data(fold_num)
        for fold in range(fold_num):
            #初始化数据集&model，清除内存中的历史数据集&model
            model = None
            data_train = None
            data_val = None
            #跳过未选择的fold
            if fold not in range_fold:continue
            data_train,data_val = next(data_gen)
            #只在必要时产生data

            print('{:@^50}'.format(' cross validation fold:%s ' % fold))
            ckpt_path = ckpt_head + 'corss{}/model.ckpt'.format(fold)
            print('save model path:{}'.format(ckpt_path))

            #time measure
            start = time.time()

            model = self.model_init(param_model) if param_model!=None else self.Model()

            #history:dict {'loss','metrics','loss_val','metrics_val','time_total','time_each'}
            history = self.go_train(model=model,data_train=data_train,data_val=data_val,
                                    epoch=epoch,batch_t=batch_t,batch_v=batch_v,if_pred=if_pred,if_fit=if_fit,
                                    save_log=save_log,ckpt_path=ckpt_path,*callback_args,**callback_kwargs)

            #time
            end = time.time()
            time_spend = round(end - start)
            time_each = round(time_spend / epoch,2)
            history['time_total'] = [time_spend]
            history['time_each'] = [time_each]

            histories[str(fold)] = history
            print('{:*^100}\n'.format(
                'train done! fold:%s spend time total:%ss each epoch:%ss ' % (fold, time_spend, time_each)))

        if if_saveHis:
            save_history_path = save_history_head.strip('/') + '/cross{}.txt'.format(str(fold))
            self.dict_save(history, save_history_path)
            print('*****history save path:{}*****'.format(save_history_path))
            '''记录loss & metrics 每次迭代情况:
            loss 
            metrics: em. corralation_train
            loss_val 
            metrics: em. corralation_val
            time_total 
            time_each 
            '''
        return histories




    def go_train(self,model,data_train,batch_t,
                 batch_v,data_val=None,epoch=1,
                 if_pred=True,save_log=None,ckpt_path=None,
                 is_fit=True,*callback_args,**callback_kwargs):
        '''

        :param model: tf.Module or tf.keras.models.Model
        :param data_train: (x,y) Tensor
        :param batch_t: int for data_train
        :param batch_v: int for data_val
        :param data_val:(x,y)  Tensor
        :param epoch: iter
        :param if_pred: if prediction after
        :param ckpt_path: where to store
        :param is_fit:
        :param args:
        :param kwargs:
        :return: history
        '''
        if os.path.exists(ckpt_path  + '.index'):
            print('{0}\n{0}\n{0}'.format('*********************** load model *****************************'))
            model.load_weights(ckpt_path )

        for cb in self.callback_list:
            try:
                #对具有train_env_compile方法的callback调用该方法，传入所需参数
                cb.train_env_compile(model=model,data_train=data_train,data_val=data_val,ckpt=ckpt_path,*callback_args,**callback_kwargs)
            except AttributeError as e:
                pass
            finally:pass

        #选用特定框架的训练框架
        if is_fit:

            history = self._train_tf_fit(model=model, data_train=data_train, batch_t=batch_t, batch_v=batch_v,data_val=data_val,
                                         epoch=epoch,ckpt_path=ckpt_path, if_pred=if_pred,*callback_args,**callback_kwargs)
        else:history = self._train_tf(model=model, data_train=data_train, batch_t=batch_t, batch_v=batch_v,data_val=data_val,
                                         epoch=epoch,ckpt_path=ckpt_path, if_pred=if_pred,*callback_args,**callback_kwargs)

        if if_pred:
            print('****** estimate current model *******')
            try :
                preder = pred.Prediction(model,name='save_model',model_mes='None')
                preder.load_weights(ckpt_path)
                #val
                preder.load_data(x=data_val[0],y=data_val[1],mes='validation')
                preder.estimate(perNum=batch_v)
                preder.log(save_log)
                #train
                preder.load_data(x = data_train[0],y = data_train[1],mes='train')
                preder.estimate(perNum=batch_t)
                preder.log(save_log)
                print('***** estimatation done! log at :{} *****'.format(save_log))
            except Exception as e:
                print('train.go_train.EstimateWarning！faild to estimate model ！\nreason:{}'.format(e),file=sys.stderr)
        return history


    def _train_tf_fit(self,model,data_train,data_val,epoch,batch_t,batch_v,ckpt_path,*args,**kwargs):
        '''
        基于keras的fit框架进行的训练流程
        :param model:
        :param data_train:
        :param data_val:
        :param epoch:
        :param batch:
        :return:
        '''
        '''compile'''
        x_train,y_train = data_train
        x_val,y_val = data_val
        model.compile(optimizer=self.optimizer,
                  loss=self.loss,
                  metrics=self.metric_list)


        if os.path.exists(ckpt_path+'.index'):
            print('{0}\n{0}\n{0}'.format('*********************** load model *****************************'))
            model.load_weights(ckpt_path)


        history = model.fit(x=x_train,y=y_train,validation_data=(x_val,y_val),epochs=epoch,batch_size=batch_t,
                            callbacks=self.callback_list,
                            validation_batch_size=batch_v)

        return history.history

    def _train_tf(self,model,data_train,data_val,epoch,batch_t,batch_v,ckpt_path,if_pred,*args,**kwargs):
        '''
        尚未开发
        :param args:
        :param kwargs:
        :return:
        '''
        pass
        x_train,y_train = data_train
        x_val,y_val = data_val


    def __call__(self,if_cross, *args, **kwargs):
        if if_cross:
            self.cross_validation(*args,**kwargs)
        else:
            self.go_train(*args,**kwargs)

    def make_cross_data(self,k):
        '''
        data 要求[Tensor_dataSet(n,...),Tensor_label(n,...)]
        :param k: 交叉数两
        :return: 迭代器对象，使用next函数调用，每次返回单个data数据集
        '''
        if self.data == None:
            raise ValueError("Train_instance.make_cross_data: self.data is None!")
        x, y = self.data
        num_samples = x.shape[0]
        fold_size = num_samples // k

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size

            val_indices = slice(start, end)
            x_train, y_train = tf.concat([x[0:start],x[end:num_samples]],axis=0),tf.concat([y[0:start],y[end:num_samples]],axis=0)
            x_val, y_val = x[val_indices], y[val_indices]

            yield (x_train, y_train), (x_val, y_val)

    def reset_compile(self,loss=None,metrics=None,optimizer=None,*args,**kwargs):
        if loss!=None:self.loss = loss
        if metrics!=None:self.metric_list = metrics
        if optimizer!=None:self.optimizer = optimizer

    def add_callBacks(self,*cbs,):
        for cb in cbs:
            self.callback_list.append(cb) #compile is a instance

    def remove_callBacks(self,index=-1,*args,**kwargs):
        self.callback_list.pop(index)

    def dict_save(self,history, save_path):
        '''
        将字典每个key及其value按行存储，value应是list,用于存储history
        :param dict:
        :param save_path:
        :return:
        '''
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as file:
            for key in history:
                value_str = ' '.join(map(str, history[str(key)]))
                file.writelines(str(key) + ' ' + value_str + '\n')
        print('save dict at: {}'.format(save_path))





if __name__ == "__main__":
    data = [tf.random.normal((100,10)),tf.random.normal((100,1))]
    tmp = data
    trainer = Train()
    trainer.set_data(data)
    data_gen = trainer.make_cross_data(10)
    total = tf.ones((0,1))
    for i,data in enumerate(data_gen):
        data_train,data_val = data
        total = tf.concat([total,data_val[1]],axis=0)
        print(data_val)
    print(any(total == tmp[1]))










