'''
fit架构专属模块
指定callbacks
用于构建模型训练时的callBack(模型回调) (无论是keras内置或是自定义)

已构建callbacks:
1.checkpoint
2.
'''

'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ checkpoint @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
import  tensorflow as tf
import numpy as np

ALPHA = 0.4
GAMMA = 0.6
SITA = 1

def checkpoint(filePath,monitor,save_best=True,save_weight_only=True):
    '''
    :param filePath:check point path
    :param monitor: choose to how to checkpoint
    :param save_best:just save best model param during training time
    :param save_weight_only:just save weights when save time
    :return:
    '''
    call_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=filePath,monitor=monitor,
                                                   save_best_only=save_best,save_weight_only=save_weight_only)
    #return instance object
    return call_ckpt

def monitor_meanCor(a,b,alpha=ALPHA,gamma=GAMMA,sita=SITA):
    assert a != None or b != None, 'one of a or b must unequal None!'
    if a == None: res = b
    if b == None: res = a
    if a != None and b != None:
        mean = tf.cast(alpha * a + gamma * b, dtype=tf.float32)
        interval = tf.cast(tf.abs(a - b), dtype=tf.float32)
        res = mean - sita*interval
    return res

class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self,data_train,data_val,model,metrics):
        super(Evaluate, self).__init__()
        self.data_train = data_train #(x_train,y_train)
        self.data_val = data_val #(x_val,y_val)
        self.model = model
    def on_epoch_end(self, epoch, logs=None):
        '''
        在log_val达到
        :param epoch:
        :param logs:
        :return:
        '''

    def evaluate_batch(self,batch):
        pass



class CkptSaveByMeanCor(tf.keras.callbacks.Callback):
    '''
    基于模型在训练集与测试集上的相关系数的相关系数关系制定的模型保存逻辑
    1.当模型在当前epoch在data_val上表现较最佳cor更好时，进行训练集验证
    '''
    def __init__(self,save_path,metric_t_name,metric_v_name=None,save_weights_only=True):
        '''

        目前仅支持在epoch后进行模型保存
        :param save_path:
        :param metric_t_name:
        :param metric_v_name:
        :param save_weights_only:
        '''
        super(CkptSaveByMeanCor, self).__init__()
        #save path 给定保存的目录&文件名称，模型保存时创建3个文件checkpoint name.index name.data-0... 程序基于.index载入文件
        self.save_mapth = save_path
        self.metric_t_name = metric_t_name
        self.metric_v_name = metric_v_name
        self.save_weights_only = save_weights_only
        self.best = tf.cast(-np.inf,dtype=tf.float32)

    def _save_model(self):
        if self.save_weights_only:
            self.model.save_weights(
                self.save_mapth,
                overwrite=True,
            )
        else:
            self.model.save(
                self.save_mapth,
                overwrite=True,
            )

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):

        metric_t = logs[self.metric_t_name] if self.metric_t_name != None else None
        metric_v = logs[self.metric_v_name] if self.metric_v_name != None else None
        if metric_t==None and metric_v==None:
            current = tf.cast(np.inf,dtype=tf.float32)
        else:
            currtent = monitor_meanCor(metric_t,metric_v,ALPHA,GAMMA,SITA)
        if currtent >= self.best:
            self._save_model()
            self._current_epoch = epoch
            print('******@@@@@@@++++++@@@@@@ epoch:{0} save model!train {1}  val{2} @@@@@@@++++++@@@@@@******'
                  .format(self._current_epoch,metric_t,metric_v))
            self.best = currtent

#学习率随超过阈值的相关系数增大而减小
class LearningRateSchdule(tf.keras.callbacks.Callback):
    def __init__(self,threshold,metric,func):
        super(LearningRateSchdule, self).__init__()
        self.threshold = threshold
        self.metrics = metric
        self.func = func
        self.best = -np.inf
        self.switch = False



    def on_epoch_end(self, epoch, logs=None):
        metrics = [logs[name] for name in self.metrics]
        current = self.func(*metrics)
        if current >= self.threshold :
            self.switch = True
            scheduled_lr = self.scheduled(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
            print('')
        else :self.switch = False

    def scheduled(self):
        pass



class EarlyStop(tf.keras.callbacks.Callback):
    def __init__(self,choose_mertrics,judge_func,patience=20):
        super(EarlyStop, self).__init__()
        self.best = np.inf
        self.wait = 0
        self.patience = patience
        self.metics = choose_mertrics
        self.judge_func = judge_func

    def on_epoch_end(self, epoch, logs=None):
        metrics = [logs[metric] for metric in self.metics]
        current = self.judge_func(*metrics)
        if current >= self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.stop_epoch = epoch
                print('stop training at epoch :{0},best socre={1}'.format(epoch,self.best))





if __name__ == "__main__":
    monitor_meanCor(None,1)








