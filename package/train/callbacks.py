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

def monitor_meanCor(log):
    pass

class CheckpointSave(tf.keras.callbacks.Callback):
    def __init__(self,save_path,metric_t_name,metric_v_name=None):
        super(CheckpointSave, self).__init__()
        self.metric_t_name = metric_t_name
        self.metric_v_name = metric_v_name
        self.save_mapth = save_path

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

    def on_epoch_end(self, epoch, logs=None):

        self._save_model()



#学习率随超过阈值的相关系数增大而减小
class LearningRateSchdule(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LearningRateSchdule, self).__init__()


class EarlyStop(tf.keras.callbacks.Callback):
    def __init__(self,patience=20):
        super(EarlyStop, self).__init__()
        self.best = np.inf
        self.wait = 0
        self.patience = patience
    def on_epoch_end(self, epoch, logs=None):
        pass











