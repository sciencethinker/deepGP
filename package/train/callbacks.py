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
















