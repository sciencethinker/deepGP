'''
检验model水品
'''
import platform
import package.system_process.system_args as sy
sysargs = sy.getArgs()
sysargs = sy.getArgs()
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = sysargs['gpu']
os.environ["CUDA_VISIBLE_DEVICES"] = '0' if 'gpu' not in sysargs.keys() else sysargs['gpu']
import package.data_process.file_process as fp
import package.data_process.data_set as ds
import package.model.model as deepm
import package.train.train_class as tc
import tensorflow as tf

#准备数据集

