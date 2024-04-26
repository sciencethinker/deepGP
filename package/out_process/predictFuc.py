'''
预测APIfunction构建
use:给定model(无ckpt)实例，构建一个predicter,predicter通过初始化model(载入ckpt)完成predicter
predicter既是指定模型的性能预测接口，指定数据集，可以构建模型
'''
import os
import random
import time
import tensorflow as tf
import package.utils.timeProcess as timeProc
import package.utils.staticProc as staProc
import numpy as np

class Predicton:
    '''
    构造预测类
    '''

    def __init__(self,model,name = 'default',model_mes = None):

        '''
        :param model:指定初始化model 传入model实例
        :param name: 指定预测器名称 通常是model
        :param model_mes: 指定模型信息
        '''
        self.name = name
        self.model_mes = model_mes
        self.model = model
        self.data = None
        self.checkpoint = None

        #单个数据集评估需要更新的指标 ID作用：一个9位整数，用于区分载入的数据集

        self.predMes = {'id':None,'dataMes':None,'dataSize':None,'predCTime':None,'totalTime':None,'perTime':None,
                        'cor':None,'lossMes':None,'result':None}# 8 characteristics
        '''
        id:                 int
        dataMes:            str messenge of data
        dataSize            tuple (x_shape,y_shape)
        predCtime:str       current time
        total time:         float
        per time:           (per_mean,per_std)
        cor:                scalar float
        lossMes:            (float,float) mean std
        result:             Tensor Value
        '''

    def init_model(self,ckpt,model_mes = None):
        '''
        加载模型
        :param ckpt: 模型参数路径
        :return: 返回模型
        '''
        self.checkpoint = tf.train.Checkpoint(self.model)
        self.checkpoint.restore(ckpt)
        if not model_mes == None:self.model_mes += model_mes #添加模型信息


    def load_data(self,x,y = None,mes = None):
        #加载数据时创建唯一ID
        for key in self.predMes:self.predMes[key] = None # init predMessenge
        size_x = self.dataSize(x)
        size_y = self.dataSize(y)
        self.predMes['dataSize'] = (size_x,size_y)  #添加数据集信息
        self.predMes['dataMes'] = mes
        self.creteID()
        self.data = (x,y)

    def creteID(self):
        randint_start = 1e9
        randint_end = 1e10-1
        self.predMes['id'] = random.randint(randint_start,randint_end) #生成一个9位随机数作为ID

    @staticmethod
    def time_measure(func,*args,**kwargs):
        time_start = time.time()
        result = func(*args,**kwargs)
        time_end = time.time()
        time_spend = time_end - time_start
        return result,time_spend

    def predict(self,input):
        result,time_spend = self.time_measure(self.model,input)
        return result


    def predict_per(self,input):
        pass

    @staticmethod
    def dataSize(data):
        '''
        返回单个张量的shape
        :param data:
        :return:
        '''
        if data != None:
            shape = []
            for dim in data.shape:
                shape.append(dim)
            shape = tuple(shape)
        else:shape = None
        return shape

    def loss_mean_std(self,label):
        if not self.predMes['result']:raise Exception('self.predMes[\'result\'] has\'t any result!')
        result = staProc.loss_mean_std_tf(self.predMes['result'],label)
        return result

    def corrf(self,label):
        '''
        :param label: labels tensor
        :return: 1D tensor tf.Tensor(0.97735566, shape=(), dtype=float32)
        '''
        cor = staProc.corr_tf(self.predMes['result'],label)
        return cor

    def estimate(self,perNum=None):
        '''
        更新实例属性状态
        :param perNum :单次预测次数，默认为预测样本数量
        '''
        if not self.data:
            raise Exception("haven't load data yet ! use self.load_data(self,x,y = None,mes=None) to get data!")
        x, y = self.data

        # 1.预测结果 result
        self.predMes['result'] = self.predict(x)
        # 预测总时间 total time
        self.predMes['totalTime'] = self.predict.time_spend

        # 2.单个预测效率 pertime mean+std
        if perNum == None:perNum = x.shape[0] #单个预测数量
        try:#if perNum = int
            per_time = []
            for index in range(perNum):
                per_x = self.data[0][index]
                self.predict(per_x)
                per_time.append(self.predict.time_spend)
                per_time = tf.constant(per_time)
            (per_mean,per_std)= tf.math.reduce_mean(per_time),tf.math.reduce_std(per_time)
            self.predMes['perTime'] = (per_mean,per_std)
        except Exception : # if perNum = None or 0 or False or what
            self.predMes['perTime'] = None
            print('don\'t estimate per_time')

        if y:
            # 3.预测loss mean std
            self.predMes['lossMes'] = self.loss_mean_std(y)

            # 4.预测cor  corcoef
            self.predMes['cor'] = self.corrf(y)
        else:
            self.predMes['lossMes'] = None
            self.predMes['cor'] = None

        #获取当前时间 5.predCtime
        self.predMes['predCTime'] = '_'.join(time.ctime().split())

    def log(self,path,way = 'a'):
        '''

        :param path:
        :param way: 记录在日志中的方式，默认为追加至文件末尾'a',-----'c','r'
        :return:
        '''
        pass

    def saveResult(self,file_path):
        if os.path.exists(file_path):open(file_path)
        else:pass


    def draw(self):
        pass


class CrossPrediction(Predicton):
    def __init__(self,cross_num):
        super(CrossPrediction, self).__init__()
        self.cross_num = cross_num






if __name__ == "__main__":
    #:test1:prediction
    print('\n#:test:1.prediction')
    import package.data_process.data_set as ds
    input_file = 'data/input/s0_50k_5701.raw'
    label_file = 'data/label/la10110011_100age_5460.phen'
    data_dict = {}
    x_all, y_all, x_pre, id_pre = ds.loadData(input_file,
                                              label_file)  # x_all,y_all  -> 尚未区分训练验证集；x_pre,id_pre->未知label的x与对应id
    stddev = tf.math.reduce_std(y_all)
    mean = tf.reduce_mean(y_all)
    y_all = (y_all - mean) / stddev
    in_feature = x_all.shape[-1]

    import package.data_process.file_process as fp
    model_name = 'deepGblup/'
    tmp = fp.getSnpLabel_mes(input_file,label_file) #获取snp与label文件信息以创建各类out的头目录
    ckpt_head = 'out/checkpoint/' + model_name + tmp #指定ckpt


    import package.model.model as deepm
    deepG = deepm.DeepGblup(**{'ymean':mean,'snp_num':in_feature})
    predDeepGBLUP0 = Predicton(deepG,'deepGBLUP','testModel mes')


    predDeepGBLUP0.load_data(x_all,y_all,'test data mes')
    #:test1.1:load data
    print('\n:test:load data\n{}'.format(predDeepGBLUP0.predMes))

    #:test1.2:predict
    print(predDeepGBLUP0.predict(x_all))

    #:test1.3:estimate







