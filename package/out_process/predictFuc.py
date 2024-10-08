'''
预测APIfunction构建
use:给定model(无ckpt)实例，构建一个predicter,predicter通过初始化model(载入ckpt)完成predicter
predicter既是指定模型的性能预测接口，指定数据集，可以构建模型
'''
import os
import numpy as np
import random
import sys
import time
import tensorflow as tf
import package.utils.staticProc as staProc

class Prediction:
    '''
    构造预测类
    '''
    #各指标保留有效数字全局设置
    save_number = {'predCTime':None,'totalTime':1,'perTime':4,
                        'cor':3,'lossMes':2,'result':1}

    def __init__(self,model,name = 'default',model_mes = None):

        '''
        :param model:指定初始化model 传入model实例，单个predicion实例应该只对应一个实例model，通过checkpoint载入；
                    可以load不同data进行模型性能评估
        :param name: 指定预测器名称 通常是model
        :param model_mes: 说明模型信息
        '''
        self.name = name
        self.model_mes = model_mes
        self.model = model
        self.data = None       #(x,y)构建的元组，y可选为None，当其为None时表示仅测试model的时间效率
        self.checkpoint = None #

        #单个数据集评估需要更新的指标 ID作用：一个9位整数，用于区分载入的数据集

        self.predMes = {'id':None,'dataMes':None,'dataSize':None,'predCTime':None,'totalTime':None,'perTime':None,
                        'cor':None,'lossMes':None,'result':None}# 8 characteristics
        '''
        id:                 int
        dataMes:            str messenge of data       e.m.  x - s1_50k_5224;y - age_5460
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

    def load_weights(self,ckpt_path):
        print('load model weights at:{}'.format(ckpt_path))
        self.model.load_weights(ckpt_path)

    def creteID(self):
        randint_start = 1e9
        randint_end = 1e10-1
        self.predMes['id'] = random.randint(randint_start,randint_end) #生成一个9位随机整数作为ID

    @staticmethod
    def time_measure(func,*args,**kwargs):
        time_start = time.time()
        result = func(*args,**kwargs)
        time_end = time.time()
        time_spend = time_end - time_start
        return result,time_spend

    def predict(self,input):
        result,time_spend = self.time_measure(self.model,input)
        return result,time_spend

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
        if self.predMes['result'] == None:raise Exception('self.predMes[\'result\'] has\'t any result!')
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

        # 1.预测结果 result 预测总时间 total time
        self.predMes['result'],self.predMes['totalTime'] = self.predict(x)
        self.predMes['result'] = self.predMes['result']
        self.predMes['totalTime'] = np.round(self.predMes['totalTime'],self.save_number['totalTime'])

        # 2.单样本预测效率 pertime mean+std
        if perNum == None:perNum = x.shape[0] #单个预测数量
        try:#if perNum = int
            per_time = []
            for index in range(perNum):
                per_x = tf.reshape(x[index],(1,*list(x.shape[1:]))) #转变切片形状，与x一致
                res,t = self.predict(per_x)
                per_time.append(t)
            per_time = tf.constant(per_time) #1D tensor (n,)
            (per_mean,per_std)= tf.math.reduce_mean(per_time),tf.math.reduce_std(per_time)
            self.predMes['perTime'] = (np.round(per_mean.numpy(),self.save_number['perTime']),
                                       np.round(per_std.numpy(),self.save_number['perTime']))
        except Exception as e: # if perNum = 0 or False or what
            self.predMes['perTime'] = None
            print('\n ***warning: don\'t estimate per_time! ***\nbecause:',file=sys.stderr)
            print(e)

        if y != None: #如果存在y
            # 3.预测loss mean std

            self.predMes['lossMes'] = np.round(self.loss_mean_std(y),self.save_number['lossMes'])

            # 4.预测cor  corcoef
            self.predMes['cor'] = np.round(self.corrf(y).numpy(),self.save_number['cor'])
        else:
            self.predMes['lossMes'] = None
            self.predMes['cor'] = None

        #获取当前时间 5.predCtime
        self.predMes['predCTime'] = '_'.join(time.ctime().split())

        print('estimation done! \nyou can use x.log(path,way) to save this estimation!')

    def log(self,path,way = 'a'):
        '''

        :param path:
        :param way: 记录在日志中的方式，默认为追加至文件末尾'a',-----'c','r'
        :return:
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,way) as file :
            if way == 'a': #追加内容前换行
                file.write('\n')
            values = []
            for key,value in self.predMes.items():
                if key == 'result':continue
                values.append(str(value)+'\t')
            file.writelines(values)
        print('write log done as \'{0}\' ! log at:{1}'.format(way,path))

    def saveResult(self,file_path):
        if os.path.exists(file_path):
            with open(file_path) as file:
                pass
        else:pass

    def draw(self):
        '''
        绘图API
        :return:
        '''
        pass


class CrossPrediction(Prediction):
    def __init__(self,cross_num):
        super(CrossPrediction, self).__init__()
        self.cross_num = cross_num




if __name__ == "__main__":
    #:test1:prediction
    print('\n#:test:1.prediction')
    import package.data_process.data_set as ds
    input_file = 'data/input/s1_50k_5224.raw'
    label_file = 'data/label/laOrig_10age_5021.phen'
    data_dict = {}
    x_all, y_all, x_pre, id_pre = ds.loadData(input_file,
                                              label_file)  # x_all,y_all  -> 尚未区分训练验证集；x_pre,id_pre->未知label的x与对应id
    stddev = tf.math.reduce_std(y_all)
    mean = tf.reduce_mean(y_all)
    y_all = (y_all - mean) / stddev
    in_feature = x_all.shape[-1]

    import package.model.model as deepm
    deepG = deepm.DeepGblup(**{'ymean':mean,'snp_num':in_feature})
    predDeepGBLUP0 = Prediction(deepG,'deepGBLUP','testModel mes')


    predDeepGBLUP0.load_data(x_all,y_all,'test data mes')
    #:test1.1:load data
    print('\n:test:load data\n{0}\n{1}'.format(predDeepGBLUP0.predMes,predDeepGBLUP0.data))

    #:test1.2:predict
    print(predDeepGBLUP0.predict(x_all))

    #:test1.3:estimate
    predDeepGBLUP0.estimate(10)
    print('\n:test:estimate\n{}'.format(predDeepGBLUP0.predMes))

    predDeepGBLUP0.predMes['dataMes'] = 'test '
    predDeepGBLUP0.log('test.txt','a')



