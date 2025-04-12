'''
fit架构专属模块
指定callbacks
用于构建模型训练时的callBack(模型回调) (无论是keras内置或是自定义)
input
    log train_param
general method
    on_start/epoch/batch
已构建callbacks:
1.checkpoint
2.
'''

'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ checkpoint @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''

import  tensorflow as tf
import numpy as np
import package.utils.staticProc as static
SAVE_FLOAT = 2 #保留有效数字位数
# ALPHA = 4
# GAMMA = 6
# SITA = 0.3

#过拟合采用参数
ALPHA = 0.5
GAMMA = 4
SITA = 1

THRESHOLD_PORTION_CALLBACK = 0.5 #cor_current_val > (self.cor_val * self.threshold_portion)

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
    '''
    平均相关系数监测器 返回与对象a b 相关的非线性加权平均值 res
    :param a:float
    :param b:float
    :param alpha:
    :param gamma:
    :param sita:
    :return:float
    '''
    assert a != None or b != None, 'one of a or b must unequal None!'
    if a == None: res = b
    if b == None: res = a
    if a != None and b != None:
        #1.一般权重计算
        res = (ALPHA * a + GAMMA * b) / (1 + SITA*abs(a - b))

    return res

def cor_schedule(cor_current,cor_best):
    '''
    判断当前相关系数是否为最佳相关系数组合
    :param cor_current: 元组(cor_current_train,cor_current_val)
    :param cor_best: 元组(cor_best_train,cor_best_val)
    :return: bool True or False
    '''
    res = False
    score_current = monitor_meanCor(*cor_current)
    score_best = monitor_meanCor(*cor_best)
    if score_current > score_best or type(score_best)==type(np.nan):res = True
    return res

class CkptCorSaveSchedule(tf.keras.callbacks.Callback):
    '''
    基于模型在训练集与测试集上相关系数关系制定的模型保存逻辑类
    1.模型在当前epoch的data_val上表现较最佳cor更好时，进行训练集验证
    2.基于cor_schedule进行模型保存判断
    '''
    def __init__(self,batch_train=32,batch_val=None,data_train=None,data_val=None,
                 save_weights_only=True,cor_schedule = cor_schedule):
        super(CkptCorSaveSchedule, self).__init__()
        self.save_path = None #dir0/dir1/fileName
        self.save_weights_only = save_weights_only
        self.data_train = data_train #(x_train,y_train)
        self.data_val = data_val #(x_val,y_val)
        self.cor_schedule = cor_schedule
        #init
        self.batch_train = batch_train #评估相关性时batch
        self.batch_val = batch_val if batch_val != None else batch_train
        self.threshold_portion = THRESHOLD_PORTION_CALLBACK #判定是否进行验证的cor_var阈值比



    def on_train_begin(self, logs=None):
        print('\nstart cor_train:{0} &&& cor_val:{1}\n'.format(self.cor_train,self.cor_val))

    def on_epoch_end(self, epoch, logs=None):
        '''
        在验证集相关系数较高时进行评估
        在log_val达到
        :param epoch:
        :param logs:{loss_batch:...,metircs_batch:...,loss_val_batch:...,metrics_val_batch:...}  4个浮点数
        :return:
        '''
        cor_current_val = self.evaluate_batch(self.data_val,batch=self.batch_val)
        #当验证集相关系数较高时(>best*thresh)进行训练集验证
        if cor_current_val > (self.cor_val * self.threshold_portion):
            cor_current_train = self.evaluate_batch(self.data_train,batch=self.batch_train)
            if self.cor_schedule((cor_current_train,cor_current_val),(self.cor_train,self.cor_val)):
                '''在指定的相关系数策略下进行合理模型保存'''
                self._save_model()
                print('@@save model!current cor_train & cor_val VS old best cor_train & cor_val:\n{0:} + {1}  :  {2} + {3}'
                      .format(np.round(cor_current_train,SAVE_FLOAT),np.round(cor_current_val,SAVE_FLOAT),
                              np.round(self.cor_train,SAVE_FLOAT),np.round(self.cor_val,SAVE_FLOAT)))
                self.cor_train = cor_current_train
                self.cor_val = cor_current_val

    def evaluate_batch(self,data,batch):
        '''
        通过batch分段评估整体预测值相关性
        :param data: (Tensor_x,Tensor_y)
        :param batch: int
        :return: Tensor_cor
        '''
        x,y = data
        num_sample = x.shape[0]
        prediton = []
        for i in range(0,num_sample,batch):
            x_batch = x[i:i+batch]
            y_batch = self.model(x_batch,training = False)
            prediton.append(y_batch)
        y_pred = tf.concat(prediton,axis=0)
        cor = static.corr_tf(y_pred,y)
        return cor

    def _save_model(self):
        if self.save_weights_only:
            self.model.save_weights(
                self.save_path,
                overwrite=True,
            )
        else:
            self.model.save(
                self.save_path,
                overwrite=True,
            )

    def train_env_compile(self,**kwargs):
        '''
        获取Train类实例的环境参数的专属方法
        目的是提前获取fit时无法获取的参数
        '''
        self.data_train = kwargs['data_train']
        self.data_val = kwargs['data_val']
        self.set_model(kwargs['model'])
        self.add_ckpt_path(kwargs['ckpt'])
        self.cor_train = self.evaluate_batch(self.data_train,self.batch_train)
        self.cor_val = self.evaluate_batch(self.data_val,self.batch_val)

    def add_ckpt_path(self,ckpt):
        self.save_path = ckpt

#学习率随超过阈值的相关系数增大而减小
class LearningRateSchdule(tf.keras.callbacks.Callback):
    '''
    基于训练过程相关系数动态变化的学习率调整类
    '''
    def __init__(self,threshold,metric,func):
        super(LearningRateSchdule, self).__init__()
        self.metrics = metric
        self.threshold = threshold #与func计算的epoch表现得分相关的乘数阈值
        self.func = func #func 计算当前epoch表现得分 param:logs[name] ...
        self.best = -np.inf
        self.switch = False



    def on_epoch_end(self, epoch, logs=None):
        metrics = [logs[name] for name in self.metrics]
        current = self.func(*metrics)
        if current >= self.threshold*self.best :
            self.switch = True
            scheduled_lr = self.scheduled(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr) #将self.model.optimizer.lr 替换为 scheduled_lr
            print('')
        else :self.switch = False

    def scheduled(self):
        pass

class EarlyStop(tf.keras.callbacks.Callback):
    def __init__(self,choose_mertrics,judge_func,patience=20):
        super(EarlyStop, self).__init__()
        self.best = -np.inf
        self.wait = 0
        self.patience = patience
        self.metics = choose_mertrics
        self.judge_func = judge_func #评分函数 input为epoch结束时传入的log参数

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
    num_t,num_v = 1000,100
    units = 10
    data_train = (tf.random.normal((num_t,units)),tf.random.normal((num_t,1)))
    data_val = (tf.random.normal((num_v,units)),tf.random.normal((num_v,1)))
    ckpt = 'out/checkpoint/test_ckptcall_model/test_ckpt'
    ckpter = CkptCorSaveSchedule(batch_train=64)

    model = tf.keras.Sequential([tf.keras.layers.Dense(units,activation='relu'),
                                 tf.keras.layers.Dense(units*10,activation='relu'),
                                 tf.keras.layers.Dense(1,activation='relu')])

    '''
    test CkptCorSaveSchedule
    测试该callback在train instance中能够参与的过程
    '''
    #train.go_train-> train_env_compile
    #测试callback的实时训练环境添加参数功能
    ckpter.train_env_compile(data_train=data_train,data_val=data_val,model=model,ckpt='test')

    #train.go_train->fit->epoch_end 0
    #测试ccs的模型评估能力
    cor = ckpter.evaluate_batch(data_train,64)
    print(cor.numpy())

    #train.go_train->fit->epoch_end 1
    #测试ccs外包的相关系数保存能力
    cor_curent = [abs(ckpter.evaluate_batch(data_train,64)),abs(ckpter.evaluate_batch(data_val,64))]
    cor_best = [cor_curent[0]*0.6,cor_curent[1]*1]
    print(cor_schedule(cor_curent,cor_best=cor_best))

    # train.go_train->fit->epoch_end 1
    #检验ccs的模型保存能力
    ckpter.add_ckpt_path(ckpt)
    ckpter._save_model()
    model.load_weights(ckpt)
    print(model(data_val[0]))

    #train.go_train->fit->epoch_end 3
    #检验ccs的模型
    callbacks = [ckpter]
    model.compile(loss=tf.keras.losses.MeanSquaredError())
    history = model.fit(x=data_train[0],y=data_train[1],batch_size=32,epochs=100,callbacks=callbacks,validation_data=data_val)


    #train.go_train->fit fit
    #检验模型的在train架构中的适配能力
    import package.train.train_class as tc

    import package.model.model as deepm
    Model = deepm.FNN_res1
    model_param = {'blocks_arrange':[5,None,5],'activation':'relu','dropout_rate':0.5,'single_block_num':3}

    trainer_te = tc.Train()
    trainer_te.set_data(data=data_train)
    trainer_te.set_Model(Model)
    trainer_te.cross_validation(param_model=model_param,ckpt_head='out/checkPoint/testCheckpoint',fold_num=10,
                                range_fold=[0,2],epoch=10,batch_t=32,batch_v=32,if_pred=True,if_saveHis=True,)


    #测试得分函数

    ALPHA = 4
    GAMMA = 6
    SITA = 0.3
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider


    # 定义函数
    def calculate_res(a, b, ALPHA, GAMMA, SITA):
        return (ALPHA * a + GAMMA * b) / (1 + SITA * np.abs(a - b))


    # 创建网格和初始参数
    a = np.linspace(0, 1, 100)
    b = np.linspace(0, 1, 100)
    A, B = np.meshgrid(a, b)
    ALPHA_init, GAMMA_init, SITA_init = 1.0, 1.0, 1.0
    RES = calculate_res(A, B, ALPHA_init, GAMMA_init, SITA_init)

    # 初始化图像
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(A, B, RES, cmap='viridis', edgecolor='none')

    # 设置坐标轴标签和标题
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('res')
    ax.set_title('res = (ALPHA*a + GAMMA*b) / (1 + SITA*|a - b|)')

    # 添加参数调节滑块
    axcolor = 'lightgoldenrodyellow'
    ax_alpha = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor=axcolor)
    ax_gamma = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
    ax_sita = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)

    slider_alpha = Slider(ax_alpha, 'ALPHA', 0.1, 5.0, valinit=ALPHA_init)
    slider_gamma = Slider(ax_gamma, 'GAMMA', 0.1, 5.0, valinit=GAMMA_init)
    slider_sita = Slider(ax_sita, 'SITA', 0.1, 5.0, valinit=SITA_init)


    # 更新函数
    def update(val):
        alpha = slider_alpha.val
        gamma = slider_gamma.val
        sita = slider_sita.val
        res = calculate_res(A, B, alpha, gamma, sita)
        ax.clear()
        surf = ax.plot_surface(A, B, res, cmap='viridis', edgecolor='none')
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_zlabel('res')
        ax.set_title('res = (ALPHA*a + GAMMA*b) / (1 + SITA*|a - b|)')
        fig.canvas.draw_idle()


    slider_alpha.on_changed(update)
    slider_gamma.on_changed(update)
    slider_sita.on_changed(update)

    plt.tight_layout()
    plt.show()

