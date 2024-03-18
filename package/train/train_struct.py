'''
训练整合模块
'''
import tensorflow as tf
import time
import os
import package.train.callbacks as callBacks
import package.train.compile as compile

class TimeMeasure:
    def __init__(self,func):
        self.func = func
        self.time_spend = None

    def __call__(self, *args, **kwargs):
        time_start = time.time()
        result = self.func(*args,**kwargs)
        time_end = time.time()
        self.time_spend = time_end - time_start
        return result

def dict_save(dict, save_path):
    '''
    将字典每个key及其value按行存储，value应是list,用于存储history
    :param dict:
    :param save_path:
    :return:
    '''
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    with open(save_path, 'w') as file:
        for key in dict:
            value_str = ' '.join(map(str,dict[str(key)]))
            file.writelines(str(key) + ' ' + value_str + '\n')
    print('save dict at: {}'.format(save_path))

@TimeMeasure
def train(model,data_train,data_val,epoch,batch,
                 lr_init,
                 h = 1,
                 optimizer=tf.keras.optimizers.Adam,
                 shuffle_or_not=True,shuffle_size = 1024,ckpt_path='checkPoint/testCheckpoint/test'):
    ''' data '''
    data_train = data_train.shuffle(buffer_size=shuffle_size).batch(batch) if shuffle_or_not else data_train.batch(batch)
    data_val = data_val.batch(batch)

    ''' compile '''
    #loss
    loss = compile.loss()
    #metrics list
    r_corraltion = compile.Corralation(h=h)
    metrics = [r_corraltion,]
    model.compile(optimizer=optimizer(learning_rate=lr_init),
                  loss=loss,
                  metrics=metrics)

    '''call backs'''
    if os.path.exists(ckpt_path + '.index'):
        print('**************** load model **************')
        model.load_weights(ckpt_path)

    ckpt = callBacks.checkpoint(ckpt_path,'val_loss',save_best=True,save_weight_only=True) #monitor sets as 'val_loss'
    callback = [ckpt]


    '''fit'''
    history = model.fit(data_train,epochs=epoch,
                        validation_data=data_val,
                        callbacks = callback)
    return history



def cross_validation_singleThreshold(data_dict,Model,
                                     epoch,batch,ckpt_head,lr,
                                     model_param = None,choose_fold = range(10),
                                     shuffle_or_not = False,shuffle_size = 256,
                                     SaveHistory = True,
                                     save_history_head = 'out/trainHistory/model'):
    '''
    单线程交叉验证
    :param data_dict: 数据集字典，{1:(data_train,data_val),...}
    :param Model: 模型类，用于创建每次交叉验证时的模型
    :param model_param: 模型实例创建所需关键字参数,若无则为None
    :param epoch: 每次验证迭代次数
    :param batch:
    :param ckpt_head: 存储模型的check point所需 '.../'---head:corss_val创建cross目录文件，head应给出model存储路径cross目录前路径
    :param lr: 验证初始学习率
    :param choose_fold:可迭代对象
    :param shuffle_or_not: 是否对数据集进行随机化
    :param shuffle_size: 随机化尺寸
    :return: 返回histories字典 history是模型训练情况
    '''
    histories = {}
    for i in choose_fold:
        print('{:*^50}'.format(' cross validation fold:%s ' % i))
        ckpt_path = ckpt_head + 'corss{}/model.ckpt'.format(i)
        print('save model path:{}'.format(ckpt_path))
        if model_param:model = Model(**model_param)
        else:model = Model()

        data_train,data_val = data_dict[str(i)]

        history = train(model,data_train,data_val,epoch,batch,lr,
                        shuffle_or_not=shuffle_or_not,shuffle_size=shuffle_size,
                        ckpt_path=ckpt_path)
        #time measure
        time_spend = round(train.time_spend)
        time_each = round(time_spend/epoch,2)
        print('{:*^100}\n'.format(' fold:%s spend time total:%ss each epoch:%ss '%(i,time_spend,time_each)))
        history.history['time_total'] = [time_spend]
        history.history['time_each'] = [time_each]
        histories[str(i)] = history

    #是否保存训练历史记录与实践
    if SaveHistory:
        save_history_head = save_history_head
        for key,history in histories.items():
            save_history_path = save_history_head.strip('/') + '/cross{}.txt'.format(key)
            dict_save(history.history,save_history_path)
            print('history save path:'.format(save_history_path))
            '''记录loss & metrics 每次迭代情况:
            loss 
            corralation_3 
            val_loss 
            val_corralation
            time_total 
            time_each 
            '''
    return histories



if __name__ == '__main__':
    '''test: go train'''
    import package.data_process.data_set as ds

    if os.name == 'nt':
        raw_path = r'E:\a_science\deppGBLUP\code_writer\data\1000_samples.raw'
        phen_path = r'E:\a_science\deppGBLUP\code_writer\data\1000_samples.phen'
    else:
        raw_path = '../data_jyzxProcess/out/input/snp.raw'
        phen_path = '../data_jyzxProcess/out/labels/label_100age_5460_10110011.phen'

    X, Y, X_pred, id_pred = ds.loadData(raw_path, phen_path)
    mean = tf.math.reduce_mean(Y)
    stddev = tf.math.reduce_std(Y)
    Y = (Y - mean) / stddev

    tf.debugging.set_log_device_placement(True)
    GPUs = tf.config.list_physical_devices("GPU")
    deviceName = '/GPU:0' if GPUs else '/CPU:0'
    print('{0:*^100}'.format(deviceName))
    data_train = tf.data.Dataset.from_tensor_slices((X[:720], Y[:720]))
    data_val = tf.data.Dataset.from_tensor_slices((X[720:], Y[720:]))
    import package.model.model as deepm
    param_dict = {'ymean':mean,'snp_num':X.shape[1]}
    model = deepm.DeepGblup(**param_dict)
    loss = train(model,data_train, data_val,epoch=5,batch=128,lr_init=0.001)
    tf.rank(X)
    model.summary()

