'''
mian
训练模型并保存，日志记录
调整数据集需要手动调整，其他超参数与选用模型设置可采
'''
import tensorflow as tf
import platform
import package.data_process.file_process as fp
import package.system_process.system_args as sy
import package.data_process.data_set as ds
import package.model.model as deepm
import package.train.train_struct as ts
'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ super param @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
sysargs = sy.getArgs()
choose_feature = ['100fat','100back','115fat','115back','test']
allModelName = ['a','all']
''' choose model '''
if platform.system() == 'Windows':sysargs['model'] = 'deepGblup'

epoch = 1 if 'epoch' not in sysargs.keys() else int(sysargs['epoch'])
batch = 128 if 'batch' not in sysargs.keys() else int(sysargs['batch'])
lr_init = 0.0001 if 'lr' not in sysargs.keys() else int(sysargs['lr'])

#data shuffle seed
seed = 10
shuffle_or_not = True
shuffle_size = 540
cross_fold = 10
choose_fold = [0,1,2,3,4,5,6,7,8,9] #10折->0:9 始终从0开始

'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ data import @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
#test
#data_file_dict:dict 确定每次模型训练文件有哪些
input_file = 'data/input/s0_50k_5701.raw'
label_file = 'data/label/la10110011_100age_5460.phen'


data_dict = {}
x_all,y_all,x_pre,id_pre = ds.loadData(input_file,label_file) #x_all,y_all  -> 尚未区分训练验证集；x_pre,id_pre->未知label的x与对应id
stddev = tf.math.reduce_std(y_all)
mean = tf.reduce_mean(y_all)
y_all = (y_all - mean) / stddev
in_feature = x_all.shape[-1]
dataSet = ds.createDataSet(x_all,y_all)
for i,(data_train,data_val) in enumerate(ds.get_cross_data(data=dataSet,fold_num=cross_fold)):
    data_dict['{}'.format(i)] = (data_train,data_val)
print('**************************** data process done! *****************************')


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ deepGblUP @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['deepGblup','deepgblup',*allModelName]:
    Model = deepm.model_all['deepGblup']
    model_param = {'ymean':mean,'snp_num':in_feature}
    model_name = 'deepGblup/'

    tmp = fp.getSnpLabel_mes(input_file,label_file) #获取snp与label文件信息以创建各类out的头目录
    ckpt_head = 'out/checkpoint/' + model_name + tmp
    save_history_head = 'out/train_history/' + model_name + tmp
    log = 'out/log/' + model_name + tmp

    histories = ts.cross_validation_singleThreshold(data_dict,Model,epoch,batch,
                                                    ckpt_head=ckpt_head,lr=lr_init,
                                                    model_param=model_param,choose_fold=choose_fold,
                                                    save_history_head=save_history_head)









