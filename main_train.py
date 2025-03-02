'''
mian
训练模型并保存，日志记录
调整数据集需要手动调整，其他超参数与选用模型设置可采
###########################################################################################
linux使用说明
python main.train --model model_name --epoch e --batch batch_size --lr lr_rate
--model [a,all]全部模型均运行
--epoch
--batch
--lr
*** recommend use nohup command to train ***

nohup python main_train.py --model <m> --epoch <e> --batch <b> --lr <lr> --gpu <x> > train.log 2>&1 &

##########################################################################################

'''
import package.system_process.system_args as sy
import os
sysargs = sy.getArgs()
os.environ["CUDA_VISIBLE_DEVICES"] = sysargs['gpu']
import tensorflow as tf
import platform
import package.data_process.file_process as fp
import package.data_process.data_set as ds
import package.model.model as deepm
import package.train.train_struct as ts

'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ super param @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''

choose_feature = ['100fat','100back','115fat','115back','test']
allModelName = ['a','all']
''' choose model '''
if platform.system() == 'Windows':sysargs['model'] = 'sa0'
if 'gpu' in sysargs.keys():
    gpu = int(sysargs['gpu'])
    sy.gpu_choose(gpu)
epoch = 1 if 'epoch' not in sysargs.keys() else int(sysargs['epoch'])
batch = 32 if 'batch' not in sysargs.keys() else int(sysargs['batch'])
lr_init = 0.0001 if 'lr' not in sysargs.keys() else int(sysargs['lr'])
batch_val = 256 if 'batch_val' not in sysargs.keys() else int(sysargs['batch_val'])

#data shuffle seed
seed = 10
shuffle_or_not = True
shuffle_size = 540
cross_fold = 10
choose_fold = [0,1] #10折->0:9 始终从0开始
# choose_fold = [0,1,2,3,4,5,6,7,8,9]

'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ data import @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
#test
#data_file_dict:dict 确定每次模型训练文件有哪些
input_file = 'data/input/s1_50k_5224.raw'
label_file = 'data/label/laOrig_10fat_5021.phen'


data_dict = {}
x_all,y_all,x_pre,id_pre = ds.loadData(input_file,label_file) #x_all,y_all  -> 尚未区分训练验证集；x_pre,id_pre->未知label的x与对应id


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ snpAtten0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['SNPAtten0','sa0',*allModelName]:
    '''
    ####################################### data process ##############################################
    '''
    #scalar
    stddev = tf.math.reduce_std(y_all)
    mean = tf.reduce_mean(y_all)
    y_all = (y_all - mean) / stddev
    #add coloumn
    d_model = 5
    pick_num = 100
    seed = 42
    emb = deepm.Snp2Vec(depth=d_model)
    x_all = emb.random_pick(emb.add_coloumn(x_all),pick_num=pick_num,min_snp=0,max_snp=x_all.shape[1]+1,seed=seed)
    x_pre = emb.random_pick(emb.add_coloumn(x_pre),pick_num=pick_num,min_snp=0,max_snp=x_all.shape[1]+1,seed=seed)
    snp_num = x_all.shape[1] #获取snp数量
    dataSet = ds.createDataSet(x_all, y_all)
    for i, (data_train, data_val) in enumerate(ds.get_cross_data(data=dataSet, fold_num=cross_fold)):
        data_dict['{}'.format(i)] = (data_train, data_val)
    print('**************************** data process done! *****************************')

    #choose model & model param set
    Model = deepm.model_all['SNPAtten0']

    model_param = {'maxlen':snp_num,'d_model':d_model,
    'fp_units':[d_model * 3, d_model * 2, d_model, 1],'fp_acts':['relu', 'relu', 'relu', None],'fp_drop':0.2,
    'attention_units':d_model,'multi_head':8,'use_bais':True,
    'full_units':[d_model * 2, d_model],'full_act':['relu',None],
    'full_dropout_rates':[0.2,0.2],
    'attention_initializer':None,
    'pos_CONSTANT':10000,
    'bocks_num':8}
    model_name = 'snpAtten0_{0}s{1}k/'.format(seed,round(pick_num/1000,1))


    #got to train
    tmp = fp.getSnpLabel_mes(input_file,label_file) #获取snp与label文件信息以创建各类out的头目录
    ckpt_head = 'out/checkpoint/' + model_name + tmp
    save_history_head = 'out/train_history/' + model_name + tmp
    log = 'out/log/' + model_name + tmp

    histories = ts.cross_validation_singleThreshold(data_dict,Model,epoch,batch,validation_batch_size=batch_val,
                                                    ckpt_head=ckpt_head,lr=lr_init,
                                                    model_param=model_param,choose_fold=choose_fold,
                                                    save_history_head=save_history_head)


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Fnn_res1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['FNN_res1','fnn_res1','fn_res1','fn1',*allModelName]:
    '''
    ####################################### data process ##############################################
    '''
    #scalar
    stddev = tf.math.reduce_std(y_all)
    mean = tf.reduce_mean(y_all)
    y_all = (y_all - mean) / stddev
    dataSet = ds.createDataSet(x_all, y_all)
    for i, (data_train, data_val) in enumerate(ds.get_cross_data(data=dataSet, fold_num=cross_fold)):
        data_dict['{}'.format(i)] = (data_train, data_val)
    print('**************************** data process done! *****************************')

    #choose model & model param set
    model_name = 'FNN_res1/'
    Model = deepm.model_all['FNN_res1']
    model_param = {'blocks_arrange':[8192,*[None for _ in range(3)],6144,*[None for _ in range(3)],2048,*[None for _ in range(3)],
                                     2048,*[None for _ in range(3)],1024,*[None for _ in range(6)],
                                     512,*[None for _ in range(3)]],
                    'activation':'relu',
                    'dropout_rate':0.3,
                    'single_block_num':3,
                    'last_dense_units':1,
                    'last_dens_act':None}

    #got to train
    tmp = fp.getSnpLabel_mes(input_file,label_file) #获取snp与label文件信息以创建各类out的头目录
    ckpt_head = 'out/checkpoint/' + model_name + tmp
    save_history_head = 'out/train_history/' + model_name + tmp
    log = 'out/log/' + model_name + tmp

    histories = ts.cross_validation_singleThreshold(data_dict,Model,epoch,batch,validation_batch_size=batch_val,
                                                    ckpt_head=ckpt_head,lr=lr_init,
                                                    model_param=model_param,choose_fold=choose_fold,
                                                    save_history_head=save_history_head)

'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ vgg0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['vgg0','VGG0','Vgg0',*allModelName]:
    '''
    ####################################### data process ##############################################
    '''
    #scalar
    stddev = tf.math.reduce_std(y_all)
    mean = tf.reduce_mean(y_all)
    y_all = (y_all - mean) / stddev

    x_all = tf.expand_dims(x_all,2)
    dataSet = ds.createDataSet(x_all, y_all)
    for i, (data_train, data_val) in enumerate(ds.get_cross_data(data=dataSet, fold_num=cross_fold)):
        data_dict['{}'.format(i)] = (data_train, data_val)
    print('**************************** data process done! *****************************')

    #choose model & set model param & get model_name
    Model = deepm.model_all['VGG0']
    conv_param_list = [[64,3,1,],[128,3,1,],[256,3,1,'same'],[512,3,1,],[512,3,1,]]
    dropout_dense_rate = 0.2
    model_param = {'conv_param_list':conv_param_list,'dropout_dense_rate':dropout_dense_rate,'out_units':1,'out_act':None}
    model_name = 'vgg0/'

    #got to train
    tmp = fp.getSnpLabel_mes(input_file,label_file) #获取snp与label文件信息以创建各类out的头目录
    ckpt_head = 'out/checkpoint/' + model_name + tmp
    save_history_head = 'out/train_history/' + model_name + tmp
    log = 'out/log/' + model_name + tmp

    histories = ts.cross_validation_singleThreshold(data_dict,Model,epoch,batch,validation_batch_size=batch_val,
                                                    ckpt_head=ckpt_head,lr=lr_init,
                                                    model_param=model_param,choose_fold=choose_fold,
                                                    save_history_head=save_history_head)


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ chrAtten0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['Chratten0','ca0','chrAtten0','ChrAtten0',*allModelName]:
    '''
    ####################################### data process ##############################################
    '''
    #scalar
    stddev = tf.math.reduce_std(y_all)
    mean = tf.reduce_mean(y_all)
    y_all = (y_all - mean) / stddev

    dataSet = ds.createDataSet(x_all, y_all)
    for i, (data_train, data_val) in enumerate(ds.get_cross_data(data=dataSet, fold_num=cross_fold)):
        data_dict['{}'.format(i)] = (data_train, data_val)
    print('**************************** data process done! *****************************')

    #choose model & set model param & get model_name
    Model = deepm.model_all['ChrAtten0']
    # snp2chr_list: int型列表，给定不同染色体对应的snp位点数量对应chr之间的所属关系
    snp2chr_list = [8769,3484,2442,2153,2262,1811,2583,2176,2168,2368,1241,1383,1021,2643,2468,2159,1372,1094,981,2153]
    maxlen = 21
    conv_param_list = [[64,3,1,],[128,3,1,],[256,3,1],[256,3,1],[512,3,1,],[512,3,1],[512,3,1]]
    chr_emb_units = 512
    #4层全连接预测层
    fp_units = [chr_emb_units,int(chr_emb_units*0.8),int(chr_emb_units*0.8),1]
    fp_drop = 0.2
    fp_acts = ['relu','relu','relu',None]
    #self_attention units & heads
    heads = 8
    atten_units = chr_emb_units
    full_units = [int(chr_emb_units*0.8),chr_emb_units]

    dropout_dense_rate = 0.2
    model_param = {'conv_param_list':conv_param_list,
                 'snp2chr_list':snp2chr_list,'chr_emb_units':chr_emb_units,
                 'maxlen':maxlen,
                 'fp_units':fp_units,'fp_acts':fp_acts,'fp_drop':fp_drop,
                 'atten_units':atten_units,'multi_head':heads,'use_bais':True,
                 'full_units':full_units,'full_act':['relu',None],
                 'full_dropout_rates':[0.2,0.2],
                 'attention_initializer':None,
                 'pos_CONSTANT':10000,
                 'blocks_num':8}
    model_name = 'ChrAtten0/'

    #got to train
    tmp = fp.getSnpLabel_mes(input_file,label_file) #获取snp与label文件信息以创建各类out的头目录
    ckpt_head = 'out/checkpoint/' + model_name + tmp
    save_history_head = 'out/train_history/' + model_name + tmp
    log = 'out/log/' + model_name + tmp

    histories = ts.cross_validation_singleThreshold(data_dict,Model,epoch,batch,validation_batch_size=batch_val,
                                                    ckpt_head=ckpt_head,lr=lr_init,
                                                    model_param=model_param,choose_fold=choose_fold,
                                                    save_history_head=save_history_head)





'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ deepGblUP @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['deepGblup','deepgblup',*allModelName]:
    '''
    ####################################### data process ##############################################
    '''
    stddev = tf.math.reduce_std(y_all)
    mean = tf.reduce_mean(y_all)
    y_all = (y_all - mean) / stddev
    in_feature = x_all.shape[-1]
    dataSet = ds.createDataSet(x_all, y_all)
    for i, (data_train, data_val) in enumerate(ds.get_cross_data(data=dataSet, fold_num=cross_fold)):
        data_dict['{}'.format(i)] = (data_train, data_val)
    print('**************************** data process done! *****************************')

    #choose model & set model param & get model_name
    Model = deepm.model_all['deepGblup']
    model_param = {'ymean':mean,'snp_num':in_feature}
    model_name = 'deepGblup/'

    #got to train
    tmp = fp.getSnpLabel_mes(input_file,label_file) #获取snp与label文件信息以创建各类out的头目录
    ckpt_head = 'out/checkpoint/' + model_name + tmp
    save_history_head = 'out/train_history/' + model_name + tmp
    log = 'out/log/' + model_name + tmp

    histories = ts.cross_validation_singleThreshold(data_dict,Model,epoch,batch,validation_batch_size=batch_val,
                                                    ckpt_head=ckpt_head,lr=lr_init,
                                                    model_param=model_param,choose_fold=choose_fold,
                                                    save_history_head=save_history_head)

