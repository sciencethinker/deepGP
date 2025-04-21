'''
mian
训练模型并保存，日志记录
需要手动调整数据集，其他超参数与选用模型设置可采
###########################################################################################
linux使用说明
python main.train --model model_name --epoch e --batch batch_size --lr lr_rate
--model [a,all]全部模型均运行
--epoch
--batch
--lr
*** recommend use nohup command to train ***
nohup python main_train_new.py --model <m> --epoch <e> --batch <b> --lr <lr> > train.log 2>&1 &
nohup python main_train_new.py --model <m> --epoch <e> --batch <b> --lr <lr> --gpu <x> --label <0/1/2/.../7> --cf <[1,2,3]>> train.log 2>&1 &

##########################################################################################

'''

import platform
import package.system_process.system_args as sy
sysargs = sy.getArgs()
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = sysargs['gpu']
os.environ["CUDA_VISIBLE_DEVICES"] = '0' if 'gpu' not in sysargs.keys() else sysargs['gpu']
import package.data_process.file_process as fp
import package.data_process.data_set as ds
import package.model.model as deepm
import package.train.train_class as tc
import tensorflow as tf

'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ super param @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''

'''
gpu调用调试
同一脚本可以调用不同指定gpu
'''





choose_feature = ['100fat','100back','115fat','115back','test']
allModelName = ['a','all']
''' choose model '''
if platform.system() == 'Windows':sysargs['model'] = 'vgg0'

epoch = 3 if 'epoch' not in sysargs.keys() else int(sysargs['epoch'])
batch = 32 if 'batch' not in sysargs.keys() else int(sysargs['batch'])
lr_init = 0.01 if 'lr' not in sysargs.keys() else float(sysargs['lr'])
batch_val = 256 if 'batch_val' not in sysargs.keys() else int(sysargs['batch_val'])

#data shuffle seed
seed = 10
shuffle_or_not = True
shuffle_size = 540
cross_fold = 10
choose_crosses = None if 'cf' not in sysargs.keys() else [int(i) for i in str.split(str.strip(sysargs['cf'],'[]'),',')]
choose_fold = choose_crosses if choose_crosses else [0,1,2,3,4,5,6,7,8,9]  #10折->0:9 始终从0开始

# choose_fold = [9]

'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ data import @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
#test
#data_file_dict:dict 确定每次模型训练文件有哪些
input_file = 'data/input/s1_50k_5224.raw'


label_dict = {
    '0':'data/label/laOrig_10fat_5021.phen',
    '1':'data/label/laOrig_15fat_5021.phen',
    '2':'data/label/laOrig_10age_5021.phen',
    '3':'data/label/laOrig_15age_5021.phen',
    '4':'data/label/la13478_10fat_5460.phen',
    '5':'data/label/la13478_15fat_5460_.phen',
    '6':'data/label/la13478_10age_5460.phen',
    '7':'data/label/la13478_15age_5460.phen'
              }

label_file_name = None if 'label' not in sysargs.keys() else sysargs['label']
label_file = label_dict[label_file_name] if label_file_name else 'data/label/laOrig_10fat_5021.phen'

# label_file = 'data/label/laOrig_10fat_5021.phen'
# label_file = 'data/label/laOrig_15fat_5021.phen'
# label_file = 'data/label/laOrig_10age_5021.phen'
# label_file = 'data/label/laOrig_15age_5021.phen'
# label_file = 'data/label/la13478_10fat_5460.phen'
# label_file = 'data/label/la13478_15fat_5460_.phen'
# label_file = 'data/label/la13478_10age_5460.phen'
# label_file = 'data/label/la13478_15age_5460.phen'
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
    pick_num = 50
    seed = 42
    emb = deepm.Snp2Vec(depth=d_model)
    x_all = emb.random_pick(emb.add_coloumn(x_all),pick_num=pick_num,min_snp=0,max_snp=x_all.shape[1]+1,seed=seed)
    x_pre = emb.random_pick(emb.add_coloumn(x_pre),pick_num=pick_num,min_snp=0,max_snp=x_all.shape[1]+1,seed=seed)
    snp_num = x_all.shape[1] #获取snp数量

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

'''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ChrAtten0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['ChrAtten0','chr0','ca0',*allModelName]:
    # scalar
    # stddev = tf.math.reduce_std(y_all)
    # mean = tf.reduce_mean(y_all)
    # y_all = (y_all - mean) / stddev

    # choose model & set model param & get model_name
    Model = deepm.model_all['ChrAtten0']
    # snp2chr_list: int型列表，给定不同染色体对应的snp位点数量对应chr之间的所属关系
    # 注意snp位点不可以改变！若不使用s1基因序列，则需要重新调试！
    snp2chr_list = [8769, 3484, 2442, 2153, 2262, 1811, 2583, 2176, 2168, 2368, 1241, 1383, 1021, 2643, 2468, 2159,
                    1372, 1094, 981, 2153]
    maxlen = 21
    conv_param_list = [[64, 3, 1, ], [128, 3, 1, ], [256, 3, 1], [256, 3, 1], [512, 3, 1, ], [512, 3, 1], [512, 3, 1]]
    chr_emb_units = 512
    # 4层全连接预测层
    fp_units = [chr_emb_units, int(chr_emb_units * 0.8), int(chr_emb_units * 0.8), 1]
    fp_drop = 0.4
    fp_acts = ['relu', 'relu', 'relu', None]
    # self_attention units & heads
    heads = 8
    atten_units = chr_emb_units
    full_units = [int(chr_emb_units * 0.8), chr_emb_units]

    dropout_dense_rate = 0.35
    model_param = {'conv_param_list': conv_param_list,
                   'snp2chr_list': snp2chr_list, 'chr_emb_units': chr_emb_units,
                   'maxlen': maxlen,
                   'fp_units': fp_units, 'fp_acts': fp_acts, 'fp_drop': fp_drop,
                   'atten_units': atten_units, 'multi_head': heads, 'use_bais': True,
                   'full_units': full_units, 'full_act': ['relu', None],
                   'full_dropout_rates': [0.3, 0.3],
                   'attention_initializer': None,
                   'pos_CONSTANT': 10000,
                   'blocks_num': 8}
    # model_name = 'ChrAtten0/'
    model_name = 'ChrAtten0_unstd/'

'''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ChrAtten1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['ChrAtten1','chr1','ca1',*allModelName]:
    # scalar
    # stddev = tf.math.reduce_std(y_all)
    # mean = tf.reduce_mean(y_all)
    # y_all = (y_all - mean) / stddev
    # model_name = 'ChrAtten1/'
    model_name = 'chrAtten1_unstd/'

    # choose model & set model param & get model_name
    Model = deepm.model_all['ChrAtten1']
    # snp2chr_list: int型列表，给定不同染色体对应的snp位点数量对应chr之间的所属关系
    # 注意snp位点不可以改变！若不使用s1基因序列，则需要重新调试！
    snp2chr_list = [8769, 3484, 2442, 2153, 2262, 1811, 2583, 2176, 2168, 2368, 1241, 1383, 1021, 2643, 2468, 2159,
                    1372, 1094, 981, 2153]
    maxlen = 21
    conv_param_list = [[32, 3, 1,'same','relu',0.4 ], [64, 3, 1,'same','relu',0.4],
                       [128, 3, 1,'same','relu',0.4], [256, 3, 1,'same','relu',0.4],
                       [512, 3, 1,'same','relu',0.4 ], [512, 3, 1,'same','relu',0.4],
                       [1024, 3, 1,'same','relu',0.4]]
    chr_emb_units = 1024


    # self_attention units & heads
    heads = 8
    atten_units = chr_emb_units
    full_units = [int(chr_emb_units * 0.8), chr_emb_units]
    blocks_num = 9 #

    # 4层全连接预测层
    fp_units = [chr_emb_units, int(chr_emb_units * 0.8), int(chr_emb_units * 0.8), 1]
    fp_acts = ['relu', 'relu', 'relu', None]


    #dropout
    dropout_chrEmb =  0.4  #编码时的全连接层与卷积结构dropout
    dropout_atten =  [0.2, 0.2]  #self_attention的全连接层dropout
    fp_drop = 0.4  #末尾全连接层的dropout



    model_param = {'conv_param_list': conv_param_list,
                   'snp2chr_list': snp2chr_list, 'chr_emb_units': chr_emb_units,'dropout_chrEmb':dropout_chrEmb,
                   'maxlen': maxlen,
                   'fp_units': fp_units, 'fp_acts': fp_acts, 'fp_drop': fp_drop,
                   'atten_units': atten_units, 'multi_head': heads, 'use_bais': True,
                   'full_units': full_units, 'full_act': ['relu', None],
                   'full_dropout_rates':dropout_atten,
                   'attention_initializer': None,
                   'pos_CONSTANT': 10000,
                   'blocks_num': blocks_num}




'''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ VGG0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['VGG0','vgg0',*allModelName]:
    # scalar
    stddev = tf.math.reduce_std(y_all)
    mean = tf.reduce_mean(y_all)
    y_all = (y_all - mean) / stddev
    x_all = tf.expand_dims(x_all,2) #[n,m] -> [n,m,1]

    # choose model & set model param & get model_name
    Model = deepm.model_all['VGG0']
    conv_param_list = [[64, 3, 1, ], [128, 3, 1, ], [256, 3, 1, 'same'], [512, 3, 1, ], [512, 3, 1, ]]
    # dropout_dense_rate = 0.2
    #降低过拟合
    dropout_dense_rate = 0.6
    model_param = {'conv_param_list': conv_param_list, 'dropout_dense_rate': dropout_dense_rate, 'out_units': 1,
                   'out_act': None}
    model_name = 'vgg0/'

# got to train
tmp = fp.getSnpLabel_mes(input_file, label_file)  # 获取snp与label文件信息以创建各类out的头目录
ckpt_head = 'out/checkpoint/' + model_name + tmp
save_history_head = 'out/train_history/' + model_name + tmp
log = 'out/log/' + model_name + tmp + 'a.log'

#trainer
trainer = tc.Train()
trainer.set_data((x_all,y_all))
trainer.set_Model(Model)
trainer.cross_validation(param_model=model_param,ckpt_head=ckpt_head,fold_num=cross_fold,range_fold=choose_fold,
                         epoch=epoch,batch_t=batch,batch_v=batch_val,if_pred=True,if_saveHis=True,
                         save_history_head=save_history_head,save_log=log)











