'''
检验model水平
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
if platform.system() == 'Windows':sysargs['model'] = 'a'
ifLodedata = False if 'lode_data' not in sysargs.keys() else (int(sysargs['lode_data'])) #0 or 1
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
# choose_fold = choose_crosses if choose_crosses else [0,1,2,3,4,5,6,7,8,9]  #10折->0:9 始终从0开始
choose_fold = [9]
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
data_dict = {}

if ifLodedata:
    x_all,y_all,x_pre,id_pre = ds.loadData(input_file,label_file) #x_all,y_all  -> 尚未区分训练验证集；x_pre,id_pre->未知label的x与对应id

model_dict = {}
ckpt_dict = {}
'''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ChrAtten0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['ChrAtten0','chr0','ca0',*allModelName]:
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
    model_name = 'ChrAtten0_unstd/'

    tmp = fp.getSnpLabel_mes(input_file, label_file)  # 获取snp与label文件信息以创建各类out的头目录
    ckpt_head = 'out/checkpoint/' + model_name + tmp
    save_history_head = 'out/train_history/' + model_name + tmp
    log = 'out/log/' + model_name + tmp + 'a.log'

    model_dict[model_name] = [Model,model_param]
    ckpt_dict[model_name] = ckpt_head

'''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ VGG0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['VGG0', 'vgg0', *allModelName]:
        # scalar
        stddev = tf.math.reduce_std(y_all)
        mean = tf.reduce_mean(y_all)
        y_all = (y_all - mean) / stddev
        x_all = tf.expand_dims(x_all, 2)  # [n,m] -> [n,m,1]

        # choose model & set model param & get model_name
        Model = deepm.model_all['VGG0']
        conv_param_list = [[64, 3, 1, ], [128, 3, 1, ], [256, 3, 1, 'same'], [512, 3, 1, ], [512, 3, 1, ]]
        # dropout_dense_rate = 0.2
        # 降低过拟合
        dropout_dense_rate = 0.6
        model_param = {'conv_param_list': conv_param_list, 'dropout_dense_rate': dropout_dense_rate, 'out_units': 1,
                       'out_act': None}
        model_name = 'vgg0/'

        tmp = fp.getSnpLabel_mes(input_file, label_file)  # 获取snp与label文件信息以创建各类out的头目录
        ckpt_head = 'out/checkpoint/' + model_name + tmp
        save_history_head = 'out/train_history/' + model_name + tmp
        log = 'out/log/' + model_name + tmp + 'a.log'

        model_dict[model_name] = [Model,model_param]





# 定义模型结构（必须与保存时的结构相同）

for name in model_dict.keys():
    model = None
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {}summary @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n'.format(name) * 5)
    model,param = model_dict[name]  # 替换为你的模型类
    model = model(**param) if param!=None else model()
    optimizer = tf.keras.optimizers.Adam()  # 替换为你使用的优化器


    model.summary()
    ckpt = ckpt_dict[name] + 'corss{}/model.ckpt'.format(0)
    # 创建Checkpoint对象
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # 从指定路径恢复模型
    checkpoint.restore(ckpt).expect_partial()

    model.summary()



    # 现在可以使用恢复的模型了

