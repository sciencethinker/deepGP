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
nohup python main.tain --model <m> --epoch <e> --batch <b> --lr <lr> > train.log 2>&1 &
##########################################################################################

'''
import tensorflow as tf
import platform
import package.data_process.file_process as fp
import package.system_process.system_args as sy
import package.data_process.data_set as ds
import package.model.model as deepm
import package.train.train_class as tc
'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ super param @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
sysargs = sy.getArgs()
choose_feature = ['100fat','100back','115fat','115back','test']
allModelName = ['a','all']
''' choose model '''
if platform.system() == 'Windows':sysargs['model'] = 'sa0'

epoch = 1 if 'epoch' not in sysargs.keys() else int(sysargs['epoch'])
batch = 32 if 'batch' not in sysargs.keys() else int(sysargs['batch'])
lr_init = 0.0001 if 'lr' not in sysargs.keys() else int(sysargs['lr'])
batch_val = 256 if 'batch_val' not in sysargs.keys() else int(sysargs['batch_val'])

#data shuffle seed
seed = 10
shuffle_or_not = True
shuffle_size = 540
cross_fold = 10
choose_fold = [0,5] #10折->0:9 始终从0开始
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

'''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ChrAtten0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
if sysargs['model'] in ['ChrAtten0','chr0',*allModelName]:
    pass



#trainer
trainer = tc.Train()
trainer.set_data((x_all,y_all))
trainer.set_Model(Model)
trainer.cross_validation(param_model=model_param,ckpt_head=ckpt_head,fold_num=cross_fold,range_fold=choose_fold,
                         epoch=epoch,batch_t=batch,batch_v=batch_val,if_pred=True,if_saveHis=True,
                         save_history_head=save_history_head,save_log=log)





