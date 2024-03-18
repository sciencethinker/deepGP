'''
数据集设置

数据集创建方式：
1.从训练数据集与标签集文件提取数据
2.在本地文件data中创建数据集文件用于Dataset
3.

'''
import tqdm
import tensorflow as tf
class SNPdata():
    pass



def loadData(raw_path,phen_path):
    '''
    依据文件路径获取数据
    :param raw_path:
    :param phen_path:
    :return:
    X_tensor-已知X,Y_tensor已知X对应label
    ,X_tensor_pre-未知label对应的X,id_pred未知label对应的id
    '''
    #获取标签字典{phen_id:value}
    with open(phen_path,'r') as phenFile:

        #get phen dict
        phen_dict = {}
        train_ids = []
        for line in tqdm.tqdm(phenFile,desc='Label data load:'):
            line_ = line.split()
            phen_dict[line_[0]] = float(line_[1])
            train_ids.append(line_[0])

        #get raw list(X,Y for train;X,Y for predict)

        start = 6 #数据起始点
        X = []
        Y = []
        X_pred = []
        id_pred = []

        # Load snp and Rearrange phen data
        raw_file = open(raw_path)  # SNP
        if True:
            SNPs = next(raw_file).split()[start:]
        for line in tqdm.tqdm(raw_file,desc='Input data load:'):
            line_ = line.split()

            #judge PID in train_ids
            if line_[1] in train_ids:
                X.append(line_[start:])
                Y.append(phen_dict[line_[1]])
            else:
                X_pred.append(line_[start:])
                id_pred.append(line_[1])

        X_tensor = tf.strings.to_number(tf.constant(X))
        Y_tensor = tf.constant(Y)
        Y_tensor = tf.reshape(Y_tensor,[Y_tensor.shape[0],1])
        X_tensor_pre = tf.strings.to_number(tf.constant(X_pred))
    return X_tensor,Y_tensor,X_tensor_pre,id_pred


        #获取snp名称，即rawFile表头

    pass

#交叉数据集处理
def get_cross_data(data,fold_num):
    '''
    获取交叉数据集，通过获取dataSet类与交叉折数，获得交叉验证数据集生成器
    :param data:
    :param fold_num:
    :return: yyield
    '''
    len_sam = len([i for i in data])
    fold_sam = len_sam // fold_num
    for i in range(fold_num):
        start = i*fold_sam
        end = start + fold_sam
        data_val = data.skip(start).take(fold_sam)
        data_train = data.take(start).concatenate(data.skip(end))

        yield (data_train,data_val)

def createDataSet(x,y,name=None):
    dataSet = tf.data.Dataset.from_tensor_slices((x,y),name=name)
    return dataSet

def getStd(label, fold):
    '''
    未完成
    label标准化
    :param label:
    :param fold: int 交叉验证折数 or 可迭代对象(返回整数)
    :return: label_std & 标准字典{'i':[(std,mean),(std,mean)]}
    '''

    assert fold is int ,"fold must be a int !,but fold is {}".format(fold)
    label_len = label.shape(0)
    fold_len = label_len // fold

    for k in range(fold):
        start = fold_len*k
        end = start + fold_len
        label_val = label[start:end]
        label_train = tf.concat(label[0:start],label_val[end:])

        #标准化


def saveData():
    pass

def shuffleData(data,seed):
    '''
    :param data: tf.data.DataSet
    :param seed:
    :return:
    '''


