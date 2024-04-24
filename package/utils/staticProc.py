'''
统计API
构建一些统计学函数
corralation coffecient
covarience
相关系数，
方差
均值等

'''
import tensorflow as tf
import numpy as np
def corr_matrix_np(*x):
    '''
    使用numpy计算相关系数矩阵
    :param x:
    :param label:
    :return: cor matrix[n,n]
    '''
    cor = np.corrcoef(*x)
    return cor

def corr_np(x,label):
    '''
    使用numpy计算单一值
    :return:cor scalar value
    '''
    cov = np.mean((x - np.mean(x))*(label - np.mean(label)))
    std_x = np.std(x)
    std_label = np.std(label)
    cor = cov/(std_x*std_label)
    return cor

def corr_tf(x,label):
    '''
    使用tensorflow计算相关系数单值
    :param x:
    :param label:
    :return:
    '''
    cov = tf.reduce_mean((x - tf.reduce_mean(x))*(label - tf.reduce_mean(x)))
    std_x = tf.math.reduce_std(x)
    std_lable = tf.math.reduce_std(label)
    cor = cov/(std_x*std_lable)
    return cor

def loss_mean_std_tf(x,y):
    '''
    使用tensorflow 对x,y之间的loss求取均值与标准差
    :param x:
    :param y:
    :return:
    '''
    ''' !!!!!!!!!!这里将loss取了绝对值!!!!!!!!!!!!!'''
    loss = tf.math.abs(x - y)
    loss_mean = tf.reduce_mean(loss)
    loss_std = tf.math.reduce_std(loss)
    return (loss_mean,loss_std)




if __name__ == '__main__':
    #1:test:--> corr_matrix_np
    print('1:test:--> corr_matrix_np')
    x = np.array([1,2,3,4,5])
    label = np.array([2,3,6,7,8])
    print(corr_matrix_np(x,label))

    #2:test:--> corr_np
    print('#2:test:--> corr_np')
    x = np.array([1,2,3,4,5])
    label = np.array([2,3,6,7,8])
    print(corr_np(x,label))

    #3:test:--> corr_tf
    # x = np.array([1,2,3,4,5])
    # label = np.array([2,3,6,7,8])
    print('#3:test:--> corr_tf')
    x = tf.constant([1,2,3,4,5],dtype=tf.float32)
    label = tf.constant([2,3,6,7,8],dtype=tf.float32)
    print(corr_tf(x,label))

    #4:test:-->loss_mean_std_tf
    print('#4:test:-->loss_mean_std_tf')
    x = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
    label = tf.constant([2, 3, 6, 7, 8], dtype=tf.float32)
    print(loss_mean_std_tf(x,label))


