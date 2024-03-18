'''
!!!!!!!!!!!!!!!!!!! 某个地方有问题 但不知道是哪里有问题 ---2023.11.28
'''
import tensorflow as tf
import math
import numpy as np


def make_G_D_E(X, invers=True):
    # Calculate allele frequencies
    n, k = X.shape
    pi = tf.reduce_sum(X, axis=0) / (2 * n)
    P = tf.expand_dims(pi, axis=0)

    # Create a diagonal formation
    A = tf.eye(len(X), dtype=tf.float32)

    # Create G (additive relationship matrix)
    Z = X - 2 * P
    G = tf.matmul(Z, tf.transpose(Z)) / (2 * tf.reduce_sum(pi * (1 - pi)))

    # Create D (dominance relationship matrix)
    print('Creating dominance matrix')
    #create a copy of X
    # W = tf.identity(X)
    W = np.array(X)
    for j in range(W.shape[1]):
        W_j = W[:, j]
        W_j = tf.where(W_j == 0, -2 * pi[j]**2, W_j)
        W_j = tf.where(W_j == 1, 2 * pi[j] * (1 - pi[j]), W_j)
        W_j = tf.where(W_j == 2, -2 * (1 - pi[j])**2, W_j)
        W[:,j] = W_j
    W = tf.constant(W,dtype=tf.float32)
    print('Done !')
    D = tf.matmul(W, tf.transpose(W)) / (tf.reduce_sum((2 * pi * (1 - pi))**2))

    # Create E (epistasis relationship matrix)
    print('Creating interaction marker')
    M = X - 1
    E = 0.5 * ((tf.matmul(M, tf.transpose(M)) * tf.matmul(M, tf.transpose(M))) -
               0.5 * (tf.matmul(M * M, tf.transpose(M * M))))
    E = E / (tf.linalg.trace(E) / n)
    print('Done!')
    # Rescale matrices with dummy A

    G = 0.99 * G + 0.01 * A
    D = 0.99 * D + 0.01 * A
    E = 0.99 * E + 0.01 * A

    if invers:
        return tf.linalg.inv(G), tf.linalg.inv(D), tf.linalg.inv(E)
    return G, D, E

def mixed_model(Z,A,lamda,y):
    y = y - tf.reduce_mean(y)
    Z_trans = tf.transpose(Z)
    A_inv = tf.linalg.inv(A)
    T = Z_trans @ Z + lamda * A_inv
    T_inv = tf.linalg.inv(T)
    b = T_inv @ Z_trans @ y
    return b

def call_Z(ref_len,whole_len):
    '''return: Z.shape = [n_train,n] n>=n_train
    like:Z =
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
    '''
    Z = np.zeros([ref_len,whole_len])
    for i in range(ref_len):Z[i,i] = 1
    Z = tf.constant(Z,dtype=tf.float32)
    return Z

class GBLUP:
    def __init__(self):
        pass
    def predict(self,X):
        pass

    def make_G_D_E(self,X, invers=True):
        # Calculate allele frequencies
        n, k = X.shape
        pi = tf.reduce_sum(X, axis=0) / (2 * n)
        P = tf.expand_dims(pi, axis=0)

        # Create a diagonal formation
        A = tf.eye(len(X), dtype=tf.float32)

        # Create G (additive relationship matrix)
        Z = X - 2 * P
        G = tf.matmul(Z, tf.transpose(Z)) / (2 * tf.reduce_sum(pi * (1 - pi)))

        # Create D (dominance relationship matrix)
        print('Creating dominance matrix')
        # create a copy of X
        # W = tf.identity(X)
        W = np.array(X)

        for j in range(W.shape[1]):
            W_j = W[:, j]
            W_j = tf.where(W_j == 0, -2 * pi[j] ** 2, W_j)
            W_j = tf.where(W_j == 1, 2 * pi[j] * (1 - pi[j]), W_j)
            W_j = tf.where(W_j == 2, -2 * (1 - pi + [j]) ** 2, W_j)
            W[:, j] = W_j

        W = tf.constant(W, dtype=tf.float32)

        D = tf.matmul(W, tf.transpose(W)) / (tf.reduce_sum((2 * pi * (1 - pi)) ** 2))

        # Create E (epistasis relationship matrix)
        print('Creating interaction marker')
        M = X - 1
        E = 0.5 * ((tf.matmul(M, tf.transpose(M)) * tf.matmul(M, tf.transpose(M))) -
                   0.5 * (tf.matmul(M * M, tf.transpose(M * M))))
        E = E / (tf.linalg.trace(E) / n)

        # Rescale matrices with dummy A

        G = 0.99 * G + 0.01 * A
        D = 0.99 * D + 0.01 * A
        E = 0.99 * E + 0.01 * A

        if invers:
            return tf.linalg.inv(G), tf.linalg.inv(D), tf.linalg.inv(E)
        return G, D, E

    def mixed_model(self,Z, A, lamda, y):
        y = y - tf.reduce_mean(y)
        Z_trans = tf.transpose(Z)
        A_inv = tf.linalg.inv(A)
        T = Z_trans @ Z + lamda * A_inv
        T_inv = tf.linalg.inv(T)
        b = T_inv @ Z_trans @ y
        return b

    def call_Z(self,ref_len, whole_len):
        '''return: Z.shape = [n_train,n] n>=n_train
        like:Z =
           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
        '''
        Z = np.zeros([ref_len, whole_len])
        for i in range(ref_len): Z[i, i] = 1
        Z = tf.constant(Z, dtype=tf.float32)
        return Z




'''new'''
def pred_abilitiy(*args,h = 1):
    '''
    args:([true,pred],[],...)  tensor or whatever
    return:[r1,r2,r3...]
    '''

    def pred_r(true,pred,h):
        pred_mean = tf.reduce_mean(pred)
        true_mean = tf.reduce_mean(true)
        f1 = tf.reduce_sum((pred - pred_mean) * (true - true_mean))
        f2 = tf.sqrt(tf.reduce_sum((pred - pred_mean) ** 2) * tf.reduce_sum((true - true_mean) ** 2))
        if f2 == 0:
            r = 0
        else:
            cor = f1 / f2
            r = float(cor / math.sqrt(h))
        return r

    ability_r = []
    for i,data in enumerate(args):
        true = data[0]
        pred = data[1]
        r = pred_r(true,pred,h)
        ability_r.append(r)
    return ability_r

'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ test @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
# Example usage:
if __name__ == '__main__':
    raw_path = r'E:\a_science\deppGBLUP\code_writer\data\1000_samples.raw'
    phen_path = r'E:\a_science\deppGBLUP\code_writer\data\1000_samples.phen'
    import package.data_process.data_set as dataSet
    X, Y, X_pred, id_pred = dataSet.loadData(raw_path, phen_path)
    mean = tf.math.reduce_mean(Y)
    stddev = tf.math.reduce_std(Y)
    Y = (Y - mean) / stddev

    data = {}
    data['X'] = X[:800]
    data['Y'] = Y[:800]
    data['X_val'] = X[800:]
    data['Y_val'] = Y[800:]
    X = tf.concat([data['X'],data['X_val']], axis=0)
    h2 = 0.37
    glamb = (1 - h2) / h2
    dlamb = elamb = (1 - h2 * 0.1) / (h2 * 0.1)
    G, D, E = make_G_D_E(X)
    Z = call_Z(len(data['X']), len(data['X']) + len(data['X_val']))

    a = mixed_model(Z, G, glamb,data['Y'])
    d = mixed_model(Z, D, dlamb,data['Y'])
    e = mixed_model(Z, E, elamb,data['Y'])


    print(pred_abilitiy([data['Y'],(a+d+e)[:800]]))
    '''
    0.9226402044296265
    0.9527196884155273 2024.2.27
    '''
    print(pred_abilitiy([data['Y_val'], (a + d + e)[800:]]))
    '''
    0.1828545778989792
    -0.21052186191082 2024.2.27
    '''




