'''
配置库
指定 loss
    metric
1.loss
训练损失为均方误差

2.metric

训练结果相关系数
'''
import tensorflow as tf

'''############ 指定loss为tensorflow的均方误差 ########'''
loss = tf.losses.MeanSquaredError

'''############ 自定义metric ##########'''
class Corralation(tf.keras.metrics.Metric):
    def __init__(self,name = 'corralation',h=1,gblup_hat = None):
        super(Corralation, self).__init__(name=name)
        self.r_cor = self.add_weight(name="corralation", initializer="zeros")
        #遗传力
        self.h = h
        self.gblup_hat = gblup_hat


    @staticmethod
    def pred_abilitiy(*args, h=1):
        '''
        args:([true,pred],[],...)  tensor or whatever
        return:[r1,r2,r3...]
        '''

        def pred_r(true, pred, h):
            import math
            pred_mean = tf.reduce_mean(pred)
            true_mean = tf.reduce_mean(true)
            f1 = tf.reduce_sum((pred - pred_mean) * (true - true_mean))
            f2 = tf.sqrt(tf.reduce_sum((pred - pred_mean) ** 2) * tf.reduce_sum((true - true_mean) ** 2))
            if f2 == 0:
                r = 0.
            else:
                cor = f1 / f2
                r = float(cor / math.sqrt(h))
            return r

        ability_r = []
        for i, data in enumerate(args):
            true = data[0]
            pred = data[1]
            r = pred_r(true, pred, h)
            ability_r.append(r)
        return ability_r

    def update_state(self,y_true,y_hat,sample_weight = None):
        self.r_cor.assign(self.pred_abilitiy([y_true,y_hat],h=self.h)[0])

    def result(self):
        return self.r_cor


    def reset_state(self):
        self.r_cor.assign(0.0)