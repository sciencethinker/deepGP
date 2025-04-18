'''
记录已设计model
deepGBLUP
'''
model_all = {}
#1.deepGBLUP
from package.model.deepGBLUP import DeepGblup
model_all['deepGblup'] = DeepGblup

#2.snpAtten0
from package.model.self_attention0 import SNPAtten0
model_all['SNPAtten0'] = SNPAtten0

#3.snpEmbedding
from package.model.snp_embedding import Snp2Vec
model_all['Snp2Vec'] = Snp2Vec

#4.
from package.model.snp_embedding import ChrEmbed
model_all['ChrEmbed'] = ChrEmbed

#5.
from package.model.self_attention0 import ChrAtten0
model_all['ChrAtten0'] = ChrAtten0

#6.ResFNN1
from package.model.fnn import FNN_res1
model_all['FNN_res1'] = FNN_res1

#7.vgg16
from package.model.cnn0 import VGG0
model_all['VGG0'] = VGG0

#8.
from package.model.self_attention0 import ChrAtten1
model_all['ChrAtten1'] = ChrAtten1


