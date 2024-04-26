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

