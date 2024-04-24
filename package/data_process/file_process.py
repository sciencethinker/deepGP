'''
处理文件名
由于ckpt,history等存储方式存在异同性，因此统一处理文件名称以便于传入cross函数
'''

def getSnpLabel_mes(snpFile,labelFile):
    '''
    用于获取整合snp与label
    :param snpFile:str snp路径
    :param labelFile:str label路径
    :return: str 整合信息
    em:a = '../data_jyzxProcess/data/data/1000_samples.raw'
       b = '../data_jyzxProcess/data/data/1000_samples.phen'
       getSnpLabel_mes(a,b)
    >> 1000_1000_samples
    '''
    if ('\\' in snpFile and '/' in snpFile) or ('\\' in labelFile and '/' in labelFile):
        raise Exception("不要发癫，文件路径应该只含有'/'or'\\'")

    #获取尾缀信息 list
    split = lambda x,y:x.split(y)[-1].split('.')[0].split('_')
    snpMes = split(snpFile,'\\') if '\\' in snpFile else split(snpFile,'/')
    labelMes = split(labelFile,'\\') if '\\' in labelFile else split(labelFile,'/')

    if snpMes[-1] == labelMes[-1]:
        allMes = '_'.join(snpMes[:-1]+labelMes)
    else:allMes = '_'.join(snpMes + labelMes)
    return allMes + '/'
if __name__ == '__main__':
    '''
    提供文件路径时不要发癫，一会儿/，一会儿//   (╯▔皿▔)╯...
    '''
    a = 'data/input/s1_50k_5224.raw'
    b = 'data/label/laOrig_10age_5021.phen'
    print(getSnpLabel_mes(a,b))



