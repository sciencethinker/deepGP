'''
用于linux系统交互
'''
import sys
import tensorflow as tf
def getArgs():
    '''
    获取传入python文件的参数
    :return: 参数映射图谱
    '''
    args = sys.argv
    argMap = {}
    count = 1
    ifcontinue = False
    for i,arg in enumerate(args):
        # 跳过关键字参数
        if ifcontinue:
            ifcontinue = False
            continue
        if arg.startswith('--') or arg.startswith('-'):
            arg_name = arg.lstrip('-')
            argMap[arg_name] = args[i+1]
            count += 1
            ifcontinue = True
            continue
        argMap[str(count)] = arg
        count += 1

    print('***************** system args:{} *****************'.format(argMap))
    return argMap


def gpu_choose(chos):
    '''
    受限与当前框架条件，单个model只能在单个gpu上训练，通过整数选定指定gpu设备
    :chos : int
    '''

    gpus = tf.config.list_physical_devices('GPU') #返回gpu名称列表
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[chos], 'GPU')
            print('cunrent gpus list\n',*gpus,'\nchoose gpus \n',gpus[chos])
        except RuntimeError as e:
            print(e)
        except Exception as e:
            print(e)


