'''
用于linux系统交互
'''
import sys

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




