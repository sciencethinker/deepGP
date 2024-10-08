import  os
def log(self, path, way='a'):
    '''

    :param path:
    :param way: 记录在日志中的方式，默认为追加至文件末尾'a',-----'c','r'
    :return:
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)


    with open(path, way) as file:
        if way == 'a':  # 追加内容前换行
            file.write('\n')
        values = []
        for key, value in self.predMes.items():
            if key == 'result': continue
            values.append(str(value) + '\t')
        file.writelines(values)
    print('write log done as \'{0}\' ! log at:{1}'.format(way, path))


