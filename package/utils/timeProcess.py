'''
时间处理
函数运行时间评估
'''
import time
class TimeMeasure:
    def __init__(self,func):
        self.func = func
        self.time_spend = None

    def __call__(self, *args, **kwargs):
        time_start = time.time()
        result = self.func(*args,**kwargs)
        time_end = time.time()
        self.time_spend = time_end - time_start
        return result


