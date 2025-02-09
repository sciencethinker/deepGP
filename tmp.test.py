def my_generator():
    print("Start")
    yield 1
    print("Middle")
    yield 2
    print("End")


# 创建生成器对象
gen = my_generator()

# 使用 next() 获取值
print(next(gen))  # 输出：Start，然后返回 1
print(next(gen))  # 输出：Middle，然后返回 2
# 再次调用 next() 会触发 StopIteration 异常，因为生成器已经结束
# print(next(gen))  # 抛出 StopIteration 异常

# 使用 for 循环遍历生成器
for value in my_generator():
    print(value)  # 输出：Start，1，Middle，2，End