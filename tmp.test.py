import matplotlib.pyplot as plt
file_name = 'data/input/s1_50k.map'
with open(file_name,'r') as file:
    chrome = {}
    for i,line in enumerate(file):
        pice = line.split('\t')
        if pice[0] not in chrome.keys():chrome[pice[0]] = []
        chrome[pice[0]] += [pice[1]]
count = 0
for i,key in enumerate(chrome.keys()):
    print('{0}:{1}'.format(key,len(chrome[key])))
    count += len(chrome[key])
print('total:{}'.format(count))





def draw_bar_chart(indicators, values):
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    # 绘制条形统计图
    plt.bar(indicators, values, color='skyblue')

    # 添加标题和标签
    plt.title('')
    plt.xlabel('chromosome name')
    plt.ylabel('num')

    # 自动调整 x 轴标签的角度，防止重叠
    plt.xticks(rotation=45, ha='right')

    # 显示图形
    plt.show()


# 示例输入数据
chrome_name = list(chrome.keys())
values = [len(chrome[key]) for key in chrome.keys()]

# 绘制条形统计图
draw_bar_chart(chrome_name, values)