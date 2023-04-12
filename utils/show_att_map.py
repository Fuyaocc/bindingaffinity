import numpy as np
import matplotlib.pyplot as plt


def generate_att_map(attention_map):

    # 可视化注意力矩阵为热力图
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(attention_map, cmap=plt.cm.Blues)

    # 设置x轴和y轴标签
    ax.set_xticks(np.arange(attention_map.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(attention_map.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    # 添加x轴和y轴标签
    ax.set_xticklabels(np.arange(1, attention_map.shape[1]+1), minor=False)
    ax.set_yticklabels(np.arange(1, attention_map.shape[0]+1), minor=False)

    # 添加图例
    plt.colorbar(heatmap)

    # 显示可视化结果
    plt.show()
