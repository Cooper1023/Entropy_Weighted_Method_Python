# -*- coding: UTF-8 -*-
"""
@Author: Cooper
@File: 熵权法
@Version: 1.0
@Time: 2023-06-06
@Email: zx_copper1023@126.com
"""

import pandas as pd
import numpy as np


def calculate_entropy(x, normal_method):
    """
    熵权法求权重
    input: 
    x               - input dataframe
    normal_method   - 归一化的处理方式, e.g. 'positive', 'negative'
    output: Series  - 每个指标的权重
    """
    
    
    if normal_method == None:
        x = x
    # 正向指标，即数值越大越好
    elif normal_method == 'positive':
        x = x.apply(lambda y: ((y - np.min(y)) / (np.max(y) - np.min(y))))
    # 负向指标，即数值越小越好
    elif normal_method == 'negative':
        x = x.apply(lambda y: ((np.max(y) - y) / (np.max(y) - np.min(y))))
    

    # 计算概率p_ij比重
    p = x
    for column in p.columns:
        sum_p = sum(p[column])
        p[column] = p[column].apply(
            lambda x_ij: x_ij / sum_p if x_ij / sum_p != 0 else 1e-6
        )  # #####避免比重为0的情况，后续无法求log，这个很重要否则会报错#####

    # K值
    k = 1 / np.log(x.index.size)  # 1/log(行数)
    # print(k)

    # 计算信息熵
    E = (-k) * np.array([sum([p_ij * np.log(p_ij)
                         for p_ij in p[column]])
                         for column in p.columns]
                        )
    E = pd.Series(E, index=p.columns)

    # 差异系数
    d = pd.Series(1 - E, index=p.columns)

    # 计算指标权重
    w = d / sum(d)
    w.name = 'weight'
    return w


if __name__ == "__main__":
    data = [[28, 10, 56], [60, 14, 58], [15, 5, 54]]
    df = pd.DataFrame(data, columns=['A', 'B', 'C'])
    print(df)
    # 计算权重，只能输入数据，不能有字符串
    weight = calculate_entropy(df)
    print('权重:\n', weight)

    # 根据权重计算综合得分
    # $ score = sum(weight1*指标1 + weight2*指标2 + …… ) $
    df_score = df.apply(lambda x: sum(x * weight), axis=1)
    print('得分\n', df_score)


