# Entropy_Weighted_Method_Python
Entropy Weighted Method 熵权法

熵权法的Python实现，支持正向/负向的归一化，或者不进行归一化。

# 使用示例
``` python
import Entropy_Weighted
data = [[28, 10, 56], [60, 14, 58], [15, 5, 54]]
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print(df)
# 计算权重
weight = calculate_entropy(df, normal_method=None)
print('权重:\n', weight)

# 根据权重计算综合得分
# $ score = sum(weight1*指标1 + weight2*指标2 + …… ) $
df_score = df.apply(lambda x: sum(x * weight), axis=1)
print('得分\n', df_score)
```
