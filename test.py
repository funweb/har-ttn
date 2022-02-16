import numpy as np


A = np.arange(95,99).reshape(2,2)    #原始输入数组

A = np.pad(A,((0,0), (1,0)),'constant',constant_values = (0,0))  #constant_values表示填充值，且(before，after)的填充值等于（0,0）

print(A)
