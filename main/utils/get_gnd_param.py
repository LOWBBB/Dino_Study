import pickle

import numpy as np
from sympy.codegen.cnodes import sizeof

'''
从gnd文件中获取roxford固定给出的文件名排序信息
'''

def inspect_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"数据类型: {type(data)}")
    print(f"数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")

    if isinstance(data, dict):
        print("字典键:", list(data.keys()))
        for key, value in data.items():
            print(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'N/A'}")
    elif isinstance(data, list):
        print("列表前5个元素类型:", [type(x) for x in data[:5]])
    elif isinstance(data, np.ndarray):
        print(f"数组形状: {data.shape}")
        print(f"数组数据类型: {data.dtype}")

    return data

# 使用
data = inspect_pkl('../data/datasets/roxford5k/gnd_roxford5k.pkl')
imlist = data.get('imlist')
qimlist = data.get('qimlist')
print(qimlist)
print('-'*50)

'''
print(qimlist[0])
print(imlist[1])
print(imlist[2956])
print(imlist[1232])
print(imlist[296])
print(imlist[2918])
'''


