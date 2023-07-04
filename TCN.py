import torch
import pandas as pd 
import numpy as np
import csv


EXIST = {}
ENTRIES = {}
with open('/Users/blackcat/Downloads/archive/NYC_subway_traffic_2017-2021.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # 读取并忽略第一行
    column_index = header.index('Datetime')  # 找到列名所在的索引
    print(column_index)
    i = 0
    for row in reader:
        value = row[column_index]  # 获取指定列的值
        i+=1
        # 对该值进行处理