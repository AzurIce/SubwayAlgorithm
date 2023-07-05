import sklearn 
import pandas as pd
from sklearn.cluster import KMeans
import csv
import numpy as np
import pymysql 

conn = pymysql.connect(host='123.60.53.131',user='root',password='123456',database='NewYork',charset='utf8')
cursor = conn.cursor()
with open('C:/Users/BlackCat\Desktop/archive/NYC_neighborhood_census_data_2020.csv','r') as f:
    reader = csv.reader(f)
    result = list(reader)
    estimator = KMeans(n_clusters=3)#构造聚类器
    label = []
    data = []
    for i in range(len(result)-1):
        label.append(result[i+1][0])
        data.append(result[i+1][1:])
data = np.array(data,dtype=float)
data = np.divide(np.subtract(data,np.min(data,axis=0)),np.subtract(np.max(data,axis=0),np.min(data,axis=0)))
print(data)
estimator.fit(data)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
for i in range(len(label_pred)):
    print(label[i]+" "+str(label_pred[i]))
    sql = "insert into bigSmall values(%s,%s)"
    cursor.execute(sql,(label[i],label_pred[i]))
    conn.commit()