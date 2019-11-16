import pandas as pd
from sklearn.cluster import KMeans

def cluster(column):
    k = 5
    kmodel = KMeans(n_clusters = k, n_jobs = 4) #建立模型，n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(column.values.reshape((len(column), 1))) #训练模型
    c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)  #输出聚类中心，并且排序（默认是随机序的）
    w = c.rolling(2).mean().iloc[1:] #相邻两项求中点，作为边界点
    w = [0] + list(w[0]) + [column.max()] #把首末边界点加上，w[0]中0为列索引
    d3 = pd.cut(column, w)

    return d3
