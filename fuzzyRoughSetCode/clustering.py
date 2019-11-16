import pandas as pd
from sklearn.cluster import KMeans

def cluster(column):
    # data = {
    #     'apples': [0.056,0.488,0.107,0.322,0.242,0.389,0.246,0.330,0.257,0.205,0.330,0.235,0.267]
    # }
    # column = pd.DataFrame(column)
    # column = column[u'apples'].copy()
    k = 5

    #等宽法
    # d1 = pd.cut(column, k, labels = range(k)) #等宽离散化，各个类比依次命名为0,1,2,3

    kmodel = KMeans(n_clusters = k, n_jobs = 4) #建立模型，n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(column.values.reshape((len(column), 1))) #训练模型
    c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)  #输出聚类中心，并且排序（默认是随机序的）
    w = c.rolling(2).mean().iloc[1:] #相邻两项求中点，作为边界点
    w = [0] + list(w[0]) + [column.max()] #把首末边界点加上，w[0]中0为列索引
    d3 = pd.cut(column, w)

    return d3
    # print(d3)