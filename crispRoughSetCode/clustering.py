import pandas as pd
from sklearn.cluster import KMeans
def cluster_table(table,decision_col,n_clusters,output_file_name = -1,cat = []):
    for column in table:
        if column != decision_col and column not in cat:
            table[column] = cluster(table[column],n_clusters).astype(str)
        else:
            table[column] = table[column].astype(str)
    if output_file_name != -1:
        table.to_csv(output_file_name,index=None, header=True)

            
    return table
def cluster(column,n_cluster):
    kmodel = KMeans(n_clusters = n_cluster, n_jobs = 8,random_state = 10) #建立模型，n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(column.values.reshape((len(column), 1))) #训练模型
    c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)  #输出聚类中心，并且排序（默认是随机序的）
    w = c.rolling(2).mean().iloc[1:] #相邻两项求中点，作为边界点
    w = [column.min()] + list(w[0]) + [column.max()] #把首末边界点加上，w[0]中0为列索引
    print(w)
    d3 = pd.cut(column, w, include_lowest = True)

    return d3
