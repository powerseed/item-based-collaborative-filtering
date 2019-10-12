import pandas as pd
m_csv = pd.read_csv('movies.dat', sep = '::')
r_csv = pd.read_csv('ratings.dat', sep = '::')
m_id_col = m_csv['movieId'].as_matrix()
for i in range(m_id_col.shape[0]):
    print(i)
    if m_id_col[i] != i + 1:
        r_csv.loc[r_csv['movieId']== m_id_col[i],['movieId']] = i + 1
        m_csv.loc[m_csv['movieId']== m_id_col[i],['movieId']] = i + 1   
m_csv.to_csv('movies.csv')
r_csv.to_csv('ratings.csv')            