import pandas as pd
m_csv = pd.read_csv('../data/movies.csv')
r_csv = pd.read_csv('../data/ratings.csv')
mr_csv = pd.merge(r_csv,m_csv, on = 'movieId', how = 'inner' )[['userId','movieId','rating','timestamp','title','genres']]
mr_csv['date'] = pd.to_datetime(mr_csv['timestamp'], unit = 's')
mr_csv = mr_csv.sort_values(by = ['timestamp'])
mr_csv.to_csv('movies_ratings.csv')