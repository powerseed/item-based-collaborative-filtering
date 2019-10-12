# -*- coding: utf-8 -*-
movie_data = open('movies.dat')
rating_data = open('ratings.dat')
m_lines = movie_data.readlines()
r_lines = rating_data.readlines()
for i in range(len(m_lines)):
    data = m_lines[i].split('::')
    
    if data[0] != i + 1:
        for j in range(len(r_lines)):
            if(r_lines[j][])
        m_csv.loc[m_csv['movieId']== m_id_col[i],['movieId']] = i + 1   
m_csv.to_csv('movies.csv')
r_csv.to_csv('ratings.csv')
            
