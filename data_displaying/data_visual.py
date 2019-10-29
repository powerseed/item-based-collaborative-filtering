import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt 
import random
mr_csv = pd.read_csv('movies_ratings.csv', parse_dates = ['date'])
mr_csv['year-week'] = mr_csv['date'].dt.strftime('%Y-%U')
#mr_csv['day-of-week'] = mr_csv['date'].dt.day_name()
#days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#mr_csv['day-of-week'] = pd.Categorical(mr_csv['day-of-week'], categories=days)
#means_by_week = mr_csv.groupby('year-week').mean().rename(columns = {'rating':'rating-mean'})

##the user is increasing
#year_week = mr_csv['year-week'].unique()
#cum_count_user = []
#for yw in year_week:
#    cum_count_user.append(len(mr_csv[mr_csv['year-week'] <= yw]['userId'].unique()))
#plt.plot(range(len(cum_count_user)),cum_count_user)
#plt.show()
#
#year_week = mr_csv['year-week'].unique()
#cum_count_movie = []
#for yw in year_week:
#    cum_count_movie.append(len(mr_csv[mr_csv['year-week'] <= yw]['movieId'].unique()))
#plt.plot(range(len(cum_count_movie)),cum_count_movie)
#plt.show()
#
#
###the rating mean is changing by time
#plt.plot(range(means_by_week.shape[0]),means_by_week['rating-mean'])
#plt.show()



########################################################################################
#user_id = [random.randint(0,len(mr_csv['userId'].unique())) for i in range(5)]
#for id in user_id:
    mr_csv[mr_csv['userId'] == id].groupby('day-of-week')['userId'].count().plot(kind='bar')
    plt.show()
########################################################################################
#    
#user movie time genres
umtg = mr_csv[['userId','movieId','year-week','genres']]
user_df = umtg[umtg['userId'] == 4780]
genres_array = user_df['genres'].values
#timestamp = user_df['timestamp'].values