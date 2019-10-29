import pandas as pd
Trainings = pd.read_csv('train_orders_data_with_targets.csv', header = 0, skiprows =  lambda i: i%5 == 0 and i > 0)
Test = pd.read_csv('train_orders_data_with_targets.csv', header = 0, skiprows =  lambda i: i%5 != 0)
Trainings.to_csv('training.csv')
Test.to_csv('tests.csv')
