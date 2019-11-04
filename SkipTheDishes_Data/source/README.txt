to get the external package(catboost)
use !pip install catboost

for statistic visulization, reader could examine the visualization.ipynb

first we use train_csv_transform.py to discretization the data into day of week, hours of the day and item count, quantities, and transform to numpy format

then we use the ten_cross_validation.py to validate our model

MAE=:5.056597136272705

RMSE=:6.316545017935914

R2=:0.6734060542218759

then we use the file model_train.py to train the whole train set, and output the model to cat_boost.pkl

then we use test_csv_transform.py to transform the test dataset to npy format(numpy format)

then we use prediction_and_output.py to read the npy of the testset and predict it and output it to the require csv file

we hope our moel will beat the baseline model!