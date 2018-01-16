import pandas as pd
import numpy  as np
import math
from sklearn import linear_model

raw_data_df = pd.read_csv(r'./processed_data/predict_level_id.csv', sep=',')

# engine_df = raw_data_df[['driven_type_id','displacement', 'wheelbase', 'power', 'front_track','rear_track', 'cylinder_number', 'car_length', 'car_width', 'car_height', 'engine_torque']]

def getDataSet(isAvaliable_set = True):
    tmp_set = raw_data_df[raw_data_df.level_id.str.contains('\d+$') == isAvaliable_set]
    x_results = tmp_set.drop('level_id', axis=1).values
    y_results = tmp_set['level_id'].values
    return x_results, y_results

result_x_avi, result_y_avi = getDataSet() 

length = len(result_x_avi)
split = math.ceil(0.7*length)


train_set_x = result_x_avi[0: split]
train_set_y = result_y_avi[0: split]

test_set_x = result_x_avi[split:]
test_set_y = result_y_avi[split:]

result_x_des, result_y_des = getDataSet(False)

X = train_set_x
Y = train_set_y
h = 0.02
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)

Z = logreg.predict(test_set_x)

time = 0
for i, item in enumerate(Z):
    if item == test_set_y[i]:
        time = time +1

print('accuracy: {}'.format(time/len(test_set_y)))

output = logreg.predict(result_x_des)

print(output)

