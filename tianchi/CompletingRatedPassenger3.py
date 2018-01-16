import numpy as np
from sklearn import linear_model

import pandas as pd
import math


raw_data_df = pd.read_csv(r'./tmp_data_files/rp_df.csv', sep=',')

rp_df = raw_data_df[['car_length', 'car_width', 'car_height', 'rated_passenger']]

def getDataSet(isAvaliable_set = True):
    tmp_set = rp_df[rp_df.rated_passenger.str.contains('\d+$') == isAvaliable_set]
    x_results = tmp_set.drop('rated_passenger', axis=1).values
    y_results = tmp_set['rated_passenger'].values
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
# np.savetxt(r"C:\Users\YW59785\Desktop\tianchi\tmp_data_files\{}_output.csv".format('rated_passenger'), output, delimiter=",", fmt="%.f")

