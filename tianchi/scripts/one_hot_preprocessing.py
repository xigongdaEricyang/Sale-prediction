import pandas as pd
from sklearn import preprocessing
import numpy as np

raw_data_df = pd.read_csv(r'C:\Users\YW59785\Desktop\tianchi\raw_data\[new] yancheng_train_20171226.csv', delimiter=',')

need_one_hot_columns = ['brand_id', 'type_id', 'gearbox_type', 'if_charging',
                        'driven_type_id', 'fuel_type_id', 'newenergy_type_id',
                        'if_MPV_id', 'if_luxurious_id']

need_one_hot_df = raw_data_df[need_one_hot_columns]

ty = ['brand_id', 'type_id','if_charging','driven_type_id', 'fuel_type_id', 'newenergy_type_id', 'if_MPV_id', 'if_luxurious_id']

le = preprocessing.LabelEncoder()
one_hot_encoder = preprocessing.OneHotEncoder()

for t in ty: 
    a = le.fit_transform(need_one_hot_df[t])
    data = one_hot_encoder.fit_transform(pd.DataFrame(a)).toarray()
    np.savetxt(r"C:\Users\YW59785\Desktop\tianchi\processed_data\{}_one_hot.csv".format(t), data, delimiter=",", fmt="%.f")

a = le.fit_transform(['MT', 'DCT', 'AT', 'CVT', 'AMT'])

gearbox_type_dict = {
    'MT' : [ 0., 0.,  0.,  0.,  1.],
    'DCT' : [ 0.,  0.,  0.,  1.,  0.],
    'AT' : [ 0.,  1. , 0. , 0. , 0.],
    'CVT' : [ 0.,  0. , 1. , 0. , 0.],
    'AMT' : [ 1. , 0.,  0. , 0. , 0.],
    'AT;DCT' : [0, 1, 0 , 1 , 0],
    'MT;AT' : [0 , 1, 0 , 0 , 1]
}

values = need_one_hot_df['gearbox_type'].tolist()

values = [gearbox_type_dict[item] for item in values]

np.savetxt(r"C:\Users\YW59785\Desktop\tianchi\processed_data\gearbox_type_one_hot.csv", values, delimiter=",", fmt="%.f")