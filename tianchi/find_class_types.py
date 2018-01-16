import pandas as pd

df_3 = pd.read_csv(r'C:\Users\YW59785\Desktop\tianchi\model2\df_3.csv', delimiter=',')

classlist = df_3['class_id'].drop_duplicates()

classdict = {}

def getCarTypesbyClassid(classid):
    a_list = df_3.drop(['sale_date', 'class_id', 'sale_quantity'], axis=1)[df_3.class_id == classid].values
    seen = set()
    newlist = []
    for item in a_list:
        t = tuple(item)
        if t not in seen:
            newlist.append(item)
            seen.add(t)
    return newlist

df_4 = df_3.drop(['sale_date', 'class_id', 'sale_quantity'], axis=1)
for classid in classlist:
    classdict[classid] = getCarTypesbyClassid(classid)


# for classid in classlist:

import csv
with open(r'C:\Users\YW59785\Desktop\tianchi\model2\dict2.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in classdict.items():
       writer.writerow([key, value])