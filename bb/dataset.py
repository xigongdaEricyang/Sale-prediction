import pandas as pd
from sklearn.model_selection import train_test_split

class DataSet(object):
    """docstring for ClassName."""


    def __init__(self, file):
        super(ClassName, self).__init__()
        self.raw_df = pd.read_csv(file)

    def normalize(self):
        normalizeColumns = ['compartment','TR','displacement','price_level','power','level_id',
         'cylinder_number','engine_torque','car_length','car_height','car_width','total_quality','equipment_quality',
          'rated_passenger','wheelbase','front_track','rear_track']
        
        leftDf = df.drop(normalizeColumns, axis =1 ).drop(['sale_quantity'], axis = 1)
        normalizeDf = self.raw_df[normalizeColumns]
        normalizeDf = (normalizeDf-normalizeDf.min())/(normalizeDf.max()-normalizeDf.min())
        inputDf = pd.concat([leftDf, normalizeDf], axis = 1)

    def getDataSet():

        