import joblib
import numpy as np
import bz2file as bz2
from sklearn.ensemble import ExtraTreesClassifier

#Defining function for Encoding the input values
def ordinal_encoder(input_val,feats):
  feat_val=list(1+np.arange(len(feats)))
  feat_key=feats
  feat_dict=dict(zip(feat_key,feat_val))
  value=feat_dict[input_val]
  return value

def get_prediction(data,model):
    """
    Predict the class of a given data
    """
    return model.predict(data)

def decompress_pickle(file):
  data = bz2.BZ2File(file, 'rb')
  data = joblib.load(data)
  return data