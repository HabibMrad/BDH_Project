# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 02:46:41 2018

@author: Srini
"""

import pandas as pd
import numpy as np

###################################

train_val = pd.read_pickle('train_val_filtered.pkl')
uniquepatientid = train_val.patient_id.unique()

#7:1 train:val ratio
val_id = np.random.choice(uniquepatientid, size=int(np.floor(len(uniquepatientid)/8)), replace= False)

val = train_val[train_val['patient_id'].isin(val_id)]
train = train_val[~train_val['patient_id'].isin(val_id)]

train.to_pickle('train.pkl')
val.to_pickle('val.pkl')

####################################