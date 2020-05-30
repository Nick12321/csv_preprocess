#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:16:10 2020

@author: nick
"""
import pandas as pd
import numpy as np

data = pd.read_csv('~/Documents/dev/csv_preprocess/to_downtown_condo/step2_needs_process/toronto_condo_to_process.csv')
print(data.shape)
data_type_dict = dict(data.dtypes)
print('Data type of each column of Dataframe:', data_type_dict)
print('-------------------------------------')
column2=data_type_dict[1,0]
print(column2)