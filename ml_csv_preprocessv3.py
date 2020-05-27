import pandas as pd
import numpy as np

data = pd.read_csv('condo_by_area_test.csv')
# Specify label column name here

col_fill_name = ['PARK','Bay Street','Dufferin Grove','Kensington / China town','Little Portugal','Niagara','Palmerston / Little Italy','Trinity Bellwoods','University','Waterfront']
for name in col_fill_name:
    data[name] = data[name].fillna(0).astype('int64')
    

data['BEDS'] = data['BEDS'].fillna(0).astype('int64')

for i in range(0,len(data['SQFT'])):

     cell = str(data.at[i,'SQFT'])
     values = cell.split('-')
     median = 0
     if(len(values) == 2):
         median = int((int(values[0]) + int(values[1]))/2)
     data.at[i,'SQFT'] = median



"""for cell in data['SQFT'].head():
    print(cell)
    values = str(cell).split('-')
    print(values)
    median = int((int (values[0])+int(values[1]))/2)
    print(median)
    cell = median
"""
data['SQFT']=data['SQFT'].astype('int64')

for c in data.columns:
     print(data[c].dtype, c )


label = 'SOLD PRICE'
print(data.head())

# Rearrage the dataset columns
cols = data.columns.tolist()
print(cols)
colIdx = data.columns.get_loc(label)
print(colIdx)
# Do nothing if the label is in the 0th position
# Otherwise, change the order of columns to move label to 0th position
if colIdx != 0:
    cols = cols[colIdx:colIdx+1] + cols[0:colIdx] + cols[colIdx+1:]
print(cols[colIdx:colIdx+1] , cols[0:colIdx] , cols[colIdx+1:])
# Change the order of data so that label is in the 0th column
modified_data = data[cols]
print(modified_data.head())
print (cols)


# Remove the useless columns
cat_cols = modified_data.select_dtypes(exclude=['int64', 'float64']).columns
print(cat_cols)
cat_cols = set(cat_cols) - {label}
# set returns an oredered list
print(cat_cols)

useless_cols = []
for cat_column_features in cat_cols:
    num_cat = modified_data[cat_column_features].nunique()
    if num_cat > 10:
        useless_cols.append(cat_column_features)
print(useless_cols)
for feature_column in modified_data.columns:
    num_cat = modified_data[feature_column].nunique()
    if num_cat <= 1:
        useless_cols.append(feature_column)
print(useless_cols)
print('---------------------------------------------------')
modified_data = modified_data.drop(useless_cols, axis=1)
print(modified_data.head())
print('---------------------------------------------------')

# One hot encode and fill missing values
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
    
# Remove label so that it is not encoded
data_without_label = modified_data.drop([label], axis=1)
# Fills missing values with the median value
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numeric_features = data_without_label.select_dtypes(include=['int64',
                                                    'float64']).columns

categorical_features = data_without_label.select_dtypes(exclude=['int64',
                                                            'float64']).columns

# Create the column transformer
preprocessor_cols = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),
                  ('cat', categorical_transformer, categorical_features)])
# Create a pipeline with the column transformer, note that
# more things can be added to this pipeline in the future
preprocessor = Pipeline(steps=[('preprocessor', preprocessor_cols)])
preprocessor.fit(data_without_label)
modified_data_without_label = preprocessor.transform(data_without_label)
if (type(modified_data_without_label) is not np.ndarray):
    modified_data_without_label = modified_data_without_label.toarray()
print('processed data')
print(modified_data_without_label)
modified_data_array = np.concatenate(
    (np.array(modified_data[label]).reshape(-1, 1),
     modified_data_without_label), axis=1)
# modified_data_array = modified_data_array.astype(int)
print('final data')
print(modified_data_array)
np.savetxt("data_processedv3.csv", modified_data_array, delimiter=",", fmt="%d")

