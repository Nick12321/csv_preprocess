import pandas as pd
# Load the dataset
# churn_dataset = pd.read_csv('../Downloads/bank_churn_project_1.csv')
churn_dataset = pd.read_csv('/home/nick/Documents/dev/ml_csv_clean/bank_churn_project_1.csv')
label = 'Exited'
# Change the order of the columns and write the file without headers
cols = churn_dataset.columns.tolist()
colIdx = churn_dataset.columns.get_loc(label)
cols = cols[colIdx:colIdx+1] + cols[0:colIdx] + cols[colIdx+1:]
modified_data = churn_dataset[cols]
# Write the file without headers
# modified_data.to_csv(‘../Downloads/bank_churn_project_1_modified.csv', header = False, index = False)
modified_data.to_csv('/home/nick/Documents/dev/ml_csv_clean/bank_churn_project_1_modified.csv', header = False, index = False)