import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import csv
scaler = MinMaxScaler()
file_name = 'yacht_hydrodynamics'

#df = scaler.fit_transform(df)
#df = pd.DataFrame.from_records(df)
#df.to_csv(f'./Datasets/{file_name}.csv', index=False, quoting=csv.QUOTE_ALL)

'''df = pd.read_csv(f'./Datasets/{file_name}.csv')
df_scaled = scaler.fit_transform(df.iloc[:,:-1])
df_scaled = pd.DataFrame.from_records(df_scaled)
df_scaled.loc[:,'target']=df['target']
df_scaled.to_csv(f'./Datasets/{file_name}_reg_x.csv', index=False, quoting=csv.QUOTE_ALL)'''


df = pd.read_csv(f'./Datasets/{file_name}.csv')
x_scaled = scaler.fit_transform(df.iloc[:,:-1])
x_scaled = pd.DataFrame.from_records(x_scaled)

scaler = StandardScaler()
y_scaled = scaler.fit_transform(df.iloc[:,-1:])
y_scaled = pd.DataFrame.from_records(y_scaled)



x_scaled.loc[:,'target']=y_scaled.iloc[:,-1:]
x_scaled.to_csv(f'./Datasets/{file_name}_reg_stand.csv', index=False, quoting=csv.QUOTE_ALL)
