import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv
scaler = MinMaxScaler()
file_name = 'airfoil_self_noise'

df = pd.read_csv(f'./Datasets/{file_name}.dat',delim_whitespace=True)

df = scaler.fit_transform(df)
df = pd.DataFrame.from_records(df)
df.to_csv(f'./Datasets/{file_name}_reg.csv', index=False, quoting=csv.QUOTE_ALL)
