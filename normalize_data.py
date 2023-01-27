import pandas as pd

file_name = 'car_evaluation_enc'

df = pd.read_csv(f'./Datasets/{file_name}.csv')

df['target']=(df['target']-df['target'].min())/(df['target'].max()-df['target'].min())
df.to_csv(f'./Datasets/{file_name}_reg.csv', index=False)
