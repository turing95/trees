import pandas as pd
from sklearn.model_selection import KFold

ds = ['breast-cancer',
      'kr-vs-kp',
      'monk1',
      'monk2',
      'monk3',
      'soybean-small',
      'tic-tac-toe',
      'car_evaluation',
      'hayes-roth',
      'house-votes-84',
      'balance-scale'
      ]

for d in ds:
    data = pd.read_csv(f'./Datasets/{d}_enc_reg.csv')
    x = data.iloc[:, :-1]
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    n_k_folds = kf.get_n_splits(x)
    fold = 1
    for train_index, test_index in kf.split(x):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        data_train.to_csv(f'./Datasets/{d}_train_{fold}.csv', index=False)
        data_test.to_csv(f'./Datasets/{d}_test_{fold}.csv', index=False)
        fold += 1
