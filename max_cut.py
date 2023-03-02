import pandas as pd



def max_cut(d: pd.DataFrame, logging=False):
    new_d = d.sort_values(by=0)
    temp_gb = d.groupby(1, sort=False, observed=True, dropna=False)[0]
    Nc = temp_gb.size()
    Sc = temp_gb.sum()
    first_class = new_d.iloc[0, 1]
    t_s_Nc = sum(Nc.values)
    t_s_Sc = sum(Sc.values)
    f_Nc = {k: t_s_Nc - Nc[k] for k in Nc.index}
    f_Sc = {k: t_s_Sc - Sc[k] for k in Nc.index}
    theta0 = f_Sc[first_class] - new_d.iloc[0, 0] * f_Nc[first_class]
    threshold = new_d.iloc[0, 0]
    max_index = new_d.iloc[0].name
    if logging is True:
        print(f'new_d {new_d}\n')
        print(f'temp_gb {temp_gb}\n')
        print(f't_s_Nc {t_s_Nc}\n')
        print(f't_s_Sc {t_s_Sc}\n')
        print(f'f_Nc {f_Nc}\n')
        print(f'f_Sc {f_Sc}\n')
        print(threshold)
        print(max_index)
    for i in range(t_s_Nc - 1):
        theta1 = theta0 + f_Sc[first_class] - new_d.iloc[i + 1, 0] * f_Nc[first_class]
        if theta1 > theta0:
            threshold = new_d.iloc[i + 1, 0]
            theta0 = theta1
            max_index = i + 1
        elif theta1 == theta0:
            threshold = (new_d.iloc[i + 1, 0] + theta0) / 2
            max_index = i + 1

    return threshold, new_d.iloc[max_index].name


