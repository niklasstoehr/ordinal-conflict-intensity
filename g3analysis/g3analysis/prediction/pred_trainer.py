
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def evaluate_pred(hat, gt):
    mse = np.mean((hat - gt) ** 2)
    return mse


def train_test(model, df, df_diff=None, x=["Q", "Z"], y=["Q"], n_splits=100, max_train_size=10, verbose=False):
    y_hats = np.empty((n_splits,))
    y_gts = np.empty((n_splits,))

    if isinstance(df_diff, pd.DataFrame):
        x_data = df_diff[x].values
    else:
        x_data = df[x].values
    y_data = df[y].values

    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=1)
    for i, (train_index, test_index) in enumerate(tscv.split(x_data)):
        if verbose: print("train:", train_index, "test:", test_index)
        x_train, y_test = x_data[train_index], y_data[test_index]

        model.fit(x_train)
        pred = model.predict(x_train)
        if verbose: print(f"gt: {y_test}, hat: {pred}")

        if isinstance(df_diff, pd.DataFrame):
            #print("diff pred", pred[0], "base", df[x].values[train_index[-1]])
            x_train_0 = df[x].values[train_index[-1]]
            pred = (x_train_0 + pred).reshape(-1)

        if len(pred.shape) > 1:
            pred = pred.squeeze()

        #print("pred", pred[x.index(y[0])], "true", y_test)
        if y[0] in x:
            y_hats[i] = pred[x.index(y[0])]
        elif len(x) == 1:
            y_hats[i] = pred[0]
        y_gts[i] = y_test

    mse_res = evaluate_pred(y_hats, y_gts)
    print(f"mse {round(mse_res,5)}")





