
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import warnings
warnings.filterwarnings("ignore")


from g0configs import helpers
from g3analysis import analysis, latents
from g3analysis.prediction import pred_trainer, evaluation


class AR_class():

    def __init__(self, verbose=False):
        self.ar_model = None
        self.verbose = verbose

    def select_order(self, train_data):
        self.sel = ar_select_order(train_data, 12, ic = "bic", seasonal=False)
        if self.sel.ar_lags != None:
            self.bestlag = self.sel.ar_lags[-1]
        else:
            self.bestlag = 1

    def fit(self, train_data):
        self.select_order(train_data)
        ar_model = AutoReg(train_data, self.bestlag)
        ar_model = ar_model.fit()
        if self.verbose: print(ar_model.summary())
        self.ar_model = ar_model


    def predict(self,x_train):
        prediction = self.ar_model.predict(start=-self.bestlag, end=-1)
        return prediction



if __name__ == "__main__":

    df = latents.main(model_file="model_gqta_nc_5", geo_date={"geo": ["yemen"], "date": ["2004-01-01", "2012-01-01"]})
    df = analysis.group_impute(df, freq = 'M')

    x = ["G"]
    y = ["G"]
    stat_test_df = df[x]

    evaluation.stationarity_test(stat_test_df, diff_type="0")
    df_diff = evaluation.df_diff(stat_test_df)
    evaluation.stationarity_test(stat_test_df, diff_type="1")

    model = AR_class()
    pred_trainer.train_test(model, df, df_diff = df_diff, x = x, y = y, max_train_size = 1000, n_splits = 10)
    print(f"optimal lag bic: {model.bestlag}")
