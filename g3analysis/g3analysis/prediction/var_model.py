
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings("ignore")


from g0configs import helpers
from g3analysis import analysis, latents
from g3analysis.prediction import pred_trainer, evaluation


class VAR_class():

    def __init__(self, bestlag=4, verbose=False):
        self.bestlag = bestlag
        self.var_model = None
        self.verbose = verbose

    def fit(self, train_data):
        var_model = VAR(train_data)
        var_model = var_model.fit(self.bestlag)
        if self.verbose: print(var_model.summary())
        self.var_model = var_model

    def predict(self, x_train, nobs=1):
        prediction = self.var_model.forecast(y=x_train[-self.bestlag:], steps=nobs)
        # df_pred = pd.DataFrame(prediction, index=df.index[-nobs:], columns=df.columns + column_name)
        return prediction



if __name__ == "__main__":

    df = latents.main(model_file="model_gqta_nc_5", geo_date={"geo": ["egypt"], "date": ["2004-01-01", "2012-01-01"]})
    df = analysis.group_impute(df, freq = 'M')

    x = ["G", "Q", "T", "A"]
    y = ["Q"]
    df = df[x]

    evaluation.stationarity_test(df, diff_type="0")

    df_diff = evaluation.df_diff(df)
    evaluation.stationarity_test(df, diff_type="1")

    aic_lag, bic_lag = evaluation.find_model_order(df_diff, maxlag=12, verbose=False)

    model = VAR_class(bestlag=bic_lag)
    pred_trainer.train_test(model, df, df_diff = df_diff, x = x, y = y, max_train_size = 1000, n_splits = 24)
    evaluation.granger_test(df_diff, maxlag = [12], variables = x, verbose = False)
