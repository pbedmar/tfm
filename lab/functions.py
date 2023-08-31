import os
import warnings
from statistics import mean
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap


from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError, mean_absolute_scaled_error, mean_absolute_error
from sktime.transformations.series.difference import Differencer
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import make_reduction

from skforecast.ForecasterAutoreg import ForecasterAutoreg

SEED = 0

##########################################################################
#                          Outlier management                            #
##########################################################################

def remove_outliers_isolation_forests(data, contamination=0.01):
    outlier_detector = IsolationForest(contamination=contamination)
    for column in data:
        outlier_detector.fit(data[[column]].values)
        outlier_prediction = outlier_detector.predict(data[[column]].values)
        data.loc[outlier_prediction == -1, column] = np.nan
        data[[column]] = data[[column]].interpolate().fillna(method='ffill').fillna(method='bfill')

    return data


def remove_outliers_isolation_forests_train_test(data_train, data_test, columns):
    outlier_detector = IsolationForest(contamination=0.01, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for column in columns:

            outlier_detector.fit(data_train[[column]])

            outlier_prediction_train = outlier_detector.predict(data_train[[column]])
            data_train.loc[outlier_prediction_train == -1, column] = np.nan
            data_train.loc[:, column] = data_train.loc[:, column].interpolate().fillna(method='ffill').fillna(method='bfill')

            outlier_prediction_test = outlier_detector.predict(data_test[[column]])
            data_test.loc[outlier_prediction_test == -1, column] = np.nan
            data_test.loc[:, column] = data_test.loc[:, column].interpolate().fillna(method='ffill').fillna(method='bfill')

    return data_train, data_test


##########################################################################
#                           ML table computation                         #
##########################################################################

def compute_lag(y, lags):
    name = y.columns[0]

    for lag in lags:
        kwargs = {name+"_shift"+str(lag): y[name].shift(lag)}
        y = y.assign(**kwargs)
        # y = pd.concat([y, y[name].shift(lag)], axis=1)

    y = y.drop(name, axis=1)
    return y


def compute_lag_fh_df(y, lags, fh):
    return pd.concat([y, compute_lag(y, lags).shift(fh-1)], axis=1)


def add_date_features(y, X, lags, fh, date_features):
    y = pd.DataFrame(y)

    if "year" in date_features:
        X['year'] = y.index.year
    if "month" in date_features:
        X['month'] = y.index.month
    if "day" in date_features:
        X['day'] = y.index.day
    if "hour" in date_features:
        X['hour'] = y.index.hour
    if "day_of_week" in date_features:
        X['day_of_week'] = y.index.day_of_week
    if "day_of_year" in date_features:
        X['day_of_year'] = y.index.day_of_year
    if "week_of_year" in date_features:
        X['week_of_year'] = y.index.weekofyear

    df = pd.concat([compute_lag_fh_df(y, lags, fh), X], axis=1)
    df = df.copy()
    return df.dropna()


##########################################################################
#                         Price forecast (sktime)                        #
##########################################################################

def crossval_model_sktime(y, X, regressor_list, regressor_str_list, window_length, max_lag, step_size, fh, save_path, save_name):
    cv = SlidingWindowSplitter(fh, window_length, step_size)
    #cv = ExpandingWindowSplitter(fh, window_length, step_size)

    cv_results_list = []
    for regressor, regressor_str in zip(regressor_list, regressor_str_list):
        print(f"Training setup: regressor {regressor_str}")

        # refit strategy makes a new model on each iteration, not an update
        forecaster = make_reduction(regressor, strategy="direct", window_length=max_lag, windows_identical=True)
        # cv_results = evaluate(forecaster, y=y, X=X, cv=cv, strategy="refit", error_score="raise", return_data=True, scoring=MeanAbsoluteScaledError(), backend="dask", compute=True)
        cv_results = evaluate(forecaster, y=y, X=X, cv=cv, strategy="refit", error_score="raise", return_data=True, scoring=MeanAbsoluteScaledError())
        cv_results_list.append(cv_results)

    cv_results_df = pd.DataFrame(columns=list(cv_results_list[0].columns))
    for element, regressor_str in zip(cv_results_list, regressor_str_list):
        element["model"] = regressor_str
        cv_results_df = pd.concat([cv_results_df, element], axis=0, ignore_index=True)

    os.makedirs(save_path, exist_ok=True)
    cv_results_df.to_pickle(save_path + save_name)
    return cv_results_df


def crossval_summary_sktime(cv_results_df):
    cv_results_df_summary = pd.DataFrame(columns=['model', 'mae', 'mase', 'fit_time'])
    for model in cv_results_df["model"].unique():
        cv_results_model = cv_results_df[cv_results_df["model"] == model]

        mae = 0
        for test, pred in zip(cv_results_model["y_test"], cv_results_model["y_pred"]):
            mae += mean_absolute_error(test, pred)
        row = {'model': model,
               'mae': mae,
               'mase': cv_results_model["test_MeanAbsoluteScaledError"].mean(),
               'fit_time': cv_results_model["fit_time"].sum()
               }
        row_df = pd.DataFrame([row])
        cv_results_df_summary = pd.concat([cv_results_df_summary, row_df], axis=0, ignore_index=True)

    cv_results_df_summary = cv_results_df_summary.sort_values(by="mase")
    return cv_results_df_summary


def crossval_plot_series(y, cv_results_df):
    for model in cv_results_df["model"].unique():
        cv_results_model = cv_results_df[cv_results_df["model"]==model]
        windows_nb = len(cv_results_model["y_pred"].tolist())

        fig, ax = plot_series(
            y,
            *cv_results_model["y_pred"].tolist(),
            markers=["o"] + ["" for x in range(windows_nb)]
        )

        ax.plot


def crossval_arima_sktime(y, X, window_lengths, step_size, fh, save_path, save_name):

    cv_results_list = []
    for window_length in window_lengths:
        cv = SlidingWindowSplitter(fh, window_length, step_size)

        forecaster_str = f"AutoARIMA_{window_length}"
        print(f"Training setup: regressor {forecaster_str}")

        forecaster = AutoARIMA(maxiter=200, random_state=0)
        cv_results = evaluate(forecaster, y=y, X=X, cv=cv, strategy="refit", error_score="raise", return_data=True, scoring=MeanAbsoluteScaledError())
        cv_results_list.append(cv_results)

    cv_results_df = pd.DataFrame(columns=list(cv_results_list[0].columns))
    for result, window_length in zip(cv_results_list, window_lengths):
        forecaster_str = f"AutoARIMA_{window_length}"
        result["model"] = forecaster_str
        cv_results_df = pd.concat([cv_results_df, result], axis=0, ignore_index=True)

    os.makedirs(save_path, exist_ok=True)
    cv_results_df.to_pickle(save_path + save_name)
    return cv_results_df


def relation_between_response_and_predictor(y, X, diff_lag):
    transformer = Differencer(lags=diff_lag, na_handling="drop_na")

    y_transform = transformer.fit_transform(y)
    x_transform = transformer.fit_transform(X)
    plot_series(y_transform)
    if isinstance(X, pd.Series):
        plot_series(x_transform)
    if isinstance(X, pd.DataFrame):
        if X.shape[1] <= 1:
            plot_series(x_transform)

    forecaster = AutoARIMA(suppress_warnings=True)
    forecaster.fit(y=y_transform, X=x_transform)

    return forecaster.get_fitted_params()


##########################################################################
#                       Price forecast (skforecast)                      #
##########################################################################

def crossval_model_skforecast(data, regressor_list, regressor_str_list, initial_window_length, nb_windows, step_size, lags, date_features, forecasting_horizon):
    cv_results_df_summary = pd.DataFrame(0, index=np.arange(len(regressor_str_list)),
                                         columns=["model", "mean_mase", "mean_mae", "modelling_time"])
    i = 0

    for regressor_str, regressor in zip(regressor_str_list, regressor_list):
        y_pred_list = []
        y_test_list = []
        mase_list = []
        mae_list = []

        start = time()
        for w in range(nb_windows):
            data_window = data[:initial_window_length + w * step_size]

            data_window_train = data_window.iloc[:-forecasting_horizon, :].copy()
            data_window_test = data_window.iloc[-forecasting_horizon:, :].copy()

            data_window_train, data_window_test = remove_outliers_isolation_forests_train_test(data_window_train, data_window_test, columns=[column for column in data_window_train.columns if column not in date_features])

            y_window_train_iso = data_window_train.iloc[:, 0]
            X_window_train_iso = data_window_train.iloc[:, 1:]
            y_window_test_iso = data_window_test.iloc[:, 0]
            X_window_test_iso = data_window_test.iloc[:, 1:]

            scaler = StandardScaler()
            scaler.fit(X_window_train_iso)
            X_window_train_scaled = scaler.transform(X_window_train_iso)
            X_window_test_scaled = scaler.transform(X_window_test_iso)
            X_window_train_scaled = pd.DataFrame(X_window_train_scaled, columns=X_window_train_iso.columns, index=X_window_train_iso.index)
            X_window_test_scaled = pd.DataFrame(X_window_test_scaled, columns=X_window_test_iso.columns, index=X_window_test_iso.index)

            forecaster = ForecasterAutoreg(
                regressor=regressor,
                lags=lags
            )

            forecaster.fit(
                y=y_window_train_iso,
                exog=X_window_train_scaled
            )

            if X_window_test_scaled.isna().sum().sum() == 0:
                y_pred = forecaster.predict(
                    steps=forecasting_horizon,
                    exog=X_window_test_scaled
                )

                y_pred = pd.Series(data=y_pred, index=y_window_test_iso.index)

                mase = mean_absolute_scaled_error(y_window_test_iso, y_pred, y_train=y_window_train_iso)
                mae = mean_absolute_error(y_window_test_iso, y_pred)

                y_test_list.append(y_window_test_iso)
                y_pred_list.append(y_pred)
                mase_list.append(mase)
                mae_list.append(mae)

        end = time()

        print("Model:", regressor_str)
        print("MASEs:", mase_list)
        print("MAEs:", mae_list)
        print("Modelling time:", end-start)
        print()

        cv_results_df_summary.iloc[i, 0] = regressor_str
        cv_results_df_summary.iloc[i, 1] = mean(mase_list)
        cv_results_df_summary.iloc[i, 2] = mean(mae_list)
        cv_results_df_summary.iloc[i, 3] = end-start
        i += 1

        fig, ax = plot_series(
            data_window.iloc[:, 0],
            *y_pred_list,
            markers=["o"] + ["" for x in range(len(y_pred_list))]
        )
        plt.title("Cross-validation of " + regressor_str)

    return cv_results_df_summary


def final_model_skforecast(data_train, data_test, best_model, best_model_str, lags, date_features,
                           forecasting_horizon, cv_results_df_summary, plot_lim, plot_title):

    data_train_iso, data_test_iso = remove_outliers_isolation_forests_train_test(data_train, data_test,
                                                                                 columns=[column for column in
                                                                                          data_train.columns if
                                                                                          column not in date_features])

    y_train_iso = data_train_iso.iloc[:, 0]
    X_train_iso = data_train_iso.iloc[:, 1:]
    y_test_iso = data_test_iso.iloc[:, 0]
    X_test_iso = data_test_iso.iloc[:, 1:]

    scaler = StandardScaler()
    scaler.fit(X_train_iso)
    X_train_scaled = scaler.transform(X_train_iso)
    X_test_scaled = scaler.transform(X_test_iso)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_iso.columns, index=X_train_iso.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_iso.columns, index=X_test_iso.index)

    forecaster = ForecasterAutoreg(
        regressor=best_model,
        lags=lags
    )

    forecaster.fit(
        y=y_train_iso,
        exog=X_train_scaled,
    )

    y_pred = forecaster.predict(
        steps=forecasting_horizon,
        exog=X_test_scaled
    )

    y_pred = pd.Series(data=y_pred, index=y_test_iso.index)
    mase = mean_absolute_scaled_error(y_test_iso, y_pred, y_train=y_train_iso)
    print("MASE:", mase)

    residuals = abs(y_pred - y_test_iso)
    RMSFE = np.sqrt(sum([x ** 2 for x in residuals]) / len(residuals))
    band_size = 1.96 * RMSFE

    if type(y_test_iso.index) is pd.core.indexes.period.PeriodIndex:
        y_test_iso.index = y_test_iso.index.to_timestamp()
    if type(y_pred.index) is pd.core.indexes.period.PeriodIndex:
        y_pred.index = y_pred.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.lineplot(data=y_test_iso, palette="deepskyblue", marker="o", ax=ax)
    sns.lineplot(data=y_pred, palette="orange", marker="o", ax=ax)
    ax.fill_between(y_test_iso.index, (y_pred - band_size), (y_pred + band_size), color='orange', alpha=.1)
    ax.set_title(plot_title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily SPOT market price')
    ax.set_ylim(plot_lim)
    ax.legend(loc="upper right", labels=["real value", "prediction"])
    plt.show()


##########################################################################
#                         Predictor's influence                          #
##########################################################################

def predictors_influence_study(data_train, data_test, best_model):

    y_train = data_train.iloc[:, 0]
    X_train = data_train.iloc[:, 1:]
    y_test = data_test.iloc[:, 0]
    X_test = data_test.iloc[:, 1:]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
    print("Prediction MASE:", mean_absolute_scaled_error(y_test, y_pred, y_train=y_train))
    explainer = shap.Explainer(best_model.predict, X_test_scaled, seed=SEED)
    shap_values = explainer(X_test_scaled)

    return shap_values


if __name__ == '__main__':
    from etl.esios.provider import ESIOSDataProvider

    esios_provider = ESIOSDataProvider()
    esios_tickers = esios_provider.get_tickers()

    esios_df = esios_provider.get_all_series(freq="H", start_index="2023-01-01 00:00", end_index="2023-03-31 23:59")

    contamination = 0.05
    #esios_df = remove_outliers_isolation_forests(esios_df, contamination)

    esios_spot = esios_df["PRECIO_MERCADO_SPOT_DIARIO"]
    esios_demand = esios_df["DEMANDA_REAL"]
    X = esios_df.drop(['PRECIO_MERCADO_SPOT_DIARIO', 'GENERACIÓN_MEDIDA_TOTAL'], axis=1)

    print(esios_spot.head(10))

    #compute_ml_ts_data(esios_spot,X,1,1)