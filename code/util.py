import pandas as pd
import numpy as np
from time import process_time, time
import gc
from keras import backend as K
from sklearn.model_selection import RandomizedSearchCV, KFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# This contains the CV-Grid
from Grid import *


def check_path(path):
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)


def check_file(file_path):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileExistsError(str(file_path) + " does not exist.")


def data_to_spherical_to_euclidean(df_input):
    # First Create Spherical Coordinates From DataFrame
    x1 = np.cos(df_input.latitude) * np.cos(df_input.longitude)
    x2 = np.cos(df_input.latitude) * np.sin(df_input.longitude)
    x3 = np.sin(df_input.latitude)
    # Write to Data-Frame
    coordinates = {"x1": x1, "x2": x2, "x3": x3}
    coordinates = pd.DataFrame(data=coordinates, index=df_input.index)

    # Compute Extrinsic (Euclidean) Mean and Project onto Sphere
    x_bar = np.mean(coordinates, axis=0)
    x_bar = x_bar / np.linalg.norm(x_bar)

    # Map to Euclidean Coordinates about the projected extrinsic mean
    def Log_Sphere(p):
        # Compute dot product between x and p
        x_dot_p = np.matmul(x_bar, p)
        # Compute Angle Between x and p
        x_p_ang = np.arccos(x_dot_p)
        # Spherical "projection" factor
        x_p_fact = x_p_ang / (np.sqrt(1 - (x_dot_p ** 2)))
        # Compute Coordinate on Tangent Space
        tangent_space_val = (p - x_bar * x_dot_p) * x_p_fact
        # Return Ouput
        return tangent_space_val

    # Return Result
    result = [Log_Sphere(row) for row in coordinates.values]
    return pd.DataFrame(result, index=df_input.index)


# This is our feature map
# Function for Joining Euclidean Coordinates with Current DataFrame
def feature_map(df_input):
    ret_vec = data_to_spherical_to_euclidean(df_input)
    df_enriched = pd.concat([df_input, ret_vec], axis=1)
    # Reset Index from 1
    df_enriched = df_enriched.reset_index(drop=True)
    return df_enriched


def add_is_train(df, test_size=0.3):
    X_train, X_test = train_test_split(
        df, shuffle=True, test_size=test_size, random_state=2000
    )
    X_train = X_train.assign(is_train=True)
    X_test = X_test.assign(is_train=False)
    X = pd.concat([X_train, X_test])

    return X


# prepare manual clusters and save them to file
def prepare_manual_clusters(source_file, sink_path):
    def print_save_messsage(path):
        print(str(path) + " is saved :)")

    df = pd.read_csv(source_file)
    df = df.drop(["total_bedrooms"], axis=1)

    df_bay = df[df.ocean_proximity == "NEAR BAY"].drop(["ocean_proximity"], axis=1)
    df_nocean = df.query("ocean_proximity in ['NEAR OCEAN', 'ISLAND']").drop(
        ["ocean_proximity"], axis=1
    )
    df_inland = df[df.ocean_proximity == "INLAND"].drop(["ocean_proximity"], axis=1)
    df_oneHocean = df[df.ocean_proximity == "<1H OCEAN"].drop(
        ["ocean_proximity"], axis=1
    )

    # Apply Feature Map to each Cut
    df_bay = add_is_train(feature_map(df_bay))
    df_nocean = add_is_train(feature_map(df_nocean))
    df_inland = add_is_train(feature_map(df_inland))
    df_oneHocean = add_is_train(feature_map(df_oneHocean))

    # Write to CSV
    df_bay.to_csv(os.path.join(sink_path, "bay.csv"), index=False)
    print_save_messsage(os.path.join(sink_path, "bay.csv"))
    df_nocean.to_csv(os.path.join(sink_path, "nocean.csv"), index=False)
    print_save_messsage(os.path.join(sink_path, "nocean.csv"))
    df_inland.to_csv(os.path.join(sink_path, "inland.csv"), index=False)
    print_save_messsage(os.path.join(sink_path, "inland.csv"))
    df_oneHocean.to_csv(os.path.join(sink_path, "oneHocean.csv"), index=False)
    print_save_messsage(os.path.join(sink_path, "oneHocean.csv"))

    # Prepare Data for FFNNs
    # Apply First Feature Map
    df_ffNN = pd.concat([df_bay, df_inland, df_nocean, df_oneHocean])
    # Map Ocean Proximity to n-ary data
    df_housing_complete = pd.get_dummies(df_ffNN)
    df_housing_complete.to_csv(
        os.path.join(sink_path, "housing_complete.csv"), index=False
    )
    print_save_messsage(os.path.join(sink_path, "housing_complete.csv"))


def prepare_columntransformer(cl):
    min_max = [
        "housing_median_age",
        "total_rooms",
        "population",
        "households",
        "median_income",
    ]
    standard = ["longitude", "latitude", "x1", "x2", "x3"]
    standard_idx = [i for i, obj in enumerate(cl) if obj in standard]
    min_max_idx = [i for i, obj in enumerate(cl) if obj in min_max]
    ct = ColumnTransformer(
        [
            ("Min-Max", MinMaxScaler(), min_max_idx),
            ("zero-one", StandardScaler(), standard_idx),
        ],
        remainder="passthrough",
    )

    return ct


def prepare_data(data_path, manual, test_size=None):
    X = pd.read_csv(data_path)
    if manual:
        X["median_house_value"] = X["median_house_value"] * (10 ** (-5))
        X_train = X[X["is_train"] == 1]
        X_test = X[X["is_train"] == 0]

        y_train = np.array(X_train["median_house_value"])
        y_test = np.array(X_test["median_house_value"])

        X_train = X_train.drop(["median_house_value", "is_train"], axis=1)
        X_test = X_test.drop(["median_house_value", "is_train"], axis=1)
    else:
        y = np.array(X["median_house_value"])
        y = y * (10 ** (-5))
        X = X.drop(["median_house_value"], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=2000
        )

    cl = X_train.columns
    X_train = pd.DataFrame(X_train, columns=cl)
    X_test = pd.DataFrame(X_test, columns=cl)

    return X_train, y_train, X_test, y_test


# MAPE, between 0 and 100
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true.shape = (y_true.shape[0], 1)
    y_pred.shape = (y_pred.shape[0], 1)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 1 hidden layer neural network
def def_model(height, learning_rate, input_dim, dp):
    ffn = Sequential()
    if dp > 0:
        ffn.add(Dropout(dp, input_shape=(input_dim,)))
        ffn.add(
            Dense(
                units=height,
                input_dim=input_dim,
                activation="sigmoid",
                kernel_initializer="random_uniform",
            )
        )
    else:
        ffn.add(
            Dense(
                units=height,
                input_dim=input_dim,
                activation="sigmoid",
                kernel_initializer="random_uniform",
            )
        )

    ffn.add(Dense(units=1))

    opt = Adam(lr=learning_rate)

    ffn.compile(optimizer=opt, loss="mae", metrics=["mse", "mae", "mape"])

    return ffn


# fit the structure defined using grid and report the predicted y_train and y_test
def fit_structure(X_train, y_train, X_test, grid):

    t_start = time()
    estimator_cur = grid.fit(X_train, y_train)
    t_dur = time() - t_start

    best_params = estimator_cur.best_params_
    y_test_hat = estimator_cur.predict(X_test)
    y_train_hat = estimator_cur.predict(X_train)
    K.clear_session()
    del estimator_cur
    gc.collect()

    return best_params, y_test_hat, y_train_hat, t_dur


# return mape, mae, mse and me based on y and y_hat
def evaluate_structure(y, y_hat):
    mape = mean_absolute_percentage_error(y_pred=y_hat, y_true=y)
    mae = mean_absolute_error(y_hat, y)
    mse = mean_squared_error(y_hat, y)
    me = np.mean(y_hat - y)

    # N Data Points
    size_test = y_hat.shape[0]

    return mape, mae, mse, me, size_test


def build_grid(param_grid, n_iter, k, n_jobs, cl):

    model = KerasRegressor(build_fn=def_model, verbose=0)
    ct = prepare_columntransformer(cl)
    model = Pipeline([("preprocess", ct), ("model", model)])

    grid = RandomizedSearchCV(
        estimator=model,
        n_jobs=n_jobs,
        cv=KFold(k, random_state=2000, shuffle=True),
        param_distributions=param_grid,
        n_iter=n_iter,
        return_train_score=True,
        random_state=2000,
    )
    return grid


# evaluate the branching structure and updte the report of fit for each branch and total
def evaluate_branching_structure(branches, param, k, n_iter, n_jobs):

    key_vec = branches.keys()

    for key in key_vec:
        branch_cur = branches[key]
        y_train = branch_cur["y_train"]
        y_test = branch_cur["y_test"]

        param_cur = param
        cl = branch_cur["X_train"].columns
        grid = build_grid(
            param_grid=param_cur, n_iter=n_iter, k=k, n_jobs=n_jobs, cl=cl
        )
        best_params, y_test_hat, y_train_hat, t_dur = fit_structure(
            branch_cur["X_train"], y_train, branch_cur["X_test"], grid
        )

        branch_cur["best_params"] = best_params
        branch_cur["y_test_hat"] = y_test_hat
        branch_cur["y_train_hat"] = y_train_hat
        branch_cur["y_test"] = y_test
        branch_cur["y_train"] = y_train
        branch_cur["t_dur"] = t_dur / 60

        mape_tr, mae_tr, mse_tr, me_tr, size_tr = evaluate_structure(
            y=y_train, y_hat=y_train_hat
        )
        branch_cur["e_train_mape"] = mape_tr
        branch_cur["e_train_mae"] = mae_tr
        branch_cur["e_train_mse"] = mse_tr
        branch_cur["e_train_me"] = me_tr
        branch_cur["e_train_size"] = size_tr

        mape_test, mae_test, mse_test, me_test, size_test = evaluate_structure(
            y=y_test, y_hat=y_test_hat
        )
        branch_cur["e_test_mape"] = mape_test
        branch_cur["e_test_mae"] = mae_test
        branch_cur["e_test_mse"] = mse_test
        branch_cur["e_test_me"] = me_test
        branch_cur["e_test_size"] = size_test


def write_results(mydict, main_path, type):
    cur_path = os.path.join(main_path, type)
    check_path(cur_path)
    txt_path = os.path.join(cur_path, type + ".txt")
    with open(txt_path, "w") as f:
        for k1, v1 in mydict.items():
            for k2, v2 in v1.items():
                if k2 in [
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "y_test_hat",
                    "y_train_hat",
                ]:

                    csv_path = str(
                        os.path.join(cur_path, str(k1) + "_" + str(k2) + ".csv")
                    )
                    pd.DataFrame(v2).to_csv(csv_path, index=False)

                else:
                    f.write(str(k1) + " >>> " + str(k2) + " >>> " + str(v2) + "\n\n")
    print("Results for " + str(type) + " is saved!")


def calculate_results(
    model_dir, name, folders=["bay", "inland", "oneHocean", "nocean"], is_train=True
):
    def evaluate_structure(y, y_hat):
        mape = mean_absolute_percentage_error(y_pred=y_hat, y_true=y)
        mae = mean_absolute_error(y_hat, y)
        mse = mean_squared_error(y_hat, y)
        me = np.mean(y_hat - y)
        sd = np.std(y_hat - y)

        return dict(mae=mae, mape=mape, mse=mse)

    i = 0
    dir_cur = os.path.join(model_dir, folders[i])
    y_test_total = pd.read_csv(os.path.join(dir_cur, "0_y_test.csv"))["0"].values
    y_test_hat_total = pd.read_csv(os.path.join(dir_cur, "0_y_test_hat.csv"))[
        "0"
    ].values
    y_train_total = pd.read_csv(os.path.join(dir_cur, "0_y_train.csv"))["0"].values
    y_train_hat_total = pd.read_csv(os.path.join(dir_cur, "0_y_train_hat.csv"))[
        "0"
    ].values
    P_time = float(
        pd.read_csv(os.path.join(dir_cur, folders[i] + ".txt"))
        .iloc[0, 0]
        .split(">>>")[2]
    )
    T_time = P_time

    if len(folders) > 1:

        for i in range(1, len(folders)):
            dir_cur = os.path.join(model_dir, folders[i])
            try:
                y_test_cur = pd.read_csv(os.path.join(dir_cur, "0_y_test.csv"))[
                    "0"
                ].values
                y_test_hat_cur = pd.read_csv(os.path.join(dir_cur, "0_y_test_hat.csv"))[
                    "0"
                ].values
                y_train_cur = pd.read_csv(os.path.join(dir_cur, "0_y_train.csv"))[
                    "0"
                ].values
                y_train_hat_cur = pd.read_csv(
                    os.path.join(dir_cur, "0_y_train_hat.csv")
                )["0"].values
                y_test_total = np.concatenate([y_test_total, y_test_cur])
                y_test_hat_total = np.concatenate([y_test_hat_total, y_test_hat_cur])
                y_train_total = np.concatenate([y_train_total, y_train_cur])
                y_train_hat_total = np.concatenate([y_train_hat_total, y_train_hat_cur])
                t_cur = float(
                    pd.read_csv(os.path.join(dir_cur, folders[i] + ".txt"))
                    .iloc[0, 0]
                    .split(">>>")[2]
                )
                P_time = np.maximum(t_cur, P_time)
                T_time = t_cur + T_time
            except:
                print(str(folders[i]) + "   is fucked up")

    if is_train:
        ret = evaluate_structure(y_train_total, y_train_hat_total)
    else:
        ret = evaluate_structure(y_test_total, y_test_hat_total)
    ret["P-time"] = P_time
    ret["T_time"] = T_time
    return pd.DataFrame(ret, index=[name])
