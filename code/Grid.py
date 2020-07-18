# grid of parameters for training
import os


def get_param_grid(is_test, input_dim, type):
    if is_test:
        partition_epochs = [10]
        normal_epochs = [20]
        param_grid = {
            "model__learning_rate": [0.07051589252206744],
            "model__height": [10],
            "model__input_dim": [input_dim],
            "model__batch_size": [128],
            "model__dp": [0],
        }
    else:
        partition_epochs = [50, 100, 150, 200, 250, 300, 350, 400]
        normal_epochs = [200, 400, 550, 700, 900, 1000, 1100, 1200, 1400]
        param_grid = {
            "model__learning_rate": [
                0.07051589252206744,
                0.08427820781418686,
                0.0013492360549976384,
                0.0002981105714244872,
                0.00010752758249461377,
                0.014675734704629786,
                0.0007235739397496827,
                0.0011097477099521785,
                0.012754636534154566,
                0.011499029280740218,
                0.0001853444592233933,
                0.0009758535452459834,
                0.23059223949138072,
                0.2889070499692642,
                0.02276623975878011,
            ],
            "model__height": [
                50,
                60,
                70,
                80,
                90,
                100,
                150,
                200,
                250,
                300,
                350,
                400,
                450,
                500,
                600,
                700,
                900,
                1100,
                1400,
                1800,
                2000,
                2200,
                2400,
            ],
            "model__input_dim": [input_dim],
            "model__batch_size": [128],
            "model__dp": [0],
        }

    if type == "partition":
        param_grid["model__epochs"] = partition_epochs
    elif type == "normal":
        param_grid["model__epochs"] = normal_epochs

    return param_grid


def ret_files(path):
    # k is the number of folde for k-fold cross validation
    # grid speciffies the parameter Grid in which we look for the optimal hyper parameters
    files = {
        "bay": {"file": "bay.csv", "k": 4, "grid": "partition"},
        "nocean": {"file": "nocean.csv", "k": 4, "grid": "partition"},
        "inland": {"file": "inland.csv", "k": 4, "grid": "partition"},
        "oneHocean": {"file": "oneHocean.csv", "k": 4, "grid": "partition"},
        "complete": {"file": "housing_complete.csv", "k": 4, "grid": "normal"},
    }
    for key in files.keys():
        files[key]["file"] = os.path.join(path, files[key]["file"])
    return files
