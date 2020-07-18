from util import *
import os
import argparse
import pickle
from Grid import ret_files


parser = argparse.ArgumentParser(description="Compare ffnn with the branching version")
parser.add_argument(
    "--is_test",
    default=False,
    type=lambda s: s == "T",
    choices=[True, False],
    required=True,
    help="Should the System be tested or not",
)
parser.add_argument(
    "--is_manual",
    default=False,
    type=lambda s: s == "T",
    choices=[True, False],
    required=True,
    help="Is the train test generation manual or not",
)
parser.add_argument(
    "--n_iter",
    default=1,
    required=True,
    help="The number of random instances for the grid search",
    type=int,
)
parser.add_argument(
    "--n_jobs",
    required=True,
    default=1,
    help="The number of jobs to process simultaneously on cpus",
    type=int,
)
parser.add_argument(
    "--test_size", default=0.3, type=float, help="The test size",
)
parser.add_argument(
    "--result_path",
    type=lambda s: os.path.expanduser(s),
    required=True,
    help="The path to save the results",
)
parser.add_argument(
    "--data_path",
    type=lambda s: os.path.expanduser(s),
    required=True,
    help="The path to save the results",
)


args = parser.parse_args()
check_path(args.result_path)
files = ret_files(args.data_path)

# get feauture engineered data
for name, ftr in files.items():
    path = ftr["file"]
    k = ftr["k"]
    grid_type = ftr["grid"]
    X_train, y_train, X_test, y_test = prepare_data(
        data_path=path, test_size=args.test_size, manual=args.is_manual
    )
    param_grid = get_param_grid(
        args.is_test, input_dim=X_train.shape[1], type=grid_type
    )

    branches = {
        "0": {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    }

    evaluate_branching_structure(
        branches, param=param_grid, k=k, n_iter=args.n_iter, n_jobs=args.n_jobs
    )
    write_results(mydict=branches, main_path=args.result_path, type=name)
    del branches
