import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from util import check_path, calculate_results

parser = argparse.ArgumentParser(description="Post process the results")
parser.add_argument(
    "--result_path",
    default=os.path.join(".", "results", "Training_Evaluate"),
    type=lambda s: os.path.expanduser(s),
    help="The path to the available current reults",
)
parser.add_argument(
    "--compile_path",
    default=os.path.join(".", "results", "Compiled"),
    type=lambda s: os.path.expanduser(s),
    help="The path to save the compiled results",
)


args = parser.parse_args()
check_path(args.compile_path)


architope_train = calculate_results(args.result_path, "architope")
architope_test = calculate_results(args.result_path, "architope", is_train=False)

ffNN_train = calculate_results(args.result_path, "ffNN", ["complete"])
ffNN_test = calculate_results(args.result_path, "ffNN", ["complete"], is_train=False)

ffNNdp_train = calculate_results(args.result_path, "ffNN-dp", ["complete_dp"])
ffNNdp_test = calculate_results(
    args.result_path, "ffNN-dp", ["complete_dp"], is_train=False
)

test_results = pd.concat([architope_test, ffNN_test, ffNNdp_test])
train_results = pd.concat([architope_train, ffNN_train, ffNNdp_train])
train_results.to_csv(os.path.join(args.compile_path, "complied_train.csv"), index=True)
test_results.to_csv(os.path.join(args.compile_path, "complied_test.csv"), index=True)

print("Compiled results are saved to " + str(args.compile_path))
