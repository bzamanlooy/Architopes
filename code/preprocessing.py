import argparse
import os
from pathlib import Path
from util import prepare_manual_clusters, check_path, check_file


parser = argparse.ArgumentParser(
    description="Manually cluster the California housing market data"
)
parser.add_argument(
    "--source_file",
    type=lambda s: Path(os.path.expanduser(s)),
    required=True,
    help="The path to for the raw data",
)
parser.add_argument(
    "--sink_path",
    type=lambda s: os.path.expanduser(s),
    required=True,
    help="The path for the processed data to be saved",
)
args = parser.parse_args()


check_file(args.source_file)
check_path(args.sink_path)
prepare_manual_clusters(args.source_file, args.sink_path)
