"""Print total and mean complete update time across the requested shards."""
import argparse

from sisa_timing import summarize


parser = argparse.ArgumentParser()
parser.add_argument("--container", required=True)
parser.add_argument("--label", default="latest")
parser.add_argument("--shards", type=int)
args = parser.parse_args()
summary = summarize(args.container, args.label, args.shards)
print("{},{}".format(summary["update_time_sec"], summary["update_time_sec"] / summary["shards"]))
