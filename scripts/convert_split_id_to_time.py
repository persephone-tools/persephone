""" Converts the IDs of trimmed segments of Na corpus into times. """

import sys

fn = sys.argv[1]
with open(fn) as f:
    for line in f:
        if line.startswith("crdo"):
            split_under = line.split("_")
            trim_id = int(split_under[-1].split(":")[0])
            mins = int(trim_id*10 / 60)
            seconds = trim_id*10 % 60
            print("%dm%ds:" % (mins, seconds))
        else:
            print(line, end="")
