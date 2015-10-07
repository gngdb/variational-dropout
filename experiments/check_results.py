#!/usr/bin/env python
#
# Checks results for NaNs in the recorded losses. Then reports where they 
# occur, and gives option to delete them.

import varout.experiments
import glob
import gzip
import pickle
import numpy as np
import os

def main(outdir, verbose=False):
    # get a list of all the results files
    results_files = glob.glob(os.path.join(outdir, "*"))
    bad_files = []
    for rfile in results_files:
        if verbose:
            print "Checking {0}".format(rfile)
        with gzip.open(rfile, "rb") as f:
            results = pickle.load(f)
            for k in results:
                for c in results[k].keys():
                    # are there any NaNs in there?
                    if any(np.isnan(results[k][c].data[:,1])):
                        # if so add this to the list of bad files
                        bad_files.append(rfile)
    # collapse list of bad files
    bad_files = list(set(bad_files))
    # pretty print a list of files
    for fname in bad_files:
        print "File: {0} contains NaNs".format(fname)
    response = None
    while response != "y" and response != "n":
        response = raw_input("Delete files? (y/n)")
        if response == "y":
            for fname in bad_files:
                os.remove(fname)
        elif response == "n":
            break

if __name__ == "__main__":
    parser = varout.experiments.get_argparser()
    args = parser.parse_args()
    main(args.output_directory, verbose=args.v)
