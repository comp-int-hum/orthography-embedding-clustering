import argparse
import glob
import gzip
import json
import numpy as np
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import jellyfish

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--error1", help="Error 1")
     parser.add_argument("--error2", help="Error 2")
     parser.add_argument("--e_outs", nargs="+", help="CSV of shared/disjoint error")


     args, rest = parser.parse_known_args()

     e1_df = pd.read_csv(args.error1,  usecols=["Group","Token","Standard","Dtag"])
     e2_df = pd.read_csv(args.error2,  usecols=["Group","Token","Standard","Dtag"])

     e1_df["LD"] = e1_df.apply(lambda r: jellyfish.levenshtein_distance(r["Token"], r["Standard"]), axis=1)
     e2_df["LD"] = e2_df.apply(lambda r: jellyfish.levenshtein_distance(r["Token"], r["Standard"]), axis=1)

     shared = pd.merge(e1_df, e2_df, on=["Group"])
     shared.to_csv(args.e_outs[0])
     print(sum(shared["LD_x"])/len(shared))

     print(shared)


     e1_dj = e1_df[~e1_df["Group"].isin(shared["Group"])]
     e1_dj.to_csv(args.e_outs[1])
     print(sum(e1_dj["LD"])/len(e1_dj))

     
     print(e1_dj)

     e2_dj = e2_df[~e2_df["Group"].isin(shared["Group"])]
     e2_dj.to_csv(args.e_outs[2])
     print(sum(e2_dj["LD"])/len(e2_dj))


     
     print(e2_dj)


     
