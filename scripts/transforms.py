import argparse
import glob
import gzip
import json
import numpy as np
import pickle

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
import difflib
import jellyfish



if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--k_csv", help="csv with cluster info")
     parser.add_argument("--edits_out", help="JSONL edits")
     parser.add_argument("--tokens_out", help="plaintext edits")
     args, rest = parser.parse_known_args()

     cluster_lds = []
     cluster_transforms = []
     cluster_distinct_transforms = []

     k_df = pd.read_csv(args.k_csv)
     with open(args.edits_out, "wt") as e_out, open(args.tokens_out, "wt") as t_out:
          for gn, g in k_df.groupby("Cluster"):
               t_out.write("[cluster "+str(gn+1)+"]\n")
               c_out = {"lds": [], "dtags": [], "transform_detail": [], "transforms": defaultdict(int)}
               for token, standard, dtag in zip(g["Token"].to_list(), g["Standard"].to_list(), g["Dtag"].to_list()):
                    dlist = []
                    t_out.write(token + " -> " + standard + " : " + str(dtag)+"\n")
                    c_out["lds"].append(jellyfish.levenshtein_distance(token, standard))
                    c_out["dtags"].append(dtag)
                    for d in difflib.ndiff(standard, token):
                         if d[0] in ["+","-"]:
                              c_out["transforms"][d]+=1
                              dlist.append(d)
                    c_out["transform_detail"].append(dlist)
               c_out["lds"] = sum(c_out["lds"])/len(c_out["lds"])
               e_out.write(json.dumps(c_out) + "\n")
     
