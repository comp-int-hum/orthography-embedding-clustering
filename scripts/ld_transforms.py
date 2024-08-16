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
     parser.add_argument("--dummy_infile")
     parser.add_argument("--n_edits", default=5, type=int, help="n most common edits/cluster")
     parser.add_argument("--edits_out")
     parser.add_argument("--lds_out")
     parser.add_argument("--outfile")
     parser.add_argument("--infile")
     args, rest = parser.parse_known_args()

     sns.set_palette("muted")

     cluster_lds = []
     cluster_transforms = []
     cluster_distinct_transforms = []
     with open(args.infile, "rt") as j_in, open(args.outfile, "wt") as j_out, open(args.outfile[:-5]+"toks.txt", "wt") as t_out:
          for i,line in enumerate(j_in):
               t_out.write("[cluster "+str(i+1)+"]\n")
               cluster = json.loads(line)
               c_out = {"lds": [], "transforms": defaultdict(int)}
               for lbl,item in cluster.items():
                    if lbl in ["std","obv","ocr","rev"]:
                         for token in item["tokens"]:
                              t_out.write(token["token"] + " -> " + token["standard"] + "\n")
                              c_out["lds"].append(jellyfish.levenshtein_distance(token["token"], token["standard"]))
                              for d in difflib.ndiff(token["standard"], token["token"]):
                                   if d[0] in ["+","-"]:
                                        c_out["transforms"][d]+=1
               c_out["lds"] = sum(c_out["lds"])/len(c_out["lds"])
               cluster_lds.append(c_out["lds"])
               cluster_transforms.append(c_out["transforms"])
               cluster_distinct_transforms.append(len(c_out["transforms"]))
               j_out.write(json.dumps(c_out) + "\n")
     

     ld_df = pd.DataFrame(cluster_lds)
     #l = pd.DataFrame(cluster_transforms).apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=args.n_edits)
     t_df = pd.DataFrame(cluster_transforms).apply(lambda s, n: s.nlargest(n)/s.sum(), axis=1, n=args.n_edits)
     print(t_df)

     ld_df.plot(kind="bar")
     plt.xlabel("Cluster")
     plt.ylabel("Avg. LD")
     plt.title("Avg. LD of candidate and standard token by cluster")
     plt.savefig(args.lds_out)
     plt.clf()

     fig, ax = plt.subplots(figsize=(12, 7))
     t_df.plot(kind="bar", stacked=True, ax=ax)
     ax.set_ylim(0,1.0)
     #for c,l in zip(ax.containers, cluster_distinct_transforms):
     #     ax.bar_label(c, labels=[l], label_type="edge")
     ax.set_xticklabels([str(c+1) + ": " + str(dt) for c, dt in enumerate(cluster_distinct_transforms)])
     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
     plt.xlabel("Cluster: N distinct transformations")
     plt.ylabel("Transformation percentages")
     plt.title("Top " + str(args.n_edits) +  " transformations by cluster")
     plt.savefig(args.edits_out)
