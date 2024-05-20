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

#"python scripts/analyze_detail.py --dummy_infile ${SOURCES[0]} --purity ${TARGETS[0]} --acc ${TARGETS[1]} --std_obs ${TARGETS[2]} --infile ${JSONL_LOC}"),

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--dummy_infile")
     parser.add_argument("--purity")
     parser.add_argument("--acc")
     parser.add_argument("--std_obs")
     parser.add_argument("--infile")
     args, rest = parser.parse_known_args()

     sns.set_palette("muted")
     index = []
     purities = defaultdict(list)
     accs = []
     std_obs = []
     total_s = 0
     
     with open(args.infile, "rt") as j_in:
          for c,line in enumerate(j_in):
               cluster = json.loads(line)
               index.append(c)
               for lbl,item in cluster.items():
                    if lbl in ["std","obv","ocr","rev"]:
                         purities[lbl].append(item["counts"])
                         total_s += item["counts"]
               accs.append(cluster["acc"])
               std_obs.append(cluster["std_obs_acc"])

     p_df = pd.DataFrame(purities, index=index)
     p_df.plot(kind="bar", stacked=True)
     plt.xlabel("Cluster")
     plt.ylabel("Token count")
     plt.title("Purity by cluster")
     plt.savefig(args.purity, format="png")
     plt.clf()

     fig, ax = plt.subplots()
     p_df.plot(kind="bar", stacked=True, ax=ax)
     ax.set_ylim(1, total_s)
     plt.xlabel("Cluster")
     plt.ylabel("Token count")
     plt.title("Purity by cluster")
     plt.savefig(args.purity[0:-4]+"_pinned.png")
     plt.clf()
     

     acc_df = pd.DataFrame(accs, index=index, columns=["Accs"])
     sns.barplot(acc_df.reset_index(), x="index", y="Accs")
     plt.xlabel("Cluster")
     plt.ylabel("Cluster accuracy")
     plt.title("Accuracy by cluster")
     plt.savefig(args.acc, format="png")
     plt.clf()

     acc_std_df = pd.DataFrame(std_obs, index=index, columns=["std_obs"])
     sns.barplot(acc_std_df.reset_index(), x="index", y="std_obs")
     plt.xlabel("Cluster")
     plt.ylabel("Cluster accuracy, std+obs")
     plt.title("Std+obs accuracy by cluster")
     plt.savefig(args.std_obs, format="png")
     plt.clf()     
