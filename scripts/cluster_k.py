import argparse
import glob
import gzip
import json
import numpy as np
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

import torch
from collections import defaultdict

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--embeds", help="JSONgz embed")
     parser.add_argument("--outfile", help="csv out")
     parser.add_argument("--k", type=int, default=15, help="K cluster range")
     parser.add_argument("--cluster_element", choices=["Diffs","Embeds"], help="Element to cluster over")
     parser.add_argument("--label_set", nargs="+", default=["std","rev","ocr","obv"])
     parser.add_argument("--random_state", type=int, default=0)
     
     args, rest = parser.parse_known_args()

     with open(args.embeds, "rb") as e_in:
          embeds = pickle.load(e_in)
          x_embeds = embeds["x_embeds"]
          x_diffs = embeds["x_diffs"]
          y = embeds["y"]
          standards = embeds["standards"]
          labels = embeds["labels"]

          filtered =  [f for f in zip(x_embeds, x_diffs, y, standards, labels) if f[4] in args.label_set] 
          x_embeds = [f[0] for f in filtered]

          x_diffs = [f[1] for f in filtered]
          y = [f[2] for f in filtered]
          standards = [f[3] for f in filtered]
          labels = [f[4] for f in filtered]
          
          
          group_labels = []
          ln = 0
          for i, s in enumerate(standards):
               if i % len(args.label_set) == 0:
                    ln += 1
               group_labels.append(ln)

          if args.cluster_element == "Diffs":
               analyze = x_diffs
          else:
               analyze = x_embeds

     print("K: "+str(args.k))
     kmeans = KMeans(n_clusters=args.k, random_state=0, n_init="auto").fit(np.array(analyze))

          
     k_df = pd.DataFrame(zip(y, standards, kmeans.labels_, x_diffs, x_embeds, labels, group_labels), columns = ["Token", "Standard", "Cluster", "Diffs", "Embeds","Label", "Group"])
     full_PCA = PCA(n_components=2)
     fit = full_PCA.fit_transform(k_df[args.cluster_element].tolist())
     k_df["x"] = [f[0] for f in fit]
     k_df["y"] = [f[1] for f in fit]
     
          
     k_df.to_csv(args.outfile, columns=["Token","Cluster","Label","Group", "Standard", "x", "y"], index=False)

