import argparse
import glob
import gzip
import json
import numpy as np
import pickle

import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--pca_csv", help="CSV with pca information")
     parser.add_argument("--transforms", help="JSONl with transformations")
     parser.add_argument("--ntransforms", type=int, default=3)
     parser.add_argument("--outfile", help="Cluster image file")
     parser.add_argument("--ncols", type=int, default=3)
     parser.add_argument("--k", type=int)

     args, rest = parser.parse_known_args()

     k_df = pd.read_csv(args.pca_csv)

     rows = args.k//args.ncols
     rows = rows + 1 if args.k%args.ncols != 0 else rows + 0
     position = range(1, args.k+1)

     h = 4.8
     if rows > 3:
          h = 7
     if rows > 4:
          h = 8.5
     fig = plt.figure(1, figsize=(7,h))
     
     commons = []
     with open(args.transforms, "rt") as t_in:
          for line in t_in:
               j_line = json.loads(line)
               c = Counter(j_line["transforms"])
               
               commons.append(str(len(c)) + ": " + ", ".join(t[0] +" " + str(t[1]) for t in  c.most_common(args.ntransforms)))
     
     for dfg, cs in zip(k_df.groupby("Cluster"), commons):
          c_name = dfg[0]
          c_group = dfg[1]
          c_group = c_group.reset_index(drop=True)
          
          origin = [0] * len(c_group)
          colors = ["blue" if l == "obv" else "red" for l in c_group["Label"].to_list()]

          ax = fig.add_subplot(rows, args.ncols, position[int(c_name)])
          ax.quiver(origin, origin, c_group["x"].to_list(), c_group["y"].to_list(), angles="xy", scale_units="xy", scale=60, color=colors)
          ax.set_xticks([])
          ax.set_yticks([])
          ax.title.set_text(str(c_name+1))
          ax.set_xlabel(cs)
     
     plt.tight_layout()
     plt.savefig(args.outfile, format="png")
               
          

     
