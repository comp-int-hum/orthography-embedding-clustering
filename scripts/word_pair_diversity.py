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
     parser.add_argument("--outfile", help="Cluster image file")
     parser.add_argument("--remove_singletons", action="store_false")

     args, rest = parser.parse_known_args()

     k_df = pd.read_csv(args.pca_csv)
     print(k_df)
     input()
     if args.remove_singletons:
          multi = k_df["Standard"].value_counts() > 1
          k_df = k_df[k_df["Standard"].isin(multi[multi].index)]

     #avg number of unique combos per
     #multiple exact matches pairings are ok
     #so what we really want is something that penalizes splitting unique std-obv pairings and rewards clumping them, while also penalizing higher incidence of clustering standards with different obvs

     #total = len(Counter([o+"*"+s for o, s in zip(k_df["Token"], k_df["Standard"])]))
     #total = 0
     total = len(k_df)
     total_max = 0
     for w_name, w_group in k_df.groupby(["Standard"]):
          print(w_name)
          print(w_group)
          input()
     #for c_name, c_group in k_df.groupby("Cluster"):
          #c_group = c_group.reset_index(drop=True)
          #std_count = Counter([s for s in c_group["Standard"]])
          #total_max += std_count.most_common(1)[0][1]
          #full_count = Counter([o+"*"+s for o, s in zip(c_group["Token"], c_group["Standard"])])
          #print(full_count)
          #input()
          #unique_row = [u.split("*")[1] for u in full_count.keys()]
          #print(unique_row)
          #std_counts = Counter(unique_row)
          #total_max += sum([v for v in std_counts.values() if v > 1])
          #print(std_counts)
          #print(std_counts.most_common(1)[0][1])
          #total_max += std_counts.most_common(1)[0][1]
          #input()
          
          #ax = fig.add_subplot(rows, args.ncols, position[int(c_name)])
          #ax.quiver(origin, origin, c_group["x"].to_list(), c_group["y"].to_list(), angles="xy", scale_units="xy", scale=60, color=colors)
          #ax.set_xticks([])
          #ax.set_yticks([])
          #ax.title.set_text(str(c_name+1))
          #ax.set_xlabel(cs)
     #print(1-total_max/total)
     #plt.tight_layout()
     #plt.savefig(args.outfile, format="png")
               
          

     
