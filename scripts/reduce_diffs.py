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
     parser.add_argument("--outdir", help="Dir for cluster images")

     args, rest = parser.parse_known_args()

     k_df = pd.read_csv(args.pca_csv)


     for c_name, c_group in k_df.groupby("Cluster"):
          c_group = c_group.reset_index(drop=True)
          print(c_group)
          
          origin = [0] * len(c_group)

          plt.quiver(origin, origin, c_group["x"].to_list(), c_group["y"].to_list(), angles="xy", scale_units="xy", scale=35)
          plt.xticks([])
          plt.yticks([])
          plt.title("Cluster: " + str(c_name+1))
          plt.savefig(args.outdir+str(c_name+1)+".png", format="png")
               
          plt.clf()
          #legend_labels.append(str(len(c_group)) + ": "+ " ".join(c_group["Token"][0:3]))

          #for t_name, t_group in c_group.groupby("Label"):
          #     cluster[t_name]["tokens"] = [{"token": t,  "standard": s} for t,s in zip(t_group["Token"].tolist(), t_group["Standard"].tolist())]
          #clusters.append(cluster)
          #sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, 1.35), ncol=3, title=None, frameon=False
          

     
