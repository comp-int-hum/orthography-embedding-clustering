import argparse
import glob
import gzip
import json
import numpy as np
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
import math

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--cluster_csv", help="Clustered data csv")
     parser.add_argument("--proportion_out", help="CSV of dtag proportions")
     #parser.add_argument("--label_set", nargs="+", default=["obv","std","rev","ocr"])

     args, rest = parser.parse_known_args()


     dtag_proportions = []
     
     k_df = pd.read_csv(args.cluster_csv)
     dtags = sorted([str(t) for t in list(set(k_df["Dtag"]))])
     print(dtags)


     
     for gn, g in k_df.groupby(["Cluster"]):
          k = gn[0]
          dtag_proportion = {dt:0 for dt in dtags}
          dtag_proportion["k"] = k
          dtag_proportion["count"] = len(g)
          
          for tag, val in g["Dtag"].value_counts(dropna=False, normalize=True).to_dict().items():
               dtag_proportion[tag] = round(val,3)

          dtag_proportions.append(dtag_proportion)

               
     o_df = pd.DataFrame.from_dict(dtag_proportions)
     o_df = o_df[["k","count"]+dtags]
     print(o_df)
     o_df.to_csv(args.proportion_out, index=False)
     #d = np.array([ks, purities, dtag_purity, avg_accs, avg_std_obv_accs, dtag_concentration]).T.tolist()
     #o_df = pd.DataFrame(data=d, columns = ["K", "Purity", "Dtag_purity",  "Avg_Acc", "Avg_SO_Acc", "D_conc"])
     #o_df.to_csv(args.summary_out)

     #e_df = pd.DataFrame({"Group": error_groups, "Cluster": error_clusters, "Token": error_tokens})
     #e_df.to_csv(args.so_e, index=False)
          
