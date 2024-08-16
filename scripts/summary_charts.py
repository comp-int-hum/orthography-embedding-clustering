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

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--cluster_csvs", nargs="+", help="List of clustered data csvs")
     parser.add_argument("--summary_out", help="CSV of summary statistics")
     parser.add_argument("--purity", help="Purity chart out")
     parser.add_argument("--accs_out", help="Accuracies out")
     parser.add_argument("--dtag_purity", help = "Dtag purity out")
     parser.add_argument("--d_conc")
     parser.add_argument("--so_acc", help="Standard obvs acc out")
     parser.add_argument("--label_set", nargs="+", default=["obv","std","rev","ocr"])

     args, rest = parser.parse_known_args()

     purities = []
     avg_accs = []
     avg_std_obv_accs = []
     ks = []
     dtag_purity = []
     dtag_concentration = []
     
     for k, c_csv in enumerate(args.cluster_csvs):
          ks.append(k+1)
          cluster_acc = 0
          cluster_std_obs_acc = 0
          k_df = pd.read_csv(c_csv)
          k_df = k_df.loc[k_df["Label"].isin(args.label_set)]
          purities.append(k_df.groupby("Cluster")["Label"].value_counts().groupby("Cluster").max().sum()/len(k_df))
          dtag_purity.append(k_df.groupby("Cluster")["Dtag"].value_counts().groupby("Cluster").max().sum()/len(k_df))
          d_c = [c for c in k_df.groupby("Dtag")["Cluster"].max()]
          dtag_concentration.append(sum(d_c)/len(d_c))
          for gn, g in k_df.groupby(["Cluster", "Group"]):
               if g["Label"].str.contains("std").sum() > 0 and g["Label"].str.contains("obv").sum() > 0:
                    cluster_std_obs_acc += 1
               if g["Label"].count() == len(args.label_set):
                    cluster_acc += 1
          avg_accs.append(cluster_acc/(len(k_df)/len(args.label_set)))
          avg_std_obv_accs.append(cluster_std_obs_acc/(len(k_df)/len(args.label_set)))

     
     d = np.array([ks, purities, dtag_purity, avg_accs, avg_std_obv_accs, dtag_concentration]).T.tolist()
     o_df = pd.DataFrame(data=d, columns = ["K", "Purity", "Dtag_purity",  "Avg_Acc", "Avg_SO_Acc", "D_conc"])
     o_df.to_csv(args.summary_out)
          
     
     plt.plot(ks, purities, "bx-")
     plt.xticks(np.arange(min(ks), max(ks), 1))
     plt.xlabel("K")
     plt.ylabel("Purity")
     plt.savefig(args.purity, format="png")

     plt.clf()
     plt.plot(ks, dtag_concentration, "bx-")
     plt.xticks(np.arange(min(ks), max(ks), 1))
     plt.xlabel("K")
     plt.ylabel("Avg. Dtag concentration")
     plt.savefig(args.d_conc, format="png")
     
     plt.clf()
     plt.plot(ks, dtag_purity, "bx-")
     plt.xticks(np.arange(min(ks), max(ks), 1))
     plt.xlabel("K")
     plt.ylabel("Dtag purity")
     plt.savefig(args.dtag_purity, format="png")

     plt.clf()
     plt.plot(ks, avg_accs, "bx-")
     plt.xticks(np.arange(min(ks), max(ks), 1))
     plt.xlabel("K")
     plt.ylabel("Avg Acc")
     plt.savefig(args.accs_out, format="png")

     plt.clf()
     plt.plot(ks, avg_std_obv_accs, "bx-")
     plt.xticks(np.arange(min(ks), max(ks), 1))
     plt.xlabel("K")
     plt.ylabel("Avg SO Acc")
     plt.savefig(args.so_acc, format="png")
     
