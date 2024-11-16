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
     parser.add_argument("--so_e", help="so error out")
     parser.add_argument("--acc_e", help="acc error out")
     parser.add_argument("--label_set", nargs="+", default=["obv","std","rev","ocr"])
     parser.add_argument("--error_stem")

     args, rest = parser.parse_known_args()

     purities = []
     avg_accs = []
     avg_std_obv_accs = []
     ks = []
     dtag_purity = []
     dtag_concentration = []

     dtag_proportion = {}


     error_groups = []
     error_tokens = []
     error_clusters = []

     so_groups = []
     so_tokens = []
     so_clusters = []
     
     for k, c_csv in enumerate(args.cluster_csvs):
          ks.append(k+1)
          cluster_acc = 0
          cluster_std_obs_acc = 0
          so_e_g = []
          so_e_t = []
          so_e_d = []
          so_e_s = []
          
          e_g = []
          e_t = []
          e_d = []
          e_s = []

          so_c_g = []
          so_c_t = []
          so_c_d = []
          so_c_s = []
          
          k_df = pd.read_csv(c_csv)
          k_df = k_df.loc[k_df["Label"].isin(args.label_set)]
          purities.append(k_df.groupby("Cluster")["Label"].value_counts().groupby("Cluster").max().sum()/len(k_df))
          dtag_purity.append(k_df.groupby("Cluster")["Dtag"].value_counts().groupby("Cluster").max().sum()/len(k_df))
          d_c = [c for c in k_df.groupby("Dtag")["Cluster"].max()]
          dtag_concentration.append(sum(d_c)/len(d_c))
          for gn, g in k_df.groupby(["Group"]):
               if "obv" in g["Label"].to_list() and "std" in g["Label"].to_list():
                    if g.loc[g["Label"] == "obv"]["Cluster"].item() == g.loc[g["Label"] == "std"]["Cluster"].item():
                         cluster_std_obs_acc += 1
                         so_c_g.append(gn[0])
                         so_c_t.append(g.loc[g["Label"] == "obv"]["Token"].item())
                         so_c_d.append(g.loc[g["Label"] == "obv"]["Dtag"].item())
                         so_c_s.append(g.loc[g["Label"] == "obv"]["Standard"].item())
                    else:
                         so_e_g.append(gn[0])
                         so_e_t.append(g.loc[g["Label"] == "obv"]["Token"].item())
                         so_e_d.append(g.loc[g["Label"] == "obv"]["Dtag"].item())
                         so_e_s.append(g.loc[g["Label"] == "obv"]["Standard"].item())
               if g["Cluster"].eq(g["Cluster"].to_list()[0]).all():
                    cluster_acc +=1
               else:
                    e_g.append(gn[0])
                    e_t.append(g.loc[g["Label"] == "obv"]["Token"].item())
                    e_d.append(g.loc[g["Label"] == "obv"]["Dtag"].item())
                    e_s.append(g.loc[g["Label"] == "obv"]["Standard"].item())
                    
          e_df = pd.DataFrame({"Group": e_g, "Token": e_t, "Standard": e_s, "Dtag": e_d})
          e_df.to_csv(args.error_stem+str(k+1)+"error_acc.csv", index=False)
          s_df = pd.DataFrame({"Group": so_e_g, "Token": so_e_t, "Standard":so_e_s, "Dtag":so_e_d})
          s_df.to_csv(args.error_stem+str(k+1)+"error_so.csv")
          c_df = pd.DataFrame({"Group": so_c_g, "Token": so_c_t, "Standard":so_c_s, "Dtag":so_c_d})
          c_df.to_csv(args.error_stem+str(k+1)+"correct_so.csv")
          error_groups += e_g
          error_tokens += e_t
          so_groups += so_e_g
          so_tokens += so_e_t
          
          avg_accs.append(cluster_acc/(len(k_df)/len(args.label_set)))
          avg_std_obv_accs.append(cluster_std_obs_acc/(len(k_df)/len(args.label_set)))
     
     d = np.array([ks, purities, dtag_purity, avg_accs, avg_std_obv_accs, dtag_concentration]).T.tolist()
     o_df = pd.DataFrame(data=d, columns = ["K", "Purity", "Dtag_purity",  "Avg_Acc", "Avg_SO_Acc", "D_conc"])
     o_df.to_csv(args.summary_out)

     e_df = pd.DataFrame({"Group": error_groups, "Token": error_tokens})
     e_df.to_csv(args.acc_e, index=False)

     so_df = pd.DataFrame({"Group": so_groups, "Token": so_tokens})
     so_df.to_csv(args.so_e, index=False)
     
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
     
