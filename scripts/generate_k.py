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
     parser.add_argument("--embeds", help="JSONgz embed")
     parser.add_argument("--outfile", help="Elbow png out")
     parser.add_argument("--purity", help="Purity chart out")
     parser.add_argument("--distincts_out", help="Distinct stds in clusters out")
     parser.add_argument("--so_acc")
     #parser.add_argument("--combined_accs_out")
     parser.add_argument("--k", nargs=2, type=int, default=[1,150], help="K cluster range")
     parser.add_argument("--cluster_out", default="pngs/")
     parser.add_argument("--cluster_element", choices=["Diffs","Embeds"], help="Element to cluster over")
     parser.add_argument("--label_set", nargs="+", default=["std","rev","ocr","obv"])

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

     inertias = []
     ks = []
     avg_cps = []
     avg_accs = []
     avg_std_obv_accs = []
     
     for k in range(args.k[0], args.k[1]):
          print("K: "+str(k))
          kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(np.array(analyze))
          inertias.append(kmeans.inertia_)
          print("Inertia: "+str(inertias[-1]))
          ks.append(k)

          
          k_df = pd.DataFrame(zip(y, standards, kmeans.labels_, x_diffs, x_embeds, labels, group_labels), columns = ["Token", "Standard", "Cluster", "Diffs", "Embeds","Label", "Group"])

          full_PCA = PCA(n_components=2)
          fit = full_PCA.fit_transform(k_df[args.cluster_element].tolist())
          k_df["x"] = [f[0] for f in fit]
          k_df["y"] = [f[1] for f in fit]

          legend_labels = []
          cluster_pures = []
          cluster_accs = 0
          std_ob_accs = 0
          clusters = []


          for c_name, c_group in k_df.groupby("Cluster"):
               c_group = c_group.reset_index(drop=True)
               cluster = {l: {"tokens":[]} for l in args.label_set}
               #print(c_group)
               #cluster purity method: how well does each K gather known nontargets (std, rev, ocr labels) into single clusters
               #avg. purity: less mixed clusters are on avg. the better

               cluster_counts = [c_group["Label"].str.count(inc).sum() for inc in args.label_set]
               for l,c in zip(args.label_set, cluster_counts):
                    cluster[l]["counts"] = c.item()
                    cluster[l]["percent"] = c.item()/sum(cluster_counts)
               cluster_counts = sorted(cluster_counts, reverse=True)
               #cluster_pures.append( (cluster_counts[0]/sum(cluster_counts)) * (sum(cluster_counts)/len(k_df)))
               cluster_pures.append(cluster_counts[0])
               legend_labels.append(str(len(c_group)) + ": "+ " ".join(c_group["Token"][0:3]))

               for g_name, g_group in c_group.groupby("Group"):
                    if g_group["Label"].count() == 2:
                         if g_group["Label"].str.contains("std").sum() > 0 and g_group["Label"].str.contains("obv").sum() > 0:
                              std_ob_accs += 1
                    if g_group["Label"].count() == len(args.label_set):
                         cluster_accs += 1

               cluster["acc"] = cluster_accs/len(c_group)
               cluster["std_obs_acc"] = std_ob_accs/len(c_group)
               for t_name, t_group in c_group.groupby("Label"):
                    cluster[t_name]["tokens"] = [{"token": t, "group": g, "standard": s} for t,g,s in zip(t_group["Token"].tolist(), t_group["Group"].tolist(), t_group["Standard"].tolist())]
               clusters.append(cluster)
            
          #print(k_df.assign(Cluster=k_df["Cluster"].map(lambda x: legend_labels[x])))
          ax = sns.scatterplot(x="x", y="y", hue="Cluster", data=k_df.assign(Cluster=k_df["Cluster"].map(lambda x: legend_labels[x])), legend="full")
          #sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, 1.35), ncol=3, title=None, frameon=False)
          
          avg_accs.append(cluster_accs/(k_df["Group"].count()/len(args.label_set)))
          avg_std_obv_accs.append(std_ob_accs/(k_df["Group"].count()/len(args.label_set)))
          #avg_cps.append(sum(cluster_pures))#/len(cluster_pures))
          avg_cps.append(sum(cluster_pures)/len(k_df))
          plt.savefig(args.cluster_out+"kpca"+str(k)+args.cluster_element+".png", format="png")
          k_df.to_csv(args.cluster_out+"kpca"+str(k)+".csv", columns=["Token","Cluster","Label","Group"], index=False)
          with open(args.cluster_out+"details"+str(k)+".jsonl", "wt") as jout:
               for c in clusters:
                    jout.write(json.dumps(c)+"\n")
          plt.clf()
     
     plt.plot(ks, inertias, "bx-")
     plt.xticks(np.arange(min(ks), max(ks)+1, 1))
     plt.xlabel("K")
     plt.ylabel("Inertia")
     plt.savefig(args.outfile, format="png")
     
     plt.clf()
     plt.plot(ks, avg_cps, "bx-")
     plt.xticks(np.arange(min(ks), max(ks)+1, 1))
     plt.xlabel("K")
     plt.ylabel("Purity")
     plt.savefig(args.purity, format="png")


     plt.clf()
     plt.plot(ks, avg_accs, "bx-")
     plt.xticks(np.arange(min(ks), max(ks)+1, 1))
     plt.xlabel("K")
     plt.ylabel("Avg. Acc")
     plt.savefig(args.distincts_out, format="png")

     plt.clf()
     plt.plot(ks, avg_accs, "bx-")
     plt.xticks(np.arange(min(ks), max(ks)+1, 1))
     plt.xlabel("K")
     plt.ylabel("Avg. S/O Acc")
     plt.savefig(args.so_acc, format="png")
