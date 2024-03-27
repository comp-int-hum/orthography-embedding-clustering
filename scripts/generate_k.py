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
     parser.add_argument("--purity_out", help="Purity chart out")
     parser.add_argument("--k", nargs=2, type=int, default=[1,150], help="K cluster range")
     parser.add_argument("--cluster_out", default="pngs/")

     args, rest = parser.parse_known_args()

     with open(args.embeds, "rb") as e_in:
          embeds = pickle.load(e_in)
          x_embeds = embeds["x_embeds"]
          x_diffs = embeds["x_diffs"]
          y = embeds["y"]
          standards = embeds["standards"]
          labels = embeds["labels"]

     inertias = []
     ks = []
     avg_cps = []

     
     for k in range(args.k[0], args.k[1]):
          print("K: "+str(k))
          kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(np.array(x_diffs))
          inertias.append(kmeans.inertia_)
          print("Inertia: "+str(inertias[-1]))
          ks.append(k)

          
          #kmeans = KMeans(n_clusters=args.k, random_state=0, n_init="auto").fit(np.array(x_diffs).reshape(-1, 1))
          k_df = pd.DataFrame(zip(y, standards, kmeans.labels_, x_diffs, x_embeds, labels), columns = ["Token", "Standard", "Cluster", "Diffs", "Embeds","Label"])

          full_PCA = PCA(n_components=2)
          fit = full_PCA.fit_transform(k_df["Diffs"].tolist())
          k_df["x"] = [f[0] for f in fit]
          k_df["y"] = [f[1] for f in fit]

          #cluster purity method: how well does each K gather known nontargets (std, rev, ocr labels) into single clusters
          #avg. purity: less mixed clusters are on avg. the better
          legend_labels = []
          cluster_pures = []
          for c_name, c_group in k_df.groupby("Cluster"):
               c_group = c_group.reset_index(drop=True)
               cluster_counts = sorted([
                    c_group["Label"].str.count("std").sum(),
                    c_group["Label"].str.count("ocr").sum(),
                    c_group["Label"].str.count("rev").sum(),
                    c_group["Label"].str.count("obv").sum(),
               ], reverse=True)
               cluster_pures.append( cluster_counts[0]/sum(cluster_counts))
               legend_labels.append(str(len(c_group)) + ": "+ " ".join(c_group["Token"][0:3]))
          #print(k_df.assign(Cluster=k_df["Cluster"].map(lambda x: legend_labels[x])))
          ax = sns.scatterplot(x="x", y="y", hue="Cluster", data=k_df.assign(Cluster=k_df["Cluster"].map(lambda x: legend_labels[x])), legend="full")
          #sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, 1.35), ncol=3, title=None, frameon=False)
          #pos = ax.get_position()
          #ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.75])
          #plt.legend(title="Clusters", loc="upper center", labels=legend_labels, bbox_to_anchor=(0.5, 1.35), ncol=3)
          avg_cps.append(sum(cluster_pures)/len(cluster_pures))
          plt.savefig(args.cluster_out+"kpca"+str(k)+".png", format="png")
          k_df.to_csv(args.cluster_out+"kpca"+str(k)+".csv", columns=["Token","Cluster","Label"], index=False)
          plt.clf()
     
     plt.plot(ks, inertias, "bx-")
     plt.xticks(np.arange(min(ks), max(ks)+1, 1))
     plt.xlabel("K")
     plt.ylabel("Inertia")
     #fig, ax = plt.subplots()
     #for i, k in enumerate(ks):
     #     ax.annotate(str(k), (ks[i], inertias[i]))
     plt.savefig(args.outfile, format="png")
     
     plt.clf()
     plt.plot(ks, avg_cps, "bx-")
     plt.xticks(np.arange(min(ks), max(ks)+1, 1))
     plt.xlabel("K")
     plt.ylabel("Avg. Purity")
     plt.savefig(args.purity_out, format="png")

     #for ob_name, ob_group in k_df.groupby("Standard"):
     #     ob_group = ob_group.reset_index(drop=True)
     #     ax = sns.scatterplot(x="x", y="y", hue="Cluster", data=ob_group)
     #     for line in range(0,ob_group.shape[0]):
     #          plt.text(ob_group.x[line]+0.01, ob_group.y[line], 
     #                   str(ob_group.Token[line]), horizontalalignment="left", 
     #                   size="small", color="black")
     #     plt.savefig("pngs/"+str(args.k)+"/"+ob_group["Standard"].iloc[0]+".png", format="png")
     #     plt.clf()


