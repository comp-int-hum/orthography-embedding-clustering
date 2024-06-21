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
import difflib
import jellyfish



if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--edits_in", help="jsonl edits file")
     parser.add_argument("--n_edits", default=5, type=int, help="n most common edits/cluster")
     parser.add_argument("--outfile", help="png chart")
     args, rest = parser.parse_known_args()

     sns.set_palette("muted")

     cluster_transforms = []
     cluster_distinct_transforms = []
     with open(args.edits_in, "rt") as e_i:
          for line in e_i:
               cluster_transforms.append(json.loads(line)["transforms"])
               cluster_distinct_transforms.append(len(cluster_transforms[-1])) 

     t_df = pd.DataFrame(cluster_transforms).apply(lambda s, n: s.nlargest(n)/s.sum(), axis=1, n=args.n_edits)
     print(t_df)

     fig, ax = plt.subplots(figsize=(12, 7))
     t_df.plot(kind="bar", stacked=True, ax=ax)
     ax.set_ylim(0,1.0)
     #for c,l in zip(ax.containers, cluster_distinct_transforms):
     #     ax.bar_label(c, labels=[l], label_type="edge")
     ax.set_xticklabels([str(c+1) + ": " + str(dt) for c, dt in enumerate(cluster_distinct_transforms)])
     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
     plt.xlabel("Cluster: N distinct transformations")
     plt.ylabel("Transformation percentages")
     plt.title("Top " + str(args.n_edits) +  " transformations by cluster")
     plt.savefig(args.outfile)

