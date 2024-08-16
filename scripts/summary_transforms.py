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
     parser.add_argument("--transform_jsons", nargs="+", help="List of jsonl transform information")
     #parser.add_argument("--summary_out", help="CSV of summary statistics")
     parser.add_argument("--tag_transforms", help="Tag transforms out")


     args, rest = parser.parse_known_args()

     ks = []
     tag_transforms = []
     
     for k, t_json in enumerate(args.transform_jsons):
          ks.append(k+1)
          pair_count = defaultdict(lambda: defaultdict(int))
          with open(t_json, "rt") as j_in:
               for c_no, line in enumerate(j_in):
                    j_line = json.loads(line)
                    for dtag, transforms in zip(j_line["dtags"], j_line["transform_detail"]):
                         pair_count[str(dtag)+": ".join(sorted(transforms))][c_no]+=1
          print(pair_count)
          pair_spread = 0
          for pair, counts in pair_count.items():
               pair_spread += len(counts)
          tag_transforms.append(pair_spread/len(pair_count))
                         

     #d = np.array([ks, purities, dtag_purity, avg_accs, avg_std_obv_accs, dtag_concentration]).T.tolist()
     #o_df = pd.DataFrame(data=d, columns = ["K", "Purity", "Dtag_purity",  "Avg_Acc", "Avg_SO_Acc", "D_conc"])
     #o_df.to_csv(args.summary_out)          
     
     plt.plot(ks, tag_transforms, "bx-")
     plt.xticks(np.arange(min(ks), max(ks), 1))
     plt.xlabel("K")
     plt.ylabel("Average dtag+transformation pairs/cluster")
     plt.savefig(args.tag_transforms, format="png")

   
