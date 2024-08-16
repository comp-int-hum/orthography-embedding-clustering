import argparse
import glob
import gzip
import json
import numpy as np
import pickle



import pandas as pd


from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--k_csv", help="csv with cluster info")
     parser.add_argument("--sim_out", help="CSV similarity/cluster")
     args, rest = parser.parse_known_args()

     cluster_scores = []
     c_no = []

     lem = WordNetLemmatizer()
     
     k_df = pd.read_csv(args.k_csv)
     with open(args.sim_out, "wt") as e_out:
          for gn, g in k_df.groupby("Cluster"):
               syns = []
               for tok in g["Standard"]:
                    ss = wn.synsets(tok)
                    if len(ss) == 0:
                         ss = wn.synsets(lem.lemmatize(tok))
                    if len(ss) > 0:
                         syns.append(ss)
               c_scores = []
               if len(syns) > 1:
                    for n, s_set in enumerate(syns):
                         comps = []
                         for i, set_comp in enumerate(syns):
                              if n != i:
                                   for syn in s_set:
                                        max_score = 0
                                        for syn_comp in set_comp:
                                             sc =  syn.path_similarity(syn_comp)
                                             if sc > max_score:
                                                  max_score = sc
                                   comps.append(max_score)
                         c_scores.append(sum(comps)/len(comps))
                    cluster_scores.append(sum(c_scores)/len(c_scores))
                    c_no.append(int(gn))
                    print(c_no)
                    print(cluster_scores)
     o_df = pd.DataFrame({"Cluster": c_no, "Avg_Sim": cluster_scores})
     print(o_df)
     o_df.to_csv(args.sim_out, index=False)
