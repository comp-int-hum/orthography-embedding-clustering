import argparse
import glob
import gzip
import json
import numpy as np
import pickle

from nltk.corpus import wordnet as wn

import pandas as pd


if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--k_csv", help="csv with cluster info")
     parser.add_argument("--comp_table")
     parser.add_argument("--sim_out", help="CSV similarity/cluster")
     args, rest = parser.parse_known_args()

     cluster_scores = []
     c_no = []


     with open(args.comp_table, "rt") as c_in:
          c_t = json.load(c_in)
     
     k_df = pd.read_csv(args.k_csv)
     with open(args.sim_out, "wt") as e_out:
          for gn, g in k_df.groupby("Cluster"):
               ss_s = []
               comps = []
               for i, (tok, sid) in enumerate(zip(g["Standard"], g["SID"])):
                    for n, (t2, s2) in enumerate(zip(g["Standard"],g["SID"])):
                         if i != n:
                              comp = c_t[str(sid)][tok][str(s2)][t2]
                              if comp != None:
                                   comps.append(comp)
               if len(comps) > 0:
                    cluster_scores.append(sum(comps)/len(comps))
               else:
                    cluster_scores.append(0)
               c_no.append(int(gn))
          o_df = pd.DataFrame({"Cluster": c_no, "Avg_Sim": cluster_scores})
          print(o_df)
          o_df.to_csv(args.sim_out, index=False)

               
"""
                   ss_s.append(ss_lookup[sid][tok])

               syns = [wn.synset(ss) for ss in ss_s if ss != ""]
               if len(syns) > 1:
                    for n, s_set in enumerate(syns):
                         for i, set_comp in enumerate(syns):
                              if n != i:
                                   comps.append(s_set.path_similarity(set_comp))
                    cluster_scores.append(sum(comps)/len(comps))
                    c_no.append(int(gn))
                                   
          o_df = pd.DataFrame({"Cluster": c_no, "Avg_Sim": cluster_scores})
          print(o_df)
          o_df.to_csv(args.sim_out, index=False)
                              



                    
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
"""
