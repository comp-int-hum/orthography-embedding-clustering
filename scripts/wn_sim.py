import argparse
import glob
import gzip
import json
import numpy as np
import pickle


import pandas as pd

import jellyfish

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--k_csv", help="csv with cluster info")
     parser.add_argument("--sim_out", help="CSV similarity/cluster")
     parser.add_argument("--comp_table")
     args, rest = parser.parse_known_args()

     cluster_scores = []
     dm_scores = []
     c_no = []

     with open(args.comp_table, "rt") as c_in:
          c_l = json.load(c_in)
     
     k_df = pd.read_csv(args.k_csv)
     with open(args.sim_out, "wt") as e_out:
          for gn, g in k_df.groupby("Cluster"):

                cs = []
                dms = []
                for n, (tok, sid, dm) in enumerate(zip(g["Token"], g["SID"], g["DM1"])):
                     comps = []
                     dcomps = []
                     for i, (t2, s2, dm2) in enumerate(zip(g["Token"], g["SID"], g["DM1"])):
                          if n != i:
                               try:
                                    dcomp = jellyfish.levenshtein_distance(dm, dm2)
                               except:
                                    dcomp = None
                               if dcomp != None:
                                    dcomps.append(dcomp)
                               try:
                                    comp = c_l[str(sid)][tok.replace("’","'").replace("‘","'")][str(s2)][t2.replace("’","'").replace("‘","'")]
                               except:
                                    comp = None
                               if comp != None:
                                    comps.append(comp)
                     if len(comps) > 0:
                          cs.append(sum(comps)/len(comps))
                     if len(dcomps) > 0:
                          dms.append(sum(dcomps)/len(dcomps))
                if len(dms) > 0:          
                     dm_scores.append(sum(dms)/len(dms))
                else:
                     dm_scores.append(0)
                if len(cs) > 0:
                     cluster_scores.append(sum(cs)/len(cs))
                else:
                     cluster_scores.append(0)
                c_no.append(int(gn))

     o_df = pd.DataFrame({"Cluster": c_no, "WV_Avg_Sim": cluster_scores, "DM_Avg_Ld":dm_scores})
     print(o_df)
     o_df.to_csv(args.sim_out, index=False)
     
