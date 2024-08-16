import argparse
import glob
import gzip
import json
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity
import fasttext.util
import fasttext

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


     fasttext.util.download_model('en', if_exists='ignore')
     ft = fasttext.load_model('cc.en.300.bin')
     
     k_df = pd.read_csv(args.k_csv)
     with open(args.sim_out, "wt") as e_out:
          for gn, g in k_df.groupby("Cluster"):
               if len(g) > 1:
                    cs = []
                    for n, tok in enumerate(g["Standard"]):
                         comps = []
                         for i, t2 in enumerate(g["Standard"]):
                              if n != i:
                                   comps.append(cosine_similarity([ft.get_word_vector(tok)], [ft.get_word_vector(t2)]).item())
                         cs.append(sum(comps)/len(comps))
                    cluster_scores.append(sum(cs)/len(cs))
                    c_no.append(gn)

     o_df = pd.DataFrame({"Cluster": c_no, "FT_Avg_Sim": cluster_scores})
     print(o_df)
     o_df.to_csv(args.sim_out, index=False)
