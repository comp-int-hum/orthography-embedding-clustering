import argparse
import glob
import gzip
import json
import numpy as np
import pickle

from nltk.corpus import wordnet as wn

import pandas as pd
from collections import defaultdict
from gensim.models import Word2Vec


if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--wv_model", help="trained wv model")
     parser.add_argument("--full_sc", help="full corpus sents file")
     parser.add_argument("--wv_out", help="WV comparisons")
     parser.add_argument("--wn_out", help="Wordnet comparisons")
     args, rest = parser.parse_known_args()

     wv = Word2Vec.load(args.wv_model)


     wv_lookup = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
     wn_lookup = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

     ss_lookup = {}
     with open(args.full_sc, "rt") as sc_in:
          for line in sc_in:
               j_l = json.loads(line)
               ss_lookup[j_l["sample_id"]] = {i["Std"]: [i["syn"], w.replace("’","'").replace("‘","'").lower()] for w,i in j_l["words"].items()}

     for sid, item in ss_lookup.items():
          for word, ss in item.items():
               for sid2, item2 in ss_lookup.items():
                    for word2, ss2 in item2.items():
                         print(ss, ss2)
                         if ss[0] != "" and ss2[0] != "":
                              wn_lookup[sid][word][sid2][word2] = wn.synset(ss[0]).path_similarity(wn.synset(ss2[0]))
                         else:
                              wn_lookup[sid][word][sid2][word2] = None
                         try:
                              wv_lookup[sid][ss[1]][sid2][ss2[1]] = wv.wv.similarity(ss[1], ss2[1]).item()
                         except:
                              wv_lookup[sid][ss[1]][sid2][ss2[1]] = None

with open(args.wv_out, "wt") as wvo, open(args.wn_out, "wt") as wno:
     json.dump(wv_lookup, wvo)
     json.dump(wn_lookup, wno)
                         
