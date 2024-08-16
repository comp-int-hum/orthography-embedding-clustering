import argparse
import glob
import gzip
import json
import numpy as np
import pandas as pd
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from collections import defaultdict


def dist(a, b):
     return a-b

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--datasets", nargs="+", default=["../llm-direct-embeddings/work/GB_0_2/bert-large-uncased/embeds/chunk_embed_custom_0_*.json.gz"], help="Dataset stem")
     parser.add_argument("--outfile", help="gzjson of embeddings from particular model" )
     parser.add_argument("--model", help="Model name")

     args, rest = parser.parse_known_args()
     layers = ""
     print(args.model)
     if args.model == "google/canine-c":
          layers="last"
          args.datasets = ["../llm-direct-embeddings/work/GB_0_3/google/canine-c/embeds/chunk_embed_custom_0_*.json.gz"]
     elif args.model == "google/canine-s":
          layers="last"
          args.datasets = ["../llm-direct-embeddings/work/GB_0_3/google/canine-s/embeds/chunk_embed_custom_0_*.json.gz"]
     elif args.model == "bert-base-uncased":
          layers="last_four"
          args.datasets = ["../llm-direct-embeddings/work/GB_0_3/bert-base-uncased/embeds/chunk_embed_custom_0_*.json.gz"]
     elif args.model == "bert_forced":
          layers="last_four"
          args.datasets = ["../llm-direct-embeddings/work/GB_0_3/bert_forced/embeds/chunk_embed_custom_0_*.json.gz"]
     else:
          layers="last_four"
          

     n_anns = 0
     obv = 0
     y = []
     standards = []
     x_diffs = []
     x_embeds = []
     labels = []
     dtags = []
     gids = []
     for d_stem in args.datasets:
         for dataset in glob.glob(d_stem):
              print(dataset)
              with gzip.open(dataset, "rt") as d_in:
                   for line in d_in:
                        js_line = json.loads(line)
                        for ann in js_line["annotations"]:
                             n_anns += 1
                             if ann["observed"]:
                                  
                                  obv+=1
                                  y += [ann["standard"], ann["observed"], ann["rev_token"], ann["ocr_token"], ann["swap_token"], ann["rand_token"]]
                                  labels += ["std","obv","rev","ocr","swp","rnd"]
                                  standards += [ann["standard"]]*6

                                  x_embeds += [np.mean(np.array(ann["standard_embeddings"][layers]), axis=0),
                                               np.mean(np.array(ann["observed_embeddings"][layers]), axis=0),
                                               np.mean(np.array(ann["reverse_embeddings"][layers]), axis=0),
                                               np.mean(np.array(ann["error_embeddings"][layers]), axis=0),
                                               np.mean(np.array(ann["swap_embeddings"][layers]), axis=0),
                                               np.mean(np.array(ann["rand_embeddings"][layers]), axis=0)
                                               ]

                                  x_diffs += [dist(x_embeds[-6],x_embeds[-6]),dist(x_embeds[-6],x_embeds[-5]),dist(x_embeds[-6], x_embeds[-4]),dist(x_embeds[-6],x_embeds[-3]), dist(x_embeds[-6],x_embeds[-2]), dist(x_embeds[-6],x_embeds[-1])]
                                  dtags += ["std", ann["dtag"], "rev", "ocr", "swp", "rnd"]
                                  gids += [js_line["g_id"]]*6
                                 

     with open(args.outfile, "wb") as of:
          pickle.dump({"y":y, "standards": standards, "x_embeds": x_embeds, "x_diffs": x_diffs, "labels":labels, "dtags": dtags, "gids": gids}, of)
     print(n_anns)
     print(obv)
     
          
