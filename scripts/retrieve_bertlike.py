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
import nlpaug.augmenter.char as nac

def dist(a, b):
     return a-b

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--datasets", nargs="+", default=["../llm-direct-embeddings/work/GB_0_2/bert-large-uncased/embeds/chunk_embed_custom_0_*.json.gz"], help="Dataset stem")
     parser.add_argument("--outfile", help="gzjson of embeddings from particular model" )
     parser.add_argument("--model", help="Model name")

     args, rest = parser.parse_known_args()

     aug = nac.OcrAug()
     layers = ""
     print(args.model)
     if args.model == "google/canine-c":
          layers="last"
          args.datasets = ["../llm-direct-embeddings/work/GB_0_2/google/canine-c/embeds/chunk_embed_custom_0_*.json.gz"]
     else:
          layers="last_four"
          

     n_anns = 0
     obv = 0
     y = []
     standards = []
     x_diffs = []
     x_embeds = []
     labels = []
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
                                  y += [ann["standard"], ann["observed"], ann["standard"][::-1], aug.augment(ann["standard"])[0]]
                                  labels += ["std","obv","rev","ocr"]
                                  standards += [ann["standard"]]*4

                                  x_embeds += [np.mean(np.array(ann["standard_embeddings"][layers]), axis=0),
                                               np.mean(np.array(ann["observed_embeddings"][layers]), axis=0),
                                               np.mean(np.array(ann["reverse_embeddings"][layers]), axis=0),
                                               np.mean(np.array(ann["error_embeddings"][layers]), axis=0)
                                               ]

                                  x_diffs += [dist(x_embeds[-4],x_embeds[-4]),dist(x_embeds[-4],x_embeds[-3]),dist(x_embeds[-4], x_embeds[-2]),dist(x_embeds[-4],x_embeds[-1])]
                                       
                                 

     with open(args.outfile, "wb") as of:
          pickle.dump({"y":y, "standards": standards, "x_embeds": x_embeds, "x_diffs": x_diffs, "labels":labels}, of)
     print(n_anns)
     print(obv)
     
          
