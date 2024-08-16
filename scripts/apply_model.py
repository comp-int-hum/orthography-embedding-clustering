import argparse
import glob
import gzip
import json
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import pickle

from gensim.models import FastText

import torch
from collections import defaultdict
import nlpaug.augmenter.char as nac

def dist(a,b):
     return a - b
     #return cosine(a,b)

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--datasets", nargs="+", default=["../llm-direct-embeddings/work/GB_0_2/bert-large-uncased/embeds/chunk_embed_custom_0_*.json.gz"], help="Dataset stem")
     parser.add_argument("--outfile", help="gzjson of embeddings from particular model" )
     parser.add_argument("--model", help="Trained model path")
     parser.add_argument("--ft_pt", default=0, type=int, help="Is model pretrained fasttext?")

     args, rest = parser.parse_known_args()

     aug = nac.OcrAug()
     print(args.ft_pt)
     if args.ft_pt == True:
          import fasttext.util
          import fasttext
          fasttext.util.download_model('en', if_exists='ignore')  # English
          ft = fasttext.load_model('cc.en.300.bin')
     elif args.ft_pt == False:
          model = FastText.load(args.model)
          
          
     y = []
     standards = []
     x_diffs = []
     x_embeds = []
     labels = []
     dtags = []
     g_ids = []
     for d_stem in args.datasets:
         for dataset in glob.glob(d_stem):
              print(dataset)
              with gzip.open(dataset, "rt") as d_in:
                   for line in d_in:
                        js_line = json.loads(line)
                        for ann in js_line["annotations"]:
                             if ann["observed"]:
                                  
                                  y += [ann["standard"], ann["observed"], ann["rev_token"], ann["ocr_token"], ann["swap_token"], ann["rand_token"]]
                                  labels += ["std","obv","rev","ocr","swp", "rnd"]
                                  standards += [ann["standard"]]*6
                                  dtags += ["std", ann["dtag"], "rev", "ocr", "swp", "rnd"]
                                  g_ids += [js_line["g_id"]]*6

                                  if args.ft_pt == False:
                                       x_embeds += [model.wv[ann["standard"]],
                                               model.wv[ann["observed"]],
                                               model.wv[ann["rev_token"]],
                                               model.wv[ann["ocr_token"]],
                                               model.wv[ann["swap_token"]],
                                               model.wv[ann["rand_token"]]]     

                                       x_diffs += [dist(x_embeds[0],x_embeds[0]), dist(x_embeds[0], x_embeds[1]),dist(x_embeds[0], x_embeds[2]), dist(x_embeds[0], x_embeds[3]),
                                                   dist(x_embeds[0], x_embeds[4]),dist(x_embeds[0], x_embeds[5])]      

                                  elif args.ft_pt == True:
                                       x_embeds += [ft.get_word_vector(ann["standard"]),
                                               ft.get_word_vector(ann["observed"]),
                                               ft.get_word_vector(ann["rev_token"]),
                                               ft.get_word_vector(ann["ocr_token"]), ft.get_word_vector(ann["swap_token"]), ft.get_word_vector(ann["rand_token"])]

                                       x_diffs += [dist(x_embeds[0],x_embeds[0]),
                                               dist(x_embeds[0],x_embeds[1]),
                                               dist(x_embeds[0],x_embeds[2]),
                                               dist(x_embeds[0],x_embeds[3]),
                                               dist(x_embeds[0],x_embeds[4]),
                                               dist(x_embeds[0],x_embeds[5])]
                                       
                                 

     with open(args.outfile, "wb") as of:
          pickle.dump({"y":y, "standards": standards, "x_embeds": x_embeds, "x_diffs": x_diffs, "labels": labels, "dtags": dtags, "gids": g_ids}, of)
