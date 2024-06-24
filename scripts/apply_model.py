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
     for d_stem in args.datasets:
         for dataset in glob.glob(d_stem):
              print(dataset)
              with gzip.open(dataset, "rt") as d_in:
                   for line in d_in:
                        js_line = json.loads(line)
                        for ann in js_line["annotations"]:
                             if ann["observed"]:
                                  ocr_tok = aug.augment(ann["standard"])[0]
                                  swap_tok = ann["standard"][:2][::-1]+ann["standard"][2:]
                                  
                                  y += [ann["standard"], ann["observed"], ann["standard"][::-1], ocr_tok, swap_tok]
                                  labels += ["std","obv","rev","ocr","swp"]
                                  standards += [ann["standard"]]*5

                                  if args.ft_pt == False:
                                       x_embeds += [model.wv[ann["standard"]],
                                               model.wv[ann["observed"]],
                                               model.wv[ann["standard"][::-1]],
                                               model.wv[ocr_tok],
                                               model.wv[swap_tok]]

                                       x_diffs += [dist(x_embeds[0],x_embeds[0]), dist(model.wv[ann["standard"]], model.wv[ann["observed"]]),
                                              dist(model.wv[ann["standard"]], model.wv[ann["standard"][::-1]]),
                                                   dist(model.wv[ann["standard"]], model.wv[ocr_tok]), dist(model.wv[ann["standard"]], model.wv[swap_tok])]
                                  elif args.ft_pt == True:
                                       x_embeds += [ft.get_word_vector(ann["standard"]),
                                               ft.get_word_vector(ann["observed"]),
                                               ft.get_word_vector(ann["standard"][::-1]),
                                               ft.get_word_vector(ocr_tok), ft.get_word_vector(swap_tok)]

                                       x_diffs += [dist(x_embeds[-4],x_embeds[-4]), dist(ft.get_word_vector(ann["standard"]), ft.get_word_vector(ann["observed"])),
                                              dist(ft.get_word_vector(ann["standard"]), ft.get_word_vector(ann["standard"][::-1])),
                                              dist(ft.get_word_vector(ann["standard"]), ft.get_word_vector(ocr_tok)),
                                              dist(ft.get_word_vector(ann["standard"]), ft.get_word_vector(swap_tok))]
                                       
                                 

     with open(args.outfile, "wb") as of:
          pickle.dump({"y":y, "standards": standards, "x_embeds": x_embeds, "x_diffs": x_diffs, "labels": labels}, of)
