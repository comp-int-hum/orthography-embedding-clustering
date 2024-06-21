import argparse
import glob
import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--to_analyze", nargs="+", help="Summary CSVS")
     parser.add_argument("--model_names", nargs="+", help="Model names")
     parser.add_argument("--element", help="element to chart")
     parser.add_argument("--outfile", help="PNG outfiles")

     
     args, rest = parser.parse_known_args()

     for model, model_n in zip(args.to_analyze, args.model_names):
          k_df = pd.read_csv(model,  dtype={"K": "Int32"})
          print(k_df)
          ser = k_df[args.element].tolist()
          xs = k_df["K"].tolist()
          plt.plot(xs, ser, label=model_n)

     plt.xticks(xs)
     plt.xlabel("K")
     plt.ylabel(args.element.replace("_"," "))
     plt.legend()
     plt.savefig(args.outfile, format="png")
     
          

          

     
