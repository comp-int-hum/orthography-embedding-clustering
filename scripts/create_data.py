
import argparse
import re
import csv
import gzip
import json

import os
import glob

import spacy
from string import punctuation
import logging


def parse_gb_directory(base_dir, text_num):
        path_elements = "/".join([c for c in text_num[:-1]])
        return os.path.join(base_dir, path_elements, text_num)


def retokenize(orig_tok):
        found_prefix = None
        new_toks = []
        for t in orig_tok:
                if t[0] == "SUFFIX" and t[1] in ["’","'","‘"]:
                        new_toks[-1] = new_toks[-1]+t[1]
                elif t[0] == "PREFIX" and t[1] in ["’","'","‘"]:
                        found_prefix = t[1]
                else:
                        if found_prefix:
                                new_toks.append(found_prefix+t[1])
                                found_prefix = None
                        else:
                                new_toks.append(t[1])
        return new_toks


def paragraph_chunk(full_text):
        for c in full_text.split("\n\n"):
                yield c

nlp = spacy.load("en_core_web_trf", exclude=["ner"])

#remove special rules concerning apos
nlp.tokenizer.rules = {key: value for key, value in nlp.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}

log_format = "%(asctime)s::%(filename)s::%(message)s"
logging.basicConfig(level='INFO', format=log_format)


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("metadata_csv", help="Gutenberg metadata CSV")
        parser.add_argument("outfile", help="Outfile text gzipped one sent/line")
        parser.add_argument("gutenberg_directory", help="Base Gutenberg corpus directory")
        parser.add_argument("--efile", default="error.csv", help="Error reporting")

        args, rest = parser.parse_known_args()

        with open(args.metadata_csv, "r", encoding="utf-8") as metadata_in, gzip.open(args.outfile, "wt") as text_out, open(args.efile, "wt", newline="") as e_out:
                m_reader = csv.DictReader(metadata_in)
                for l in m_reader:
                        text_dir = parse_gb_directory(args.gutenberg_directory, l["Text#"])
                        for t_part in glob.glob(text_dir+"/*.txt"):
                                logging.info("Textfile: %s", t_part)
                                logging.info("GID: %s", l["Text#"])
                                try:
                                        with open(t_part, "r") as text_in:

                                                text = text_in.read()


                                                for paragraph in paragraph_chunk(text.lower()):
                                                        for sent in nlp(paragraph).sents:
                                                                r_t = retokenize(nlp.tokenizer.explain(sent.text))
                                                                r_t = [t.replace("’","'").replace("‘","'") for t in r_t if any([c.isalpha() for c in t])]
                                                                if len(r_t) == 0:
                                                                        continue
                                                                if any([True if t in ["gutenberg","ebook"] else False for t in r_t]):
                                                                        continue
                                                                text_out.write(" ".join(r_t) + "\n")
                                except:
                                        logging.info("Error: %s", t_part)
                                        e_out.write(t_part+"\n")
