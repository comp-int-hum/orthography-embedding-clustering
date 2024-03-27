import argparse
from gensim.models import Word2Vec, FastText
import gzip

class GzipIter(object):
    def __init__(self, gzipf):
        self.gzipf = gzip.open(gzipf, "rt")

    def __iter__(self):
        self.gzipf.seek(0)
        for line in self.gzipf:
            if len(line.split()) < 1:
                continue
            yield line.split()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--embed_width", type=int, help="Embedding size")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    args, rest = parser.parse_known_args()


    
    model = FastText(sentences=GzipIter(args.dataset), vector_size=args.embed_width, window=5, min_count=1, workers=4, negative=10)
    model.save(args.outputs[0])
    

