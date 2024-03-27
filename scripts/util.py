
import glob
import gzip
import json

class ChunkCorpus(object):
    def __init__(self, chunk_stem):
        self.chunk_stem = chunk_stem

    def __iter__(self):
        stem_glob = glob.glob(self.chunk_stem)
        for df in stem_glob:
            with gzip.open(df, "rt") as d_in:
                for line in d_in:
                    js_line = json.loads(line)
                    #for annotation in js_line["annotations"]:
                        

                    yield js_line["text"]



class GutGZCorpus(object):
    def __init__(self, filename):
        self.filename=filename

    def __iter__(self):
        with gzip.open(self.filename, "rt") as d_in:
            for line in d_in:
                print(json.loads(line))

                    
