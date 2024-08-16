import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
import steamroller
import glob
from collections import defaultdict

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

# Variables control various aspects of the experiment.  Note that you have to declare
# any variables you want to use here, with reasonable default values, but when you want
# to change/override the default values, do so in the "custom.py" file (see it for an
# example, changing the number of folds).
vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("MODEL_TYPES", "", ["fasttext_pretrained", "google/canine-c", "bert-large-uncased", "bert-base-uncased", "google/canine-s", "bert_forced"]),
    ("PRETRAINED_LOC","","cc.en.300.bin"),
    ("PARAMETER_VALUES", "", {"trained_300": [300], "trained_128": [128]}),
    ("K","",[1,20]),
    ("DATASET_NAMES", "", ["GB19"]),
    ("METADATA","", ["../gutenberg-ns-extractor/pg_19cam.csv"]),
    ("GUT_DIR", "", ["../../../export/corpora/gutenberg/"]),
    ("SENT_DATASET_STEMS", "", ["../llm-direct-embeddings/work/GB_0_3/bert-large-uncased/embeds/chunk_embed_custom_0_*.json.gz"]),
    ("FOLDS", "", 1),
    ("CLUSTER_ELEMENTS","",["Diffs","Embeds"]),
    ("N_TRANSFORMS","",3),
    ("LABEL_SETS", "", [["std","rev","ocr","obv","swp","rnd"],["std","rev","ocr","obv","swp"],["std","rev","ocr","obv"], ["rev","ocr","obv"], ["rev","ocr","obv","swp"], ["ocr","obv"], ["std","ocr","obv"], ["obv"], ["swp","obv","ocr"], ["rnd","obv","ocr"], ["rnd","obv"], ["std","obv"]]),
    ("ANALYZE_MODELS", "", ["bert-large-uncased_1", "fasttext_pretrained_1", "google/canine-c_1", "bert-base-uncased_1", "google/canine-s_1", "bert_forced_1"]),
    ("STD_APOS", "", False),
    ("USE_GRID","", False)
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[steamroller.generate],

    BUILDERS={
        "CreateData" : Builder(
            action="python scripts/create_data.py ${METADATA} ${TARGETS[0]} ${DIR}"
        ),
        "TrainModel" : Builder(
            action="python scripts/train_model.py --dataset ${SOURCES[0]} --outputs ${TARGETS[0]} --embed_width ${EMBED_WIDTH}"
        ),
        "ShuffleData" : Builder(
            action="python scripts/shuffle_data.py --dataset ${SOURCES[0]} --outputs ${TARGETS}"
        ),
        "ApplyModel" : Builder(
            action="python scripts/apply_model.py --model ${SOURCES[0]} --datasets ${SOURCE_DATASET} --outfile ${TARGETS[0]} --ft_pt ${PRETRAINED}"
        ),
        "RetrieveBertlike" : Builder(
            action="python scripts/retrieve_bertlike.py --model ${MODEL_TYPE} --datasets ${SOURCE_DATASET} --outfile ${TARGETS[0]}"
        ),
        "GenerateK": Builder(
            action="python scripts/generate_k.py --embeds ${SOURCES[0]} --outfile ${TARGETS[0]} --purity ${TARGETS[1]} --distincts_out ${TARGETS[2]} --so_acc ${TARGETS[3]} --k ${K} --cluster_element ${CLUSTER_ELEMENT} --label_set ${LS} --cluster_out ${CLUSTER_OUT}"
        ),
	"Cluster": Builder(action="python scripts/cluster_k.py --embeds ${SOURCES[0]} --outfile ${TARGETS[0]} --k ${K} --cluster_element ${CLUSTER_ELEMENT} --label_set ${LS}"),
        "SummaryCharts": Builder(action="python scripts/summary_charts.py --cluster_csvs ${SOURCES} --purity ${TARGETS[0]} --accs_out ${TARGETS[1]} --so_acc ${TARGETS[2]} --summary_out ${TARGETS[3]} --dtag_purity ${TARGETS[4]} --d_conc ${TARGETS[5]} --label_set ${LS}"),
        "Transforms": Builder(action="python scripts/transforms.py --k_csv ${SOURCES[0]} --edits_out ${TARGETS[0]} --tokens_out ${T_O}"),
        "PathDistance": Builder(action="python scripts/path_distance.py --k_csv ${SOURCES[0]} --sim_out ${TARGETS[0]}"),
        "FTPathDistance": Builder(action="python scripts/ft_path_distance.py --k_csv ${SOURCES[0]} --sim_out ${TARGETS[0]}"),
        "TransformChart": Builder(action="python scripts/transform_chart.py --edits_in ${SOURCES[0]} --n_edits ${N_TRANSFORMS} --outfile ${TARGETS[0]}"),
        "DirChart": Builder(action="python scripts/combined_reduce_diffs.py --pca_csv ${SOURCES[0]} --transforms ${SOURCES[1]} --outfile ${TARGETS[0]} --k ${K}"),
        "AnalyzeDetails": Builder(
            action="python scripts/analyze_detail.py --dummy_infile ${SOURCES[0]} --purity ${TARGETS[0]} --acc ${TARGETS[1]} --std_obs ${TARGETS[2]} --infile ${JSONL_LOC}"),
        "LDTransforms" : Builder(
            action="python scripts/ld_transforms.py --dummy_infile ${SOURCES[0]} --edits_out ${TARGETS[1]} --lds_out ${TARGETS[2]} --infile ${JSONL_LOC} --n_edits ${N_TRANSFORMS} --outfile ${TARGETS[0]}"),
        "SummaryTransforms" : Builder(action = "python scripts/summary_transforms.py --transform_jsons ${SOURCES} --tag_transforms ${TARGETS[0]}"),
        "AggCharts" : Builder(
            action="python scripts/agg_charts.py --outfile ${TARGETS[0]} --to_analyze ${SOURCES} --element ${ELEMENT} --model_names ${ANALYZE_MODELS}")
   }

)

embeds = []
embed_names = []
results = {}
#for dataset_name in env["SENT_DATASET_STEMS"]:
for metadata, corpus_dir, dataset_name in zip(env["METADATA"], env["GUT_DIR"], env["DATASET_NAMES"]):
    for fold in range(1, env["FOLDS"] + 1):
        for model_type in env["MODEL_TYPES"]:
            if model_type == "fasttext_pretrained":
                embed_names.append(model_type + "_"+ str(fold))
                embeds.append(
                    env.ApplyModel(
                        "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/embeds.gz.json",
                        [env["PRETRAINED_LOC"]],
                        SOURCE_DATASET=env["SENT_DATASET_STEMS"][0],
                        DATASET_NAME=dataset_name,
                        PRETRAINED=1,
                        MODEL_TYPE=model_type)
                    )
            elif model_type in ["google/canine-c","bert-large-uncased", "google/canine-s", "bert-base-uncased", "bert_forced"]:
                embed_names.append(model_type + "_"+ str(fold))
                embeds.append(
                    env.RetrieveBertlike(
                        "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/embeds.gz.json",
                        [],
                        SOURCE_DATASET=env["SENT_DATASET_STEMS"][0],
                        DATASET_NAME=dataset_name,
                        MODEL_TYPE=model_type)
                    )


res = defaultdict(dict)

for ename, embed in zip(embed_names, embeds):
    for ls in env["LABEL_SETS"]:
        for c_element in env["CLUSTER_ELEMENTS"]:
            cluster_csvs = []
            ld_transforms = []
            for k in range(env["K"][0], env["K"][1]+1):
                cluster_csvs.append(env.Cluster(["work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/${K}.csv"], [embed],
                    ENAME = ename,
                    CLUSTER_ELEMENT = c_element,
                    LS = ls,
                    LSN = "".join(ls),
                    K = k
                ))

                ld_transforms.append(env.Transforms(["work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/edits/${K}.jsonl"], [cluster_csvs[-1]],
                    ENAME = ename,
                    CLUSTER_ELEMENT = c_element,
                    LSN = "".join(ls),
                    K = k,
                    T_O = "work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/edits/${K}_tokens.txt"
                ))

                if k >=10:
                   #pd = env.PathDistance(["work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/path_distance/${K}.csv"], [cluster_csvs[-1]],
                      #ENAME=ename,
                      #CLUSTER_ELEMENT = c_element,
                      #LSN = "".join(ls),
                      #K=k
                  #)

                   pd_ft = env.FTPathDistance(["work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/path_distance/${K}_ft.csv"], [cluster_csvs[-1]],
                      ENAME=ename,
                      CLUSTER_ELEMENT = c_element,
                      LSN = "".join(ls),
                      K=k
                  )

                tc = env.TransformChart(["work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/edits/${K}.png"], [ld_transforms[-1]],
                    ENAME = ename,
                    CLUSTER_ELEMENT = c_element,
                    LSN = "".join(ls),
                    K = k
                )

                dc = env.DirChart(["work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/dir_chart/${K}.png"], [cluster_csvs[-1], ld_transforms[-1]],
                    ENAME = ename,
                    CLUSTER_ELEMENT = c_element,
                    LSN = "".join(ls),
                    K = k
                )

            t_dtag_chart = env.SummaryTransforms(["work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/dtag_transforms.png"], ld_transforms, ENAME=ename, CLUSTER_ELEMENT=c_element, LSN="".join(ls))
            purity, acc, std_ob_acc, summ, dtag, conc = env.SummaryCharts(["work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/purity.png", "work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/acc.png",
                "work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/std_obs.png", "work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/summary.csv", "work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/dtag_purity.png", "work/results/${ENAME}/${LSN}/${CLUSTER_ELEMENT}/dtag_conc.png"],
                [cluster_csvs],
                ENAME = ename,
                CLUSTER_ELEMENT = c_element,
                LS = ls,
                LSN = "".join(ls))
            res["".join(ls)+"_"+c_element][ename] = summ

for sum_name, summs in res.items():
    for element in ["Purity", "Avg_Acc", "Avg_SO_Acc", "Dtag_purity","D_conc"]:
        chart = env.AggCharts(["work/results/summaries/${SUM_NAME}/${ELEMENT}.png"], [summs[m] for m in env["ANALYZE_MODELS"]],
              SUM_NAME = sum_name,
              ELEMENT = element
)
        


