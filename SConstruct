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
    ("MODEL_TYPES", "", ["fasttext", "fasttext_pretrained", "google/canine-c", "bert-large-uncased"]),
    ("PRETRAINED_LOC","","cc.en.300.bin"),
    ("PARAMETER_VALUES", "", {"trained_300": [300], "trained_128": [128]}),
    ("K","",[1,20]),
    ("DATASET_NAMES", "", ["GB19"]),
    ("METADATA","", ["../gutenberg-ns-extractor/pg_19cam.csv"]),
    ("GUT_DIR", "", ["../../../export/corpora/gutenberg/"]),
    ("SENT_DATASET_STEMS", "", ["../llm-direct-embeddings/work/GB_0_2/bert-large-uncased/embeds/chunk_embed_custom_0_*.json.gz", "../llm-direct-embeddings/work/HT_Prose_0_2/bert-large-uncased/embeds/chunk_embed_custom_0_*.json.gz"]),
    ("FOLDS", "", 1),
    ("USE_GRID","", False)
)

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[steamroller.generate],
    
    # Defining a bunch of builders (none of these do anything except "touch" their targets,
    # as you can see in the dummy.py script).  Consider in particular the "TrainModel" builder,
    # which interpolates two variables beyond the standard SOURCES/TARGETS: PARAMETER_VALUE
    # and MODEL_TYPE.  When we invoke the TrainModel builder (see below), we'll need to pass
    # in values for these (note that e.g. the existence of a MODEL_TYPES variable above doesn't
    # automatically populate MODEL_TYPE, we'll do this with for-loops).
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
        #"TrainModel" : Builder(
        #    action="python scripts/train_model.py --parameter_value ${PARAMETER_VALUE} --model_type ${MODEL_TYPE} --train ${SOURCES[0]} --dev ${SOURCES[1]} --outputs ${TARGETS[0]}"            
        #),
        "ApplyModel" : Builder(
            action="python scripts/apply_model.py --model ${SOURCES[0]} --datasets ${SOURCE_DATASET} --outfile ${TARGETS[0]} --ft_pt ${PRETRAINED}"
        ),
        "RetrieveBertlike" : Builder(
            action="python scripts/retrieve_bertlike.py --model ${MODEL_TYPE} --datasets ${SOURCE_DATASET} --outfile ${TARGETS[0]}"
        ),
        "GenerateK": Builder(
            action="python scripts/generate_k.py --embeds ${SOURCES[0]} --outfile ${TARGETS[0]} --purity_out ${TARGETS[1]} --k ${K} --cluster_out ${CLUSTER_OUT}"
        ),
        "GenerateReport" : Builder(
            action="python scripts/generate_report.py --experimental_results ${SOURCES} --outputs ${TARGETS[0]}"
        )
   }

)

embeds = []
embed_names = []
results = []
#for dataset_name in env["SENT_DATASET_STEMS"]:
for metadata, corpus_dir, dataset_name in zip(env["METADATA"], env["GUT_DIR"], env["DATASET_NAMES"]):
    data = env.CreateData("work/${DATASET_NAME}/data.txt.gz", [], METADATA=metadata, DIR=corpus_dir, DATASET_NAME=dataset_name)
    for fold in range(1, env["FOLDS"] + 1):
        #train, dev, test = env.ShuffleData(
           # [
               # "work/${DATASET_NAME}/${FOLD}/train.txt",
               # "work/${DATASET_NAME}/${FOLD}/dev.txt",
               # "work/${DATASET_NAME}/${FOLD}/test.txt",
           # ],
           # data,
           # FOLD=fold,
           # DATASET_NAME=dataset_name,
       # )
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
            elif model_type in ["google/canine-c","bert-large-uncased"]:
                embed_names.append(model_type + "_"+ str(fold))
                embeds.append(
                    env.RetrieveBertlike(
                        "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/embeds.gz.json",
                        [],
                        SOURCE_DATASET=env["SENT_DATASET_STEMS"][0],
                        DATASET_NAME=dataset_name,
                        MODEL_TYPE=model_type)
                    )

            else:
                for pname, pvals in env["PARAMETER_VALUES"].items():
                    embed_names.append(model_type+pname+"_"+str(fold))
                    model = env.TrainModel(
                          "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/${PNAME}/model.model",
                           [data],
                           DATASET_NAME=dataset_name,
                           FOLD = fold,
                           PNAME=pname,
                           EMBED_WIDTH = pvals[0],
                           MODEL_TYPE=model_type)

                    embeds.append(
                        env.ApplyModel(
                            "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/${PNAME}/embeds.gz.json",
                            [model],
                            SOURCE_DATASET=env["SENT_DATASET_STEMS"][0],
                            DATASET_NAME=dataset_name,
                            PNAME=pname,
                            PRETRAINED=0,
                            MODEL_TYPE=model_type)
                        )

for ename, embed in zip(embed_names, embeds):
    results.append(
        env.GenerateK(
                ["work/results/${ENAME}/elbow.png", "work/results/${ENAME}/purity.png"],
                [embed],
                ENAME = ename,
                CLUSTER_OUT = "work/results/${ENAME}/"
        )
)
	    
"""
            for parameter_value in env["PARAMETER_VALUES"]:
                model = env.TrainModel(
                    "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/${PARAMETER_VALUE}/model.bin",
                    [train, dev],
                    FOLD=fold,
                    DATASET_NAME=dataset_name,
                    MODEL_TYPE=model_type,
                    PARAMETER_VALUE=parameter_value,
                )
		for k in env["K"]:
                    results.append(
                         env.ApplyModel(
                            "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/${PARAMETER_VALUE}/applied.txt",
                            [model, test],
                            FOLD=fold,
                            DATASET_NAME=dataset_name,
                            MODEL_TYPE=model_type,
                            PARAMETER_VALUE=parameter_value,              
                        )
                    )

# Use the list of applied model outputs to generate an evaluation report (table, plot,
# f-score, confusion matrix, whatever makes sense).
report = env.GenerateReport(
    "work/report.txt",
    results
)
"""
