# Experiments Clustering Literary Variant Orthography

Code and data in support of [Examining Language Modeling Assumptions Using an Annotated Literary Dialect Corpus](https://arxiv.org/abs/2410.02674). Presented at NLP4DH 2024 at EMNLP and published in the ACL Anthology.

##Data Details

The corpus (found in data/) consists of 4032 orthovariant tokens and their context drawn from a version of the Project Gutenberg corpus restricted to U.S. Literary works published in the early 19th to early 20th centuries.

Messner provided two data annotations for each sample:
1. The modern "standard" version of each orthovariant token
2. A dialect tag (dtag) indicating the subject position, as intended by the author, of the utterer of the token. For more details on the annotation process, see the paper.


##Running the Code

1. Embed the dataset using [this code](https://github.com/comp-int-hum/llm-direct-embeddings)
2. Install dependencies using requirements.txt and pip
3. Run the experiments using scons -Q.




