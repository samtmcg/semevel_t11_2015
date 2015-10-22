#!/bin/bash

# Only run the tokenizer ark CMU
dir=$LOWLANDS_HOME/tools/ark-tweet-nlp-0.3.2
set -eu
java -XX:ParallelGCThreads=2 -Xmx100m -jar $dir/ark-tweet-nlp-0.3.2.jar --just-tokenize "$@" | awk -F'\t' '{print $1}' 