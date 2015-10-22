#MASC Sentence Corpus
#/home/dirkh/lowlands/data/corpora/masc_wordsense
cat data/Full_set/round*/*/*txt |egrep -oiw "as [[:alnum:]]+ as [[:alpha:]]+( [[:alnum:]]+)?" | grep -iw -vFf stopwords | sort |uniq -c | sort -nr > counts-asXasY-masc.txt
