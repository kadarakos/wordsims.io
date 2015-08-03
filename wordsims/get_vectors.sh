#!/bin/bash
wget http://nlp.stanford.edu/~socherr/ACL2012_wordVectorsTextFile.zip > logfile
unzip ACL2012_wordVectorsTextFile.zip
paste -d " " vocab.txt wordVectors.txt > socher_vectors.txt
sed -i 's/ *$//' socher_vectors.txt
sed -i '8d' socher_vectors.txt