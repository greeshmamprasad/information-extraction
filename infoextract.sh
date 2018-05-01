#!/bin/bash
pip install --user virtualenv
mkdir venv
python -m virtualenv venv -p /home/rachmani/python3.5/bin/python3.5
source venv/bin/activate
pip install numpy
pip install scipy
pip install scikit-learn
pip install nltk
pip install spacy
python -m spacy download en
python download.py
python -m nltk.downloader all

export STANFORD_MODELS="/home/u1141804/StanfordParser/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz:/home/u1141804/StanfordParser/stanford-postagger-2015-12-09/models/english.conll.4class.distsim.crf.ser.gz:/home/u1141804/StanfordParser/stanford-ner-2015-12-09/english.muc.7class.distsim.crf.ser.gz"

export CLASSPATH="/home/u1141804/StanfordParser/stanford-ner-2017-06-09/stanford-ner.jar:/home/u1141804/StanfordParser/stanford-parser-full-2015-12-09/stanford-parser.jar:/home/u1141804/StanfordParser/stanford-postagger-2015-12-09/stanford-postagger.jar"


python ie.py $1


#python nltk_download.py

