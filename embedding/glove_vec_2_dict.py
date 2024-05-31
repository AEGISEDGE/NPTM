# -*- coding: utf-8 -*-
import argparse, sys, os
import numpy as np
import pickle
#==================================================================================================================

#==================================================================================================================

def main(argv):
    if len(argv)<2:
        print("Plz enter the corpus filename to be processed.")
        os._exit(0)
    else:
        filename = str(argv[1])
    word2vec_dict = {}
    vocab = []
    with open(filename, 'r', encoding='utf-8') as fin:
        wordwithvecs = fin.readlines()
        for row in wordwithvecs:
            word = row.strip().split()[0]
            try:
                vec = [float(value) for value in row.strip().split()[1:]]
            except:
                continue
            word2vec_dict[word]=vec
            vocab.append(word)
    pickle.dump(word2vec_dict, open("word2vec_" + filename + '.bin', 'wb'))
    embedding_vocab = list(word2vec_dict.keys())
    pickle.dump(vocab, open("embedding_vocab" +filename +".bin", 'wb'))

if __name__ == '__main__':
    main(sys.argv)
