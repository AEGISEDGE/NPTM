# -*- coding: utf-8 -*-
import spacy
import math, sys, pickle
#import symbolic constant
from torch.utils.data import Dataset
import math, sys, pickle, torch

class UserDocCorpus(Dataset):
    def __init__(self, filename,
                vocabulary_size,
                train_split=0.8,
                val_split=0.1,
                data_flg="train"):
        super().__init__()
        self.vocabulary_size=vocabulary_size
        doc_list=[]
        # split data
        # Data format:
        # doc = {"name": user_name,
        #         "text": rawtxt,
        #         "clean_bow": bow,
        #         "bow_word_cnt": word_cnt,
        #         "timestamp": creat_time}
        # read corpus file
        corpus_by_user = pickle.load(open(filename, 'rb'))
        guid=0
        user2uid = {}
        # split by user
        # If we need to split train, validate and test data from one "corpus-by-user.bin":
        for user in corpus_by_user:
            if user in user2uid:
                uid = user2uid[user]
            else:
                user2uid[user] = guid
                uid = guid
                guid += 1
            doc_cnt = len(corpus_by_user[user])
            if data_flg=="train":
                for doc in corpus_by_user[user][:int(doc_cnt * train_split)]:
                    doc["uid"] = uid
                    doc_list.append(doc)
            elif data_flg=="val":
                for doc in corpus_by_user[user][int(doc_cnt * train_split):int(doc_cnt * (train_split + val_split))]:
                    doc["uid"] = uid
                    doc_list.append(doc)
            else:
                for doc in corpus_by_user[user][int(doc_cnt * (train_split + val_split)):]:
                    doc["uid"] = uid
                    doc_list.append(doc)
        self.doc_list = doc_list
        self.user2uid = user2uid
        self.uid2user = {uid:user for user,uid in user2uid.items()}
        del corpus_by_user
    
    def Get_user2uid(self):
        return self.user2uid

    def Get_uid2user(self):
        return self.uid2user
    
    def Get_user_cnt(self):
        return len(self.user2uid)

    def __len__(self):
        return len(self.doc_list)

    def count_word(self, clean_bow):
        return sum([tf for _, tf in clean_bow])
    
    def bow2tensor(self, bow):
        vec = torch.zeros(self.vocabulary_size)
        for (token,tf) in bow:
            vec[token]=tf
        return vec
    
    def __getitem__(self, index):
        doc = self.doc_list[index]
        # Return: text, bow, bow_word_cnt, label, other metadata can be retrieved either
        return doc['text'], self.bow2tensor(doc['clean_bow']), doc["bow_word_cnt"], doc['uid'] 
