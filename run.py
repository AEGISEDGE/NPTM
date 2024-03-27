# -*- coding: utf-8 -*-
import torch, argparse, sys, os, pickle, math, transformers

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from model import NPTM
from dataset import UserDocCorpus
from utils import build_wordembedding_from_dict

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoTokenizer, AutoModel

from palmettopy.palmetto import Palmetto

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

#====================================================================================================================================================
parser = argparse.ArgumentParser(description='Gaussian Softmax Model parameters description.')
parser.add_argument('--n-hidden', type=int, default=512, metavar='N', 
    help="The size of hidden units in user inference network (default 512)")
parser.add_argument('--user-embed-size', type=int, default=64, metavar='N', 
    help="The size of hidden units in doc inference network (default 64)")
parser.add_argument('--dropout', type=float, default=0.3, metavar='N', 
    help="The drop-out probability of MLP (default 0.3)")
parser.add_argument('--lr', type=float, default=1e-3, metavar='N', 
    help="The learning rate of model (default 1e-3)")
parser.add_argument('--topics', type=int, default=50, metavar='N',
    help="The amount of topics to be discover")
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
    help="Training batch size.")
parser.add_argument('--topicembedsize', type=int, default=100, metavar='N',
    help="Topic embedding size of topic modelling")
parser.add_argument("--plm-select", default='bert-base', type=str,
    help="The option for selecting pretrain language model.")
parser.add_argument('--topk', type=int, default=50, metavar='N',
    help="Top-k for topic coherence.")
parser.add_argument('--max-epoch', type=int, default=1000, metavar='N',
    help="Maximum epoch size for wake sleep algorithm")
parser.add_argument('--subepoch', type=int, default=10, metavar='N',
    help="Sub epoch for opitimizing doc net and decoder.")
parser.add_argument('--patience', type=int, default=10, metavar='N',
    help="patience for earlystop.")
parser.add_argument('--seed', type=int, default=0, metavar='N',
    help="Set seed for all.")
parser.add_argument('--measure', default='npmi',type=str, metavar='N',
    help="Measure for TC.")
parser.add_argument('--precision', default='high',type=str, metavar='N',
    help="Matmul precision setting for GPU.")
parser.add_argument('--note', default='None',type=str, metavar='N',
    help="Comment on current run.")
parser.add_argument('--ulr', type=float, default=5.0, metavar='N', 
    help="The lr ratio for user network optimizer.")
parser.add_argument('--savckpt', action="store_true", 
    help="Enable automatic checkpoint save.")
parser.add_argument('--expvec', action="store_true", 
    help="Export vectors when training is finished.")
parser.add_argument('--disablerapidload', action='store_true',
    help="Flag for disable RAPID load corpus from previous binary file.")
parser.add_argument('--disabledisplay', action='store_false',
    help="Flag for disable progress bar and model summary in pytorch lightning.")

#=======================================Corpus path setup=============================================
parser.add_argument('--data-path', default='twitter2016', metavar='N',
    help="Directory for corpus.")

# Pre-download pretrain language model options to be loaded
D_PLM={"bert-base": ("data/huggingface_model/bert-base-uncased", 768),
       "sbert-all-mpnet": ("data/huggingface_model/sentence-transformers/all-mpnet-base-v2/", 768)}

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
#==================================================================================================================================================
#=================================================Global parameter settings========================================================================
def main(args):
    # Global seed set
    pl.seed_everything(args.seed)
    # Load dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.set_float32_matmul_precision(args.precision)
    # Load vocab object and get vocabulary information
    vocab = pickle.load(open('data' + os.sep + args.data_path + os.sep + "vocab.bin", 'rb'))
    word2id = vocab.token2id
    vocabulary_size=len(word2id)
    assert vocabulary_size==len(word2id), "Unequal length of vocabulary size"
    id2word = {id:word for word,id in word2id.items()}
    print("===========================================================================================================")
    print('Encoder hidden units: %d, topic number: %d, dropout rate: %f, vocab size: %d' % (args.n_hidden, args.topics, args.dropout, vocabulary_size))
    #==========================================================================
    # Rapid reload dataset object:
    if os.path.exists("corpus_obj.bin") and not args.disablerapidload:
        corpus = pickle.load(open("corpus_obj.bin", 'rb'))
        corpus_name = corpus["name"]
        if corpus_name != args.data_path:
            # Re-build corpus
            print("Re-build corpus... ...")
            corpus = Build_corpus(args.data_path, vocabulary_size)
            pickle.dump(corpus, open("corpus_obj.bin", 'wb'))
            # Process corpus and save
        print("*** Rapid reload dataset object from previously dumped file. ***")
    else:
        print("Processing dataset file... ...")
        corpus = Build_corpus(args.data_path, vocabulary_size)
        pickle.dump(corpus, open("corpus_obj.bin", 'wb'))
    train_dataset = corpus["train"]
    val_dataset = corpus["val"]
    test_dataset  = corpus["test"]
    #==========================================================================
    # Load word embedding 
    word2embedding_dict = pickle.load(open("embedding/word2vec_glove.6B.100d.txt.bin", 'rb'))
    wordembedding_mat = build_wordembedding_from_dict(word2embedding_dict, id2word)
    print("Building dataLoader... ...")
    train_loader = DataLoaderX(train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=16, 
        pin_memory=True, 
        drop_last=False)
    val_loader  = DataLoaderX(val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False)
    test_loader  = DataLoaderX(test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True, 
        drop_last=False)
    n_user=train_dataset.Get_user_cnt()
    uid2user = train_dataset.Get_uid2user()
    #=========================================================================
    #Build model
    #=========================================================================
    # Auto-topic coherence settings
    tc = Palmetto("http://127.0.0.1:7777/service/", timeout=60)
    # Pre train language model loading 
    if args.plm_select in D_PLM:
        PLM_PATH, plm_hidden = D_PLM[args.plm_select]
    else:
        print("Invalid plm selection.")
        os._exit(0)
    plm_obj = {'tokenizer': AutoTokenizer.from_pretrained(PLM_PATH),
               'model': AutoModel.from_pretrained(PLM_PATH).to(device)}
    print("Building model... ...")
    # run identifier
    run_id = {"t":args.topics, 
              "h":args.n_hidden, 
              "bs":args.batch_size, 
              "lr":args.lr,
              "ulr":args.ulr,
              "do":args.dropout,
              "plm":args.plm_select,
              "subepoch":args.subepoch,
              "note-": "" if args.note=="None" else args.note,
              "seed":args.seed}
    run_id_str = "_".join([title + str(values) for title,values in run_id.items()])
    tmodel = NPTM(vocabulary_size=vocabulary_size, 
        n_user=n_user,
        n_hidden=args.n_hidden, 
        n_topics=args.topics, 
        plm_obj=plm_obj,
        plm_hidden=plm_hidden,
        tc=tc,
        lr=args.lr,
        id2word=id2word,
        c_device=device,
        topk=args.topk,
        measure=args.measure,
        dropout_prob=args.dropout,
        user_lr_ratio=args.ulr,
        subepoch=args.subepoch,
        user_embed_size=args.user_embed_size,
        embedding_size=args.topicembedsize, 
        wordembedding_mat=wordembedding_mat,
        disable_automatic_optimize=True,
        run_id=run_id_str)
    # Callback settings
    callback_list = [ EarlyStopping(monitor="val_loss", mode="min", patience = args.patience, check_on_train_epoch_end=False) ]
    # Checkpoint settings
    if args.savckpt:
        if not os.path.exists('sav'):
            os.makedirs('sav')
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath="sav" + os.sep ,
            filename='NPTM-tw'+ run_id_str +'{epoch:02d}-{val_loss:.3f}.bin',
            save_top_k=5,
            mode='min',
            save_last=True,
            save_weights_only=True)
        callback_list.append(checkpoint_callback)
    # Tensorboard settings
    logger = TensorBoardLogger('tb_logs', name='NPTM-tw-'+run_id_str)
    # Model trainer
    trainer = pl.Trainer(accelerator='gpu', 
                        max_epochs=args.max_epoch,
                        callbacks=callback_list,
                        log_every_n_steps=10,
                        logger=logger,
                        enable_progress_bar=args.disabledisplay,
                        enable_model_summary=args.disabledisplay,
                        devices=1)
    trainer.fit(model = tmodel, 
                train_dataloaders = train_loader,
                val_dataloaders = val_loader)
    trainer.test(model = tmodel,
                dataloaders = test_loader)
    #
    tmodel=tmodel.to(device)
    tmodel.eval()

#=====================================================================================================

def build_wordembedding_from_dict(wordembedding_dict, id2word):
    vec_list = []
    idlist = list(id2word.keys())
    idlist.sort()
    for idx in idlist:
        word = id2word[idx]
        if word not in wordembedding_dict.keys():
            embedding = wordembedding_dict['unk']
        else:
            embedding = wordembedding_dict[word]
        vec_list.append(embedding)
    return vec_list

#-----------------------------------------------------------------------------------------------------

def Build_corpus(data_path, vocabulary_size):
    return {"name":data_path,
            "train":UserDocCorpus(filename='data' + os.sep + data_path + os.sep + "corpus-by-user.bin", 
                                    vocabulary_size=vocabulary_size, 
                                    train_split=0.8, 
                                    data_flg="train"),
            "val":UserDocCorpus(filename='data' + os.sep + data_path + os.sep + "corpus-by-user.bin", 
                                    vocabulary_size=vocabulary_size, 
                                    train_split=0.8, 
                                    val_split=0.1, 
                                    data_flg="val"),
            "test":UserDocCorpus(filename='data' + os.sep + data_path + os.sep + "corpus-by-user.bin", 
                                    vocabulary_size=vocabulary_size, 
                                    train_split=0.8, 
                                    val_split=0.1, 
                                    data_flg="test")}

#-----------------------------------------------------------------------------------------------------

#=====================================================================================================

if __name__ == '__main__':
    main(parser.parse_args())
