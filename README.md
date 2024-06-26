# Neural Personalized Topic Model 

## Description
This rep is the pytorch implementation for our CIKM 2023 papar "Neural Personalized Topic Modeling for Mining User Preferences on Social Media".

## Requirement

Python enviroment:

```
Anaconda3-2022.10-Linux-x86_64

torch-1.13.1+cu117-cp39-cp39-linux_x86_64

pytorch-lightning==1.9.3

transformers==4.36.2

palmettopy==3.3

prefetch-generator==1.0.3

wordcloud==1.8.2.2

tqdm

argparse

matplotlib
```

OS and driver enviroment:
```
Ubuntu 20.04

NVIDIA-Linux-x86_64-515.105.01

cuda_11.7.0_515.43.04_linux
```

Hardware enviroment:
```
Chasis: Inspur NF5468M6
CPU:    Intel(R) Xeon(R) Platinum 8362 CPU x2
RAM:    DDR4 RECC 3200 512GB
GPU:    NVIDIA A100 80G
```

## Usage

### Data and model preparation:

Plz prepare following data and models in right directory:

+ "embedding/word2vec_glove.6B.100d.txt.bin" dictionary object of GloVe word embedding file. Using provided ``glove_vec_2_dict.py`` in ``embedding`` directory to covert original glove word embedding into this format.
+ "data/corpus/twitter-2016" for preprocessed twitter political archive data and vocabulary. plz unzip the twitter2016.zip file to this directory.
+ "data/corpus/authorblog" for preprocessed authorblog data and vocabulary.
+ "data/huggingface_model/bert-base-uncased/" for BERT-BASE-UNCASED model files to be load.
+ "data/huggingface_model/sentence-transformer/all-mpnet-base-v2/" for Sentence-Transformer model files to be load. 

To load different pretrain language models, we defined a directory and corresponding hidden dimension dictionary variable in run.py file(Line-79). Plz download ur selected pretrain language model files and place them at corresponding directory:

```
D_PLM={"bert-base": ("data/huggingface_model/bert-base-uncased", 768),
       "sbert-all-mpnet": ("data/huggingface_model/sentence-transformers/all-mpnet-base-v2/", 768)}
```
plz place the corresponding pretrained language model files to their directory before running.

We provide zipped twitter-political-archive file in "twitter2016.zip" in "data/corpus/twitter2016/". plz unzip this file to this directory before running. The raw data of Twitter Political Archive is available at https://github.com/bpb27/political_twitter_archivee. The first running would rebuild the corpus object and serialize into "corpus_obj.bin" to accelerate the next running on the same data. When the "disablerapidload" argument is not set, it will rebuild corpus from select dataset in "data-path" argument. The processed version of Authorblog data can be found at "https://www.dropbox.com/scl/fi/k9718nkqqi6goxoqz3r2k/authorblog.zip?rlkey=zococh5utggiaz7mk41j12t1f&st=uc7fg8oe&dl=0". u can acquire raw Authorblog data at "https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus". Set the argument ``data-path="corpus/authorblog"`` to select authorblog as training data. u can use Python pickle to load and check the format for given binary file.

### Evaluation setting

We leverage Palmetto(https://github.com/dice-group/Palmetto) into this code for automatic topic coherence calculating. Plz prepare Palmetto server endpoint and corresponding index files. Then modify the following line-149 in "run.py" to specify ur own Palmetto endpoint URL:
```
tc = Palmetto("http://127.0.0.1:7777/service/", timeout=60)
```

### Running

Key argument for running:

```
python run.py --n-hidden <the number of hidden unit in inference network's mlp> \
              --user-embed-size <the dimension of user embedding> \
              --dropout <dropout rate> \
              --lr <learning rate> \
              --topics <the number of topics> \
              --batch-size <batchsize> \
              --topicembedsize <the dimension of topic embedding> \
              --plm-select <specify which pretrain language model to use> \
              --topk <top-k word to be extracted when computing topic coherence> \
              --max-epoch <maximum epoch in training> \
              --subepoch <the number of alternative epoch> \
              --patience <patience para in earlystop> \
              --seed <random seed> \
              --measure <topic coherence metric used in palmettopy> \
              --ulr <learning rate for user network> \
              --savckpt <flag for enabling saving checkpoint> \
              --disablerapidload <flag for disable corpus processing and use previous serialized corpus object> \
              --disabledisplay <flag for disable pytorch_lightning echo>
```

You can use following command to practice a running on twitter-politics-archive:

```
python run.py --topics=50 --batchsize=128 --dropout=0.1 --lr=1e-4 --data-path=="twitter-2016" 
```
or authorblog data:
```
python run.py --topics=100 --batchsize=256 --dropout=0.1 --lr=1e-4 --data-path=="authorblog" 
```

## Output

When the training is finished, it will generate a text file containing top-k words in each topics at "topic_file/" directory.

## Citation

If u find this code useful, plz kindly cite our paper:
```
@inproceedings{DBLP:conf/cikm/LiuLTZLWZ23,
  author       = {Luyang Liu and
                  Qunyang Lin and
                  Haonan Tong and
                  Hongyin Zhu and
                  Ke Liu and
                  Min Wang and
                  Chuang Zhang},
  editor       = {Ingo Frommholz and
                  Frank Hopfgartner and
                  Mark Lee and
                  Michael Oakes and
                  Mounia Lalmas and
                  Min Zhang and
                  Rodrygo L. T. Santos},
  title        = {Neural Personalized Topic Modeling for Mining User Preferences on
                  Social Media},
  booktitle    = {Proceedings of the 32nd {ACM} International Conference on Information
                  and Knowledge Management, {CIKM} 2023, Birmingham, United Kingdom,
                  October 21-25, 2023},
  pages        = {1545--1555},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3583780.3614987},
  doi          = {10.1145/3583780.3614987},
  timestamp    = {Fri, 27 Oct 2023 20:40:46 +0200},
  biburl       = {https://dblp.org/rec/conf/cikm/LiuLTZLWZ23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Note

**Plz be advised that different enviroment may lead to undesired results or potential issues. Thus, this code comes WITHOUT SUPPORT.**
