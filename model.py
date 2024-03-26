# -*- coding: utf-8 -*-
import torch, copy, pickle, sys, os, time, transformers
import torch.utils.data
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm, trange

from torch import nn, optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from wordcloud import WordCloud

from torch.distributions.kl import kl_divergence
#====================================================================================================================================================
# Model definition
class Doc_Infnetwork(pl.LightningModule):
    def __init__(self, 
        vocabulary_size, 
        n_hidden, 
        plm_obj,
        plm_hidden,
        n_topics,
        disable_automatic_optimize,
        c_device,
        dropout_prob=0.3):
        super(Doc_Infnetwork, self).__init__()
        #---------------------------------------------------------------------------------
        #Critical parameter record
        self.n_topics=n_topics
        self.vocabulary_size = vocabulary_size
        self.c_device = c_device
        if disable_automatic_optimize:
            self.automatic_optimization = False
        #---------------------------------------------------------------------------------
        # Load pre-trained model
        self.plm_tokenizer = plm_obj['tokenizer']
        self.plm = plm_obj['model']
        # 
        ad_linear = nn.Linear(plm_hidden, n_hidden)
        # torch.nn.init.xavier_uniform_( ad_linear.weight, gain=1.0 )
        self.adaptor = nn.Sequential(
            ad_linear,
            nn.Softplus(),
            nn.Dropout(dropout_prob)
        )
        self.loc = nn.Linear(n_hidden, n_topics)
        # torch.nn.init.xavier_uniform_(self.loc.weight, 1.0)
        self.lbn = nn.BatchNorm1d( n_topics, affine=False )
        self.logscale = nn.Linear(n_hidden, n_topics)
        self.sbn = nn.BatchNorm1d( n_topics, affine=False )
        # torch.nn.init.xavier_uniform_(self.logscale.weight, 1.0)
        self.logscale.weight.data.fill_(0.0)

    def Mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, rawtxt):
        with torch.no_grad():
            # -----------------------------------Huggingface BERT PLM interface-----------------------------------
            max_length = max([len(s) for s in rawtxt])
            encoded_input = self.plm_tokenizer(rawtxt, return_tensors='pt', max_length=min([max_length, 512]), padding=True, truncation=True, add_special_tokens=False).to(self.c_device)
            # cvec = self.plm(**encoded_input)[0].mean(dim=1) # [batch_size, plm_hidden]
            c_vec = self.Mean_pooling( self.plm(**encoded_input), encoded_input['attention_mask'] )
            #-----------------------------------------------------------------------------------------------------
            # -----------------------------------Sentence BERT PLM interface--------------------------------------
            # c_vec = torch.Tensor( self.plm.encode(rawtxt) ).to(self.c_device)
            #-----------------------------------------------------------------------------------------------------
        h_vec = self.adaptor(c_vec)
        loc = self.lbn( self.loc(h_vec) )
        # loc = self.loc(h_vec)
        logscale = self.sbn( self.logscale(h_vec) )
        return loc, logscale

class User_Infnetwork(pl.LightningModule):
    def __init__(self,
        n_topics, 
        n_user,
        disable_automatic_optimize,
        c_device,
        dropout_prob=0.3,
        embed_size=64,
        n_samples=10):
        super(User_Infnetwork, self).__init__()
        #---------------------------------------------------------------------------------
        # #Critical parameter record
        self.n_user  = n_user
        self.embed_size = embed_size
        self.c_device = c_device
        self.n_samples = n_samples
        self.softmax = nn.Softmax(-1)
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
        if disable_automatic_optimize:
            self.automatic_optimization = False
        # ==========================================
        self.embedding_layer = nn.Embedding(num_embeddings = n_user, embedding_dim=embed_size)
        # nn.init.normal_(self.embedding_layer.weight, mean=0, std=embed_size ** -0.5)
        torch.nn.init.orthogonal_(self.embedding_layer.weight.data.t())
        # ==========================================
        # uad_linear = nn.Linear(embed_size, n_topics)
        # torch.nn.init.xavier_uniform_(uad_linear.weight, 1.0)
        # self.adaptor = nn.Sequential(
        #                             nn.BatchNorm1d(embed_size, affine=False),
        #                             uad_linear,
        #                             nn.BatchNorm1d(n_topics, affine=False),
        #                             nn.ReLU())
        uad_linear = nn.Linear(embed_size, n_topics)
        # torch.nn.init.xavier_uniform_( uad_linear.weight, gain=1.0 )
        self.adaptor = nn.Sequential(
            uad_linear,
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

    def Get_user_embeddings(self, uid=None):
        if uid:
            return self.embedding_layer.weight.detach()
        else:
            return self.embedding_layer(uid).detach()

    def Get_user_interests(self, uids=None):
        with torch.no_grad():
            if uids:
                user_embeddings = self.embedding_layer(uids)
            else:
                user_embeddings = self.embedding_layer.weight
            vec = self.adaptor( user_embeddings )
            return self.softmax(vec) # [n_user, n_topics]

    def forward(self, uids):
        embeddings = self.embedding_layer(uids)
        vec = self.adaptor(embeddings)
        return self.softmax(vec) # [batch_size, n_topics]
        # return Dirichlet(vec).rsample([self.n_samples]).mean(0) # [batch_size, n_topics]

#----------------------------------------------------------------------------------------------------------------------------------------------------
class Inference_network(pl.LightningModule):
    def __init__(self,
        user_embed_size,
        n_topics,
        plm_hidden,
        plm_obj,
        n_hidden,
        n_user,
        vocabulary_size,
        disable_automatic_optimize,
        dropout_prob,
        c_device):
        #
        super(Inference_network, self).__init__()
        self.bn = nn.BatchNorm1d(n_topics, affine=False)
        if disable_automatic_optimize:
            self.automatic_optimization = False
        self.bn = nn.BatchNorm1d(n_topics, affine=False)
        # --------------User infnet Parameters:--------------
        # n_topics, 
        # n_user,
        # disable_automatic_optimize,
        # c_devices,
        # embed_size=[64, 128],
        # dropout_prob=0.2
        self.user_infnet=User_Infnetwork(n_topics=n_topics,
            # n_hidden=n_hidden,
            dropout_prob=dropout_prob,
            n_user=n_user,
            embed_size=user_embed_size,
            disable_automatic_optimize=disable_automatic_optimize,
            c_device=c_device
            )
        # ---------------Doc_infnet Parameters:--------------
        # vocabulary_size, 
        # n_hidden, 
        # dropout_prob, 
        # plm_obj,
        # plm_hidden,
        # n_topics,
        # disable_automatic_optimize,
        # c_device
        self.doc_infnet=Doc_Infnetwork(vocabulary_size=vocabulary_size, 
            n_hidden=n_hidden, 
            dropout_prob=dropout_prob,
            plm_obj=plm_obj,
            plm_hidden=plm_hidden,
            n_topics=n_topics,
            disable_automatic_optimize=disable_automatic_optimize,
            c_device=c_device)

    def KL_Gaussian(self, mu, logsigma):
        return -0.5 * torch.sum( 1 - torch.pow(mu ,2) - torch.exp(2*(logsigma)) + 2 *(logsigma) ,
                                 1)
        # return -0.5 * torch.sum(1 - torch.pow(user_interest*mu ,2) + 2 *(logsigma + torch.log(user_interest))  - torch.pow(user_interest,2)*torch.exp(2*(logsigma)), 1)

    def forward(self, uids, rawtxt):
        loc, logscale = self.doc_infnet(rawtxt)
        user_interest = self.user_infnet(uids)
        # mu = self.bn(user_interest * loc)
        return self.KL_Gaussian(loc, logscale), loc, logscale, user_interest
#---------------------------------------------------------------------------------------------------------------------------------------------------
class Generative_model(pl.LightningModule):
    def __init__(self, 
        vocabulary_size,
        dropout_prob, 
        n_topics, 
        plm_obj,
        plm_hidden,
        id2word,
        topk,
        disable_automatic_optimize,
        c_device,
        embedding_size=100,
        wordembedding_mat=None):
        super(Generative_model, self).__init__()
        # Record parameters
        self.n_topics = n_topics
        self.vocabulary_size = vocabulary_size
        self.c_device = c_device
        self.id2word=id2word
        self.topk=topk
        self.plm = plm_obj['model']
        if disable_automatic_optimize:
            self.automatic_optimization = False
        self.dropout = nn.Dropout(dropout_prob)
        self.bn = nn.BatchNorm1d(vocabulary_size, affine=False)
        # Component for document-topic-word process:
        #-----------------------------------------------------------------------------------------
        topic_embedding_mat = torch.nn.init.orthogonal_(torch.Tensor(n_topics, embedding_size)) # row-wise orthogonal init
        self.register_parameter('topic_embedding_mat', Parameter(topic_embedding_mat))
        # Word embedding def
        if wordembedding_mat==None:
            word_embedding_mat = torch.Tensor(embedding_size, vocabulary_size)
            # torch.nn.init.orthogonal_(word_embedding_mat.data, gain=1.0)
            torch.nn.init.normal_(word_embedding_mat)
            torch.nn.init.xavier_uniform_( word_embedding_mat, gain=1.0 )
            self.register_parameter('word_embedding_mat', Parameter(word_embedding_mat) )
        else:
            self.word_embedding_mat = wordembedding_mat.t()
            self.word_embedding_mat.requires_grad = False    
        #
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def Truncated_normal_init(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size+(4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def Gaussian_Reparameterization(self, mu, logsigma):
        std = torch.exp(logsigma)
        eps= torch.normal(mean = torch.zeros(mu.shape[1])).to(self.c_device)
        return torch.mul(eps, std).add_(mu)

    def Get_beta(self):
        return self.topicvec_mat

    def Get_Topic_vec(self):
        return self.topicvec_mat
    
    def Get_topic_dist(self):
        return self.softmax( torch.matmul(self.topic_embedding_mat, self.word_embedding_mat) )

    def Get_topictxtprob(self, input_mat, id2word, k):
        topic_txt_list = []
        topic_prob_list= []
        # Get top-k topic word distribution probability sparse matrix
        vec_length = len(input_mat)
        topk = input_mat.topk(dim=-1, k=k)
        topic_prob_value = topk[0] # value
        topic_indice = topk[1] # indice
        for topic in range(len(topic_indice)):
            topic_txt_list.append(" ".join([id2word[int(x)] for x in topic_indice[topic]]))
            topic_prob_list.append(topic_prob_value[topic])
        return topic_txt_list, topic_prob_list

    # Working flow of generative model
    def forward(self, mu, logsigma, doc_bow, user_interest):
        # Sampling from variational parameters
        u = user_interest
        # Normal theta
        z = self.Gaussian_Reparameterization(mu, logsigma) * user_interest
        # Document proportion generative process:
        theta = self.dropout( self.softmax(z) )# batch_size, n_topics
        logits = self.logsoftmax( self.bn(torch.matmul( theta,self.Get_topic_dist() )) + 1e-10 )
        recon_loss = torch.sum( - logits * doc_bow , -1)
        #----------------------------------------------------------------------------------------
        return recon_loss, z, theta
#====================================================================================================================================================
# Neural Generavtive User Profiling model
class NVUTM(pl.LightningModule):
    def __init__(self, 
        n_topics, 
        n_hidden,
        dropout_prob,
        vocabulary_size, 
        n_user, 
        plm_hidden,
        plm_obj,
        user_embed_size,
        embedding_size,
        disable_automatic_optimize,
        id2word,
        c_device,
        tc,
        lr,
        run_id,
        subepoch,
        user_lr_ratio=3.0,
        wordembedding_mat=None,
        topk=10,
        measure="npmi"):
        super(NVUTM, self).__init__()
        self.n_topics=n_topics
        self.n_user=n_user
        self.vocabulary_size=vocabulary_size
        self.topk=topk
        self.tc=tc
        self.c_device = c_device
        # self.plm_tokenizer = plm_obj['tokenizer']
        self.plm = plm_obj['model']
        self.learning_rate = lr
        self.measure = measure
        self.user_lr_ratio = user_lr_ratio
        self.subepoch = subepoch
        self.opt_flg4user_net = False
        self.subepoch_cnt = 0
        self.run_id = run_id
        if disable_automatic_optimize:
            self.automatic_optimization = False
        #----------------------------Inf network---------------------------------
        # user_embed_size,
        # n_topics,
        # plm_hidden,
        # plm_obj,
        # d_n_hidden,
        # n_user,
        # vocabulary_size,
        # disable_automatic_optimize,
        # c_device,
        # dropout_prob
        self.inf_net=Inference_network(user_embed_size=user_embed_size,
            n_topics=n_topics,
            plm_hidden=plm_hidden,
            plm_obj=plm_obj,
            n_user=n_user,
            n_hidden=n_hidden,
            vocabulary_size=vocabulary_size,
            dropout_prob=dropout_prob,
            disable_automatic_optimize=disable_automatic_optimize,
            c_device=c_device)
        self.encoder = self.inf_net
        self.encoder_parameters = self.encoder.parameters()
        self.doc_parameters = self.encoder.doc_infnet.parameters()
        self.user_parameters = self.encoder.user_infnet.parameters()
        #----------------------------Generative model---------------------------------
        # vocabulary_size, 
        # n_topics, 
        # plm_obj,
        # plm_hidden,
        # id2word,
        # topk,
        # disable_automatic_optimize,
        # c_device
        if wordembedding_mat:
            self.wordembedding_mat = torch.Tensor(wordembedding_mat).to(self.c_device)
        self.gen_model=Generative_model(vocabulary_size=vocabulary_size, 
            n_topics=n_topics, 
            plm_hidden=plm_hidden,
            plm_obj=plm_obj,
            dropout_prob=dropout_prob,
            id2word=id2word,
            topk=topk,
            embedding_size=embedding_size,
            wordembedding_mat=self.wordembedding_mat,
            disable_automatic_optimize=disable_automatic_optimize,
            c_device=c_device)
        self.decoder = self.gen_model
        self.decoder_parameters = self.decoder.parameters()
        self.softmax=nn.Softmax(-1)

    def forward(self, uids, rawtxt, doc_bow):
        # Invoke Inference network working flow
        KLD, mu, logsigma, user_interest = self.inf_net(uids, rawtxt)
        # Invoke Generative model working flow
        Re, z, theta = self.gen_model(mu, 
            logsigma,
            doc_bow,
            user_interest)
        loss = Re + KLD 
        return loss, Re, KLD, user_interest, z, theta 

    def Get_Beta(self):
        return self.gen_model.Get_Beta()

    def Get_user_interests(self):
        return self.inf_net.user_infnet.Get_user_interests()
    
    def Get_topicword_txt_matrices(self, topic_dist=None, topk=10):
        if topic_dist==None:
            topic_dist = self.decoder.Get_topic_dist()
        topicword_txt, topic_topk_prob = self.decoder.Get_topictxtprob(input_mat=topic_dist, 
                                                                                id2word=self.decoder.id2word,
                                                                                k=topk)
        return topicword_txt, [value.detach().cpu() for value in topic_topk_prob]
    
    def Get_topic_dist(self):
        return self.decoder.Get_topic_dist()

    #----------------------------------Pytorch-Lightning training procedure------------------------------------------------------
    def configure_optimizers(self):
        self.doc_optim=optim.Adam(self.doc_parameters, lr=self.learning_rate)
        self.user_optim=optim.Adam(self.user_parameters, lr=self.learning_rate * self.user_lr_ratio)
        self.dec_optim=optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        # return [self.enc_optim, self.dec_optim]
        return [self.doc_optim, self.user_optim, self.dec_optim]
    
    #----------------------------------------------------------------------------------------
    def compute_corpus_step(self, batch):
        # [ rawtxt, doc_bow, uid ]
        rawtxt, doc_bow, word_cnt, uid = batch # batch data input: 
        # loss, Re, KL, theta, user_interest
        loss, recon_loss, kl, user_interest, z, theta = self( uid, rawtxt, doc_bow.to( self.c_device ) )
        return {"mean_loss": loss.mean(),
                "batch_loss": loss,
                "theta": theta,
                "word_cnt": word_cnt,
                "kl": kl}
    
    def compute_corpus_epoch_end(self, output, phase_name):
        loss_sum = 0.0 
        ppx_sum = 0.0
        kld_sum = 0.0
        corpus_word_count = 0
        doc_cnt = 0
        epoch = self.current_epoch
        for batch_statics in output:
            mean_loss, batch_loss, theta, word_cnt, kl = batch_statics.values()
            loss_sum += batch_loss.sum()
            kld_sum += kl.sum()
            ppx_sum += torch.sum( batch_loss / word_cnt )
            corpus_word_count += torch.sum(word_cnt)
            doc_cnt += len(word_cnt)
        # Logging metrics
        # self.log(phase_name + "loss", loss_sum / doc_cnt, prog_bar=True, on_epoch=True)
        self.log(phase_name + "loss", loss_sum / doc_cnt, on_epoch=True, prog_bar=True, logger=True)
        self.log(phase_name + "kld", kld_sum / doc_cnt, on_epoch=True, logger=True)
        self.log(phase_name + "set ppx", torch.exp2( loss_sum / corpus_word_count ), on_epoch=True, logger=True)
        self.log(phase_name + "set per doc ppx", torch.exp2( ppx_sum / doc_cnt ))

    def training_step(self, batch, batch_idx):
        if batch_idx %2 ==0:
            phase = "Doc encoder"
            opt = self.optimizers()[0]
            parameters = self.doc_parameters
            self.subepoch_cnt+=1
        else:
            phase = "Decoder"
            opt = self.optimizers()[1]
            parameters = self.decoder_parameters
        if self.subepoch_cnt == self.subepoch:
            self.subepoch_cnt=0
            phase = "user encoder"
            opt = self.optimizers()[2]
            parameters = self.user_parameters
            self.opt_flg4user_net = False
        self.train()
        # Optimizing
        opt.zero_grad()
        rawtxt, doc_bow, word_cnt, uid = batch # batch data input
        batch_loss, recon_loss, kl, user_interest, z, theta = self( uid, rawtxt, doc_bow.to( self.c_device ) )
        loss = batch_loss.mean()
        self.manual_backward( loss )
        opt.step()
        self.log("train step loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"mean_loss": loss,
                "batch_loss": batch_loss,
                "theta": theta,
                "word_cnt_sum": word_cnt,
                "kl": kl}    
    
    def training_epoch_end(self, training_step_output):
        # Variables for training epoch
        self.compute_corpus_epoch_end(training_step_output, "training ")

    #----------------------------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        self.eval()
        rawtxt, doc_bow, word_cnt, uid = batch # batch data input
        batch_loss, recon_loss, kl, user_interest, z, theta = self( uid, rawtxt, doc_bow.to( self.c_device ) )
        loss = batch_loss.mean()
        return {"val_loss": loss,
                "batch_loss": batch_loss,
                "theta":theta,
                "word_cnt_sum":word_cnt,
                "kl":kl}

    def validation_epoch_end(self, validation_step_output):
        loss_sum = 0.0
        kld_sum = 0.0
        ppx_sum = 0.0
        corpus_word_count = 0
        doc_cnt = 0
        epoch = self.current_epoch
        for batch_statics in validation_step_output:
            mean_loss, batch_loss, theta, word_cnt, kl = batch_statics.values()
            loss_sum += batch_loss.sum()
            kld_sum += kl.sum()
            ppx_sum += torch.sum(batch_loss / word_cnt)
            corpus_word_count += torch.sum(word_cnt)
            doc_cnt += len(word_cnt)
        self.log("val_loss", loss_sum / doc_cnt, on_epoch=True, prog_bar=True, logger=True)
        self.log("kld", kld_sum / doc_cnt, on_epoch=True, logger=True)
        self.log("validation set ppx", torch.exp2(loss_sum / corpus_word_count), on_epoch=True, logger=True)
        self.log("validation set per doc ppx", torch.exp2(ppx_sum / doc_cnt))

    #----------------------------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        self.eval()
        return self.compute_corpus_step(batch)

    def test_epoch_end(self, test_step_output):
        tc_avg = []
        # Common model forward flow
        self.compute_corpus_epoch_end(test_step_output, "test ")
        # Task 1: Export global topics and compute topic coherence and topic diversity
        tu_score, topic_word_list = self.compute_topic_diversity(10)
        # 
        wetc = self.WETC(self.Get_topic_dist(), self.wordembedding_mat)
        self.log("WETC: ", wetc)
        if not os.path.exists("topic_files"):
            os.makedirs("topic_files")
        print("Calculating topic coherence:", end="")
        topic_id=0
        topic_with_tc = []
        with open("topic_files" + os.sep + self.run_id + "_topic-" + str(self.current_epoch).zfill(3) + ".txt", 'w') as f:
            for line in tqdm(topic_word_list):
                tc_score = self.tc.get_coherence(line.split(), self.measure)
                time.sleep(5)
                tc_avg.append(tc_score)
                topic_with_tc.append((tc_score, topic_id, line,))
                topic_id += 1
            topic_with_tc.sort(reverse=True)
            for (tc_score,topic_id,line) in topic_with_tc:
                f.write( "Topic NO." + str(topic_id).zfill(3) + " | "+ str(tc_score) + " | " + line + os.linesep )
            # Top 10% step avg tc file write
            tc_avg.sort(reverse=True)
            for i in range(1,11):
                f.write("Top " + str(i*10) + "% avg topic coherence:" + str(np.mean(tc_avg[:int(self.n_topics * i / 10)])) + os.linesep )
            f.write("WETC: " + str(wetc) + os.linesep)
            f.write("TU: " + str(tu_score) + os.linesep)
            f.write("Avg TC: " + str(np.mean(tc_avg)) + os.linesep)
        self.log("avg topic uniqueness", tu_score)
        self.log("avg topic coherence:", np.mean(tc_avg))
        # Task 2: Export user topics over the vocabulary 
        user_topic_word_with_prob, user_topic_word_list = self.user_topic_word_export(topk=50)
        pickle.dump(user_topic_word_with_prob, open("topic_files" + os.sep + self.run_id + "_user-profile-" + str(self.current_epoch).zfill(3) + ".bin", 'wb'))
        with open("topic_files" + os.sep + self.run_id +  "user-profile-" + str(self.current_epoch).zfill(3) + ".txt", 'w') as f:
            for line in user_topic_word_list:
                f.write( line + os.linesep )
        # Export user wordcloud
        for idx,user in enumerate(user_topic_word_with_prob):
            self.export_user_wordcloud(user, idx)


    def export_topic_word_topk_words(self, topk):
        return self.Get_topicword_txt_matrices(topk=topk)

    def compute_topic_diversity(self, topk):
        topic_list, topic_topk_prob = self.export_topic_word_topk_words(topk=topk)
        topic_set_list = [topic.split() for topic in topic_list]
        n_topic = len(topic_set_list)
        tu=0.0
        for topic_i in topic_set_list:
            tu_k=0.0
            for word in topic_i:
                cnt=0.0
                for topic_j in topic_set_list:
                    if word in topic_j:
                        cnt+=1.0
                tu_k += 1.0/cnt
            tu_k = tu_k / len(topic_i)
            tu += tu_k
        tu_out = tu/(1.0*n_topic)
        return tu_out, topic_list
    
    def user_topic_word_export(self, topk):
        word_with_prob = []
        user_interest = self.Get_user_interests() # [user, n_topics]
        user_topic_word = torch.matmul(user_interest, self.Get_topic_dist()).detach().cpu() # [user, vocabulary]
        topic_word_list, topic_topk_prob = self.Get_topicword_txt_matrices(user_topic_word, topk)
        for index in range(len(topic_word_list)):
            word_with_prob.append( zip(topic_word_list[index].split(), np.array(topic_topk_prob[index])) )
        return word_with_prob, topic_word_list

    def export_user_wordcloud(self, word_with_prob, idx):
        wc = WordCloud(background_color="white", max_words=1000)
        freq_dict = {word:prob*1e5 for word,prob in word_with_prob}
        wc.generate_from_frequencies(freq_dict)
        sav_path = 'topic_files' + os.sep + self.run_id + '-wordcloud-' + str(self.current_epoch).zfill(3)
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        wc.to_file(sav_path + os.sep + str(idx).zfill(3) +'.png')
        del wc
    
    def WETC(self, topicmat, wordembeddings, topk=10):
        wetc = 0.0
        topicmat=topicmat.detach()
        K = len(topicmat)
        V = len(topicmat[0])
        topic_wordid_list = []
        cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        for topic in topicmat:
            wordid_list = []
            tmp_list=[]
            word_cnt = 1
            # Construct <wordid, prob> pairs
            for index, value in enumerate(topic):
                tmp_list.append((index, value))
            # Descently sort the list
            sorted_list = sorted(tmp_list, key = lambda s:s[1], reverse=True)
            for pair in sorted_list:
                if word_cnt > topk:
                    break            
                tokenid = int(pair[0])
                wordid_list.append(tokenid)
                word_cnt += 1
            topic_wordid_list.append(wordid_list)
        # Compute WETC based on previously topic_word_id_list
        for topic in topic_wordid_list:
            for idi, wordi in enumerate(topic):
                for idj in range(idi+1, topk):
                    if idj+1>topk:
                        break
                    u = wordembeddings[topic[idi]].unsqueeze(0)
                    v = wordembeddings[topic[idj]].unsqueeze(0)
                    assert len(u)==len(v), "Unequal size of two vector in CosineSimilarity"
                    wetc += cosine(u, v).cpu()
        return wetc/(K*topk*(topk-1)/2.0) 
        

#====================================================================================================================================================
