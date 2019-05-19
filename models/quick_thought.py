import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import collections

class encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size):
        super(encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
        self.hidden = self.init_hidden() 
        self.cell = self.init_cell()
    
    def initialize_layer(self):        
        self.hidden = self.init_hidden()
        self.cell = self.init_cell()
        
    def init_hidden(self):
        ret = torch.zeros(1, self.batch_size, self.hidden_dim)
        if torch.cuda.is_available():
            ret = ret.cuda(device)
        return ret
    
    def init_cell(self):
        ret = torch.zeros(1, self.batch_size, self.hidden_dim)
        if torch.cuda.is_available():
            ret = ret.cuda(device)
        return ret
 
    def forward(self, sentences, seq_lengths, w2v):
        s = []
        for sentence in sentences:
            ws = []
            for w in sentence:
                try:
                    ws.append(w2v[w])
                except:
                    ws.append(np.zeros(self.embedding_dim))
            s.append(ws)
        embeds = torch.FloatTensor(s)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first = True)
        
        if torch.cuda.is_available():
            embeds = embeds.cuda(device)
                
        lstm_out, (self.hidden, self.cell) = self.lstm(
            embeds,
            (self.hidden, self.cell)
        )        
        return self.hidden

    def get_vector_forward(self, sentence, w2v):
        self.hidden = torch.zeros(1, 1, self.hidden_dim)
        self.cell = torch.zeros(1, 1, self.hidden_dim)
        seq_lengths = len(sentence)
        s = []
        for w in sentence:
            try:
                s.append(w2v[w])
            except:
                s.append(np.zeros(self.embedding_dim))
        embeds = torch.FloatTensor(s)
        #embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first = True)
        
        if torch.cuda.is_available():
            embeds = embeds.cuda(device)
                
        lstm_out, (self.hidden, self.cell) = self.lstm(
            embeds.view(seq_lengths, 1, -1),
            (self.hidden, self.cell)
        )        
        return self.hidden

class quick_thought(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size, sentence_num):
        super(quick_thought, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sentence_num = sentence_num
        self.idx = 0
        
        self.f = encoder(self.embedding_dim, self.hidden_dim, self.batch_size)
        self.g = encoder(self.embedding_dim, self.hidden_dim, self.sentence_num)
        
    def initialize_encoder(self):
        self.f.initialize_layer()
        self.g.initialize_layer()
    
    def generate_batch(self, corpus, window_size):
        idx = 0 
        data = list(range(len(corpus.data)))
        sentence_num = len(data)

        context = collections.deque()
        target  = collections.deque()

        while idx < len(data):
            target_ = data[idx]

            start_idx = idx - window_size
            start_idx = 0 if  start_idx < 0 else start_idx
            end_idx = idx + 1 + window_size
            end_idx = end_idx if  end_idx < (sentence_num)  else sentence_num

            for t in range(start_idx, end_idx):
                if t > sentence_num - 1:break
                if t == idx:continue
                context.append(data[t])
                target.append(target_)
                
            idx = (idx + 1)
        x = np.array(target)
        y = np.array(context) 
        return x, y
        

    def forward(self, batch, corpus, w2v):
        y_true = batch[1]

        #select target sentence
        target_sentences = [corpus.tokenized_corpus[i] for i in batch[0]]
        target_sentences, target_seq_lengths = self.sentence_padding(target_sentences)
        u = self.f(target_sentences, target_seq_lengths, w2v)

        context_sentences, context_seq_lengths = self.sentence_padding(corpus.tokenized_corpus)
        v = self.g(context_sentences, context_seq_lengths, w2v)

        if torch.cuda.is_available():
            u = u.cuda(device)
            v = v.cuda(device)
            y_true = y_true.cuda(device)
        
        z = torch.matmul(u.view(-1, self.hidden_dim), torch.t(v.view(-1, self.hidden_dim)))
        log_softmax = F.log_softmax(z, dim = 1)
        loss = F.nll_loss(log_softmax, y_true)
        
        return loss
    
    def sentence_padding(self, sentences):
        #sort sentence
        seq_lengths = []
        for s in sentences:
            seq_lengths.append(len(s))
        seq_lengths_rnk = np.argsort(np.array(seq_lengths))[::-1]
        sentences = [sentences[i] for i in seq_lengths_rnk]

        #recalculate length
        seq_lengths = []
        for s in sentences:
            seq_lengths.append(len(s))
        seq_lengths_rnk = np.argsort(np.array(seq_lengths))[::-1]
        seq_lengths_max = max(seq_lengths)

        #padding
        sentences_pad = []
        for s in sentences:
            while len(s) <= seq_lengths_max:
                s = np.append(s,"99999999999") #"99999999999" is padding word
            sentences_pad.append(s)

        return sentences_pad, seq_lengths
