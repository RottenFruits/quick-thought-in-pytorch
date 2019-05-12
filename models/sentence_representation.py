import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import numpy as np
from .quick_thought import quick_thought

class sentence_representation(nn.Module):
    def __init__(self, corpus, w2v, embedding_dim, hidden_dim, window_size, batch_size, trace):
        super(sentence_representation, self).__init__()
        self.corpus = corpus
        self.sentence_num = len(self.corpus.data)
        self.w2v = w2v
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.trace = trace
        self.device = None
        
        if torch.cuda.is_available():
            self.model  = quick_thought(self.embedding_dim, self.hidden_dim, self.batch_size, self.sentence_num).cuda(device)
        else:
            self.model  = quick_thought(self.embedding_dim, self.hidden_dim, self.batch_size, self.sentence_num)

    def train(self, num_epochs = 100, learning_rate = 0.001):
        optimizer = optim.SGD(self.model.parameters(), lr = learning_rate)

        x, y = self.model.generate_batch(self.corpus, self.window_size)
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        dataset = torch.utils.data.TensorDataset(x, y)
        batches = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        
        for epo in range(num_epochs):
            loss_val = 0

            for batch in batches:
                self.model.initialize_encoder()
                self.model.f.zero_grad()
                self.model.g.zero_grad()
                
                optimizer.zero_grad()
                
                loss = self.model(batch, self.corpus, self.w2v)
                loss.backward()
                loss_val += loss.data
                optimizer.step()

                                    
            if self.trace == True:
                if epo % 10 == 0:
                    print(f'Loss at epo {epo}: {loss_val/len(batches)}')

    def get_vector(self, sentence):
        if torch.cuda.is_available():
            vector = self.model.f.get_vector_forward(sentence, self.w2v).view(-1).detach().cpu().numpy()
        else:
            vector = self.model.f.get_vector_forward(sentence, self.w2v).view(-1).detach().numpy()
                
        return vector

    def similarity_pair(self, sentence1, sentence2):
        return np.dot(self.get_vector(sentence1), self.get_vector(sentence2)) / (np.linalg.norm(self.get_vector(sentence1)) * np.linalg.norm(self.get_vector(sentence2)))
    
    def similarity(self, sentence, descending = True):
        sim = np.array(list(map(lambda x:  self.similarity_pair(sentence, x), self.corpus.tokenized_corpus)))
        
        sim_list  = []
        for i, j in zip(sim, self.corpus.tokenized_corpus):
            sim_list .append(list([j, i]))
            
        if descending:
            rnk = np.argsort(sim, )[::-1]
        else:
            rnk = np.argsort(sim, )
        
        sim_list = [sim_list[i] for i in rnk]
        return sim_list