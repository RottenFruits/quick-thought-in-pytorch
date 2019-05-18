# Quick Thought in pytorch

This is a Quick Thought, implemented by pytorch.

---
## Installation
### Dependencies

- python 3.6.6
- pytorch-cpu 1.0.0
- numpy 1.15.4

## quick example

- Read quick thought model
```python
from models.corpus import Corpus
from models.sentence_representation import sentence_representation
```

- Read your word embedding
```python
import gensim
w2v = gensim.models.KeyedVectors.load_word2vec_format("your word embedding")
```

- Create corpus
```python
text = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]

max_vocab_size = 5000
corpus = Corpus(data = "your text", mode = "a", max_vocabulary_size = max_vocab_size, 
                max_line = 0, minimum_freq = 0)
```

- Training
```python
import torch
hidden_dim = 50
embedding_dim = 100
window_size = 1
batch_size = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #gpu

sr = sentence_representation(corpus, w2v, embedding_dim, hidden_dim, window_size, batch_size, True)
sr.train(num_epochs = 11, learning_rate = 0.1)
```

- Similarity check
```python
sr.similarity(['he', 'is', 'a', 'king'])
```
```
[[array(['he', 'is', 'a', 'king'], dtype='<U4'), 1.0000001],
 [array(['he', 'is', 'a', 'man'], dtype='<U3'), 0.9040445],
 [array(['she', 'is', 'a', 'queen'], dtype='<U5'), 0.8714786],
 [array(['she', 'is', 'a', 'woman'], dtype='<U5'), 0.8556329],
 [array(['berlin', 'is', 'germany', 'capital'], dtype='<U7'), 0.8116812],
 [array(['warsaw', 'is', 'poland', 'capital'], dtype='<U7'), 0.7802326],
 [array(['paris', 'is', 'france', 'capital'], dtype='<U7'), 0.756376]]
```

## Reference
- [L. Logeswaran and H. Lee, 2018, AN EFFICIENT FRAMEWORK FOR LEARNING SENTENCE REPRESENTATIONS](https://arxiv.org/pdf/1803.02893.pdf)
- [Implementing word2vec in PyTorch (skip-gram model)](https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb)
