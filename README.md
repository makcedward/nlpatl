# NLPatl (NLP Active Learning)

This python library helps you to perform Active Learning in NLP. NLPatl built on top of transformers, scikit-learn and other machine learning package. It can be applied into both cold start scenario (no any labeled data) and limited labeled data scenario.

The goal of NLPatl is to make use of the state-of-the-art (SOTA) NLP models to estimate the most valueable data and making use of subject matter experts (SMEs) by having them to label limited amount data. 

<br><p align="center"><img src="https://github.com/makcedward/nlpatl/blob/master/res/architecture.png"/></p>
At the beginning, you have unlabeled (and limited labeled data) only. NLPatl apply transfer learning to convert your texts into vectors (or embeddings). After that, vectors go through unsupervised learning or supervised learning to estimate the most uncertainty (or valuable) data. SMEs perform label on it and feedback to models until accumulated enough high quailty data.

# Quick tour

For no any labeled data, you can try the following sample code:
```
from datasets import load_dataset
from nlpatl.models import ClusteringSamlping

# Get raw data
texts = load_dataset('ag_news')['train']['text']
train_texts = texts[0:5] + texts[200:205] + texts[1000:1005]

# Initialize clustering sampling apporach to estimate the most valuable data for labeling
learning = ClusteringSamlping()
learning.init_embeddings_model(
    'bert-base-uncased', return_tensors='pt', padding=True, 
    batch_size=8)
learning.init_clustering_model(
    'kmeans', model_config={'n_clusters': 3})

# Label data in notebook interactively
learning.explore_educate_in_notebook(train_texts, num_sample=2)

learn_x, learn_y = learning.get_learnt_data()
print('Features:{}'.format(learn_x))
print('Label:{}'.format(learn_y))
```

For having limited data, you can try the following sample code:
```
from datasets import load_dataset
from nlpatl.models import EntropySampling

# Get raw data
texts = load_dataset('ag_news')['train']['text']
labels = load_dataset('ag_news')['train']['label']
train_texts = texts[0:5] + texts[200:205] + texts[1000:1005]
train_labels = labels[0:5] + labels[200:205] + labels[1000:1005]
test_texts = texts[0:10] + texts[200:210]

# Initialize entropy sampling apporach to estimate the most valuable data for labeling
learning = EntropySampling()
learning.init_embeddings_model(
    'bert-base-uncased', return_tensors='pt', padding=True,
    batch_size=8)
learning.init_classification_model(
    'logistic_regression',
    model_config={'max_iter': 500})

# Train sample classification model first
learning.learn(train_texts, train_labels)

# Label data in notebook interactively
learning.explore_educate_in_notebook(train_texts, num_sample=2)

learn_x, learn_y = learning.get_learnt_data()
print('Features:{}'.format(learn_x))
print('Label:{}'.format(learn_y))
```

# Citation

```latex
@misc{ma2021nlpaug,
  title={NLP Active Learning},
  author={Edward Ma},
  howpublished={https://github.com/makcedward/nlpatl},
  year={2021}
}
```