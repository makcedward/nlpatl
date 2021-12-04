# NLPatl (NLP Active Learning)
This python library helps you to perform Active Learning in NLP. NLPatl built on top of transformers, scikit-learn and other machine learning package. It can be applied into both cold start scenario (no any labeled data) and limited labeled data scenario.

The goal of NLPatl is to make use of the state-of-the-art (SOTA) NLP models to estimate the most valueable data and making use of subject matter experts (SMEs) by having them to label limited amount data. 

<br><p align="center"><img src="https://github.com/makcedward/nlpatl/blob/master/res/architecture.png"/></p>
At the beginning, you have unlabeled (and limited labeled data) only. NLPatl apply transfer learning to convert your texts into vectors (or embeddings). After that, vectors go through unsupervised learning or supervised learning to estimate the most uncertainty (or valuable) data. SMEs perform label on it and feedback to models until accumulated enough high quailty data.

# Installation
```
pip install nlpatl
```
or
```
pip install git+https://github.com/makcedward/nlpatl.git
```

# Examples
* [Quick tour](https://colab.research.google.com/drive/1dr1GY_vO_oOMixj4clzcMR7jLsNpbbvg#scrollTo=CRxkM-D76s19)
* [Custom Embeddings, Classification and Learning function](https://colab.research.google.com/drive/1IB2OWzgoPCIOjjqhjX9boyK17K3bpgmz?usp=sharing)

# Release
0.0.2dev
* [Completed] transformers supports Tensorflow
* [Completed] performance tuning during clustering
* [Completed] support multi-label
* [Completed] Custom Embeddings, Classification, Clustering, Scoring(Learning) function
* [Completed] Support Image embeddings and display Image

# Citation
```latex
@misc{ma2021nlpatl,
  title={Active Learning for NLP},
  author={Edward Ma},
  howpublished={https://github.com/makcedward/nlpatl},
  year={2021}
}
```