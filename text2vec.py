import torch
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


def build_feature_vector(model, words):
    feature_vector = np.zeros((100,), dtype="float32")
    n_words = 0
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    if n_words > 0:
        feature_vector = np.divide(feature_vector, n_words)
    return feature_vector


def build_word2vec(sentences, labels):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

    X = []
    print('word to vec')
    for i, sentence in enumerate(sentences):
        if i%1000 == 0:
            print(i, '/', len(sentences))
        X.append(build_feature_vector(model, sentence))

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test