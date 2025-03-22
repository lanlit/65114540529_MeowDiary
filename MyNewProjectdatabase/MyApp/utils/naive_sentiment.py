# MyApp/utils/naive_sentiment.py
from nltk import NaiveBayesClassifier as nbc
from pythainlp.tokenize import word_tokenize
from itertools import chain
import codecs

def train_classifier():
    with codecs.open('MyApp/data/positive.txt', 'r', encoding="utf-8") as f:
        listpos = [e.strip() for e in f.readlines()]
    with codecs.open('MyApp/data/negative.txt', 'r', encoding="utf-8") as f:
        listneg = [e.strip() for e in f.readlines()]

    pos1 = ['pos'] * len(listpos)
    neg1 = ['neg'] * len(listneg)
    training_data = list(zip(listpos, pos1)) + list(zip(listneg, neg1))

    vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))
    feature_set = [({i: (i in word_tokenize(sentence.lower())) for i in vocabulary}, tag)
                   for sentence, tag in training_data]

    classifier = nbc.train(feature_set)
    return classifier, vocabulary

# โหลด classifier ตอน import
classifier, vocabulary = train_classifier()

def predict_sentiment(text):
    features = {i: (i in word_tokenize(text.lower())) for i in vocabulary}
    return classifier.classify(features)
