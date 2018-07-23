# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:33:28 2017

@author: JUNAID AHMED GHAURI (277220) this program is for pos taging
"""

#import numpy as np
import nltk
#from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import string
from nltk.stem.snowball import SnowballStemmer

def getFeatures(sentence, index):
    return {
        'word': sentence[index],# the word it self
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0], # check if capital
        'is_all_caps': sentence[index].upper() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],# prefix & suffix help to recognize word
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'is_numeric': sentence[index].isdigit()
    }
    
def removeTags(tagSentence):
    return [word for word, tag in tagSentence]

def transformTrainingSet(tagged_sentences):
    x, y = [], []
 
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            x.append(getFeatures(removeTags(tagged), index))
            y.append(tagged[index][1])
 
    return x, y
def posTagSentence(sentence,classifier): # my function to tag given input sentence
    sentenceTokens=sentence.split(" ")
    #print(sentenceTokens)
    tags = classifier.predict([getFeatures(sentenceTokens, index) for index in range(len(sentenceTokens))])
    #print(tags)
    return zip(sentenceTokens, tags)

def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders
#    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
#    history = ['[START2]', '[START1]'] + list(history)
 
    # shift the index with 2, to accommodate the padding
    index += 2
 
    data= tokens[index]

    word=data[0]
    pos=data[1]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = pos
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
 
        'prev-iob': previob,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }
################### Main execution #######################################################
taggedLineFromData = nltk.corpus.treebank.tagged_sents()# dataset for train and test
lengthDataDict=len(taggedLineFromData)# first I'll train on above data set then test on few lines
#print(getFeatures("My name is Junaid".split(" "),3))
# Split the dataset for training and testing

indexCut = int(.90 * len(taggedLineFromData)) # 80 % train , rest test data
training_lines = taggedLineFromData[:indexCut]
test_lines = taggedLineFromData[indexCut:]
 
#print(len(training_lines))# size of training
#print (len(test_lines)) # size of test data
 
xTrain, yTrain = transformTrainingSet(training_lines)
#print(len(xTrain)) # size of train features

# here I'm using Decision tree classifier, Other can be used just then I have to transorf ffeature accordingly
clasifier = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])
clasifier.fit(xTrain[:5000], yTrain[:5000])# for the time being only 5000 data

############## In below three lines I am computing accuracy with real tags ####
xTest, yTest = transformTrainingSet(test_lines)
predictedScore= clasifier.score(xTest, yTest)
print("Accuracy Score: "+str(predictedScore))

inputTextLine=input("Enter a sentence to tag POS:")
# output as tagg words
result=posTagSentence(inputTextLine,clasifier)
print(list(result))

from nltk.chunk import conlltags2tree, tree2conlltags
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
 
class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
 
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)
 
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)

chunker = NamedEntityChunker(training_lines[:1000])# for the tiem being only take 1000 traning lines
finalResult=chunker.parse(posTagSentence("I am leaving for Hanover this Tuesday.",clasifier))
print(finalResult)
