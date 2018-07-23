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
resultList=list(result)
print(resultList)

##############if we use buit in pos result owuld be better ######
tokens=nltk.word_tokenize(inputTextLine)
resultList2=list(nltk.pos_tag(tokens))
print(resultList2)
#grammar = "NP: {<DT>?<JJ>*<NN>}" # for desired resutlt we can update the grammer
grammar2 = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""

cp = nltk.RegexpParser(grammar2)
finalresult = cp.parse(resultList)
print(finalresult)
finalresult.draw()

#grammar = r"""
#   Person: {<NE-Cap>(<NE.*>|<FM>)*<NE-Cap>}
#   Musiker: {<NN-mus><Person>}
#            {<Person><\$,><[^\$]*>*<NN-mus><\$,>}
#"""
#
#cp = nltk.RegexpParser(grammar)
#tree = cp.parse(result)
#for node in tree:
#    if isinstance(node, nltk.tree.Tree):               
#        if node.label() == 'NP':
#            NP =  node.leaves()
#            print(NP)