# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:33:28 2017

@author: JUNAID AHMED GHAURI (277220) This is for Named entity recognition
"""

import string
from nltk import pos_tag, word_tokenize
from nltk.chunk import conlltags2tree, tree2conlltags
import os
from nltk.stem.snowball import SnowballStemmer
import pickle
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
from nltk.sem import relextract


def converToProperIOB(sentence):
    """
    `annotated_sentence` are list of triplets ,Transform a pseudo-IOB notation to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    transformedIOBTokken = []
    for indexOt, token in enumerate(sentence):
        tag, word, NER = token
 
        if NER != 'O':
            if indexOt == 0:
                NER = "B-" + NER
            elif sentence[indexOt - 1][2] == NER:
                NER = "I-" + NER
            else:
                NER = "B-" + NER
        transformedIOBTokken.append((tag, word, NER))
    return transformedIOBTokken
 
def extractFeatures(tokens, index, history):
    """
    `tokens`are sentence with POS taggs, index are of those token we wan to get features, history`of previous IOB tags
    """
 
    # here I am initializing teh stememr for Englsih words
    stemmerForTheWords = SnowballStemmer('english') # because my text is in English
 
    # put plcaeholders in the sequence
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)
 
    # shift the index with 2 to adjsut the placeholders
    index=index+2
 #######---------------------usually these are most common feature suggest for NER via IOS 
    word, posTag = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
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
        'lemma': stemmerForTheWords.stem(word),
        'pos': posTag,
        'all-ascii': allascii,
 
        'next-word': nextword,
        'next-lemma': stemmerForTheWords.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'prev-word': prevword,
        'prev-lemma': stemmerForTheWords.stem(prevword),
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
def readLoadTrainData(fileDirectory):
    for mainRoot, directories, AllFiles in os.walk(fileDirectory):
        for nameOfFile in AllFiles:
            if nameOfFile.endswith(".tags"):
                with open(os.path.join(mainRoot, nameOfFile), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    sentencesFromFile = file_content.split('\n\n')
                    for singleSentence in sentencesFromFile:
                        tekensFromSentence = [seq for seq in singleSentence.split('\n') if seq]
 
                        tokenWithStandards = []
 
                        for indexT, singleToken in enumerate(tekensFromSentence):
                            annotations = singleToken.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]
 
                            if ner != 'O':
                                ner = ner.split('-')[0]
 
                            if tag in ('LQU', 'RQU'):   # Make it NLTK compatible
                                tag = "``"
 
                            tokenWithStandards.append((word, tag, ner))
 
                        finalConvertedTokens = converToProperIOB(tokenWithStandards)
 
                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield [((w, t), iob) for w, t, iob in finalConvertedTokens]



 
 
class NERClassifier(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
 
        self.feature_detector = extractFeatures
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=extractFeatures,
            **kwargs)
 
    def parseTheData(self, tagged_sent):
        resultedChunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in resultedChunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets),iob_triplets
    
    

def detectNERViaTrainedAlgo(sentence):
    dataDirectoryLocation="data_nlp_ner"
    reader = readLoadTrainData(dataDirectoryLocation)
    loadedData = list(reader)
    training_samples = loadedData[:int(len(loadedData) * 0.8)]
    test_samples = loadedData[int(len(loadedData) * 0.8):]
    print("#Numebr of training samples are = %s" % len(training_samples))    
    print("#Numebr of test samples are = %s" % len(test_samples))
    # I can incread number of traning sample this will imrove result but need more time
    classifier = NERClassifier(training_samples[:100])
    classifieddataTree,classifieddataList=classifier.parseTheData(pos_tag(word_tokenize(sentence)))
    listOfNamedEnties=[]
    properNoun="NNP"
    noun="NN"
    properNounPlural="NNPS"
    nounPlural="NNS"
    for x in classifieddataList:
        if properNoun in x:
            listOfNamedEnties.append((x,properNoun))
        if noun in x:
            listOfNamedEnties.append((x,noun))
        if properNounPlural in x:
            listOfNamedEnties.append((x,properNounPlural))
        if nounPlural in x:
            listOfNamedEnties.append((x,nounPlural))
    print("*****---------- NER detected by Learned Annotator")
    print(listOfNamedEnties)
    
    print("**** ---below is the output from code to extract realtion between entities by library--****")
    relationResult = relextract.tree2semi_rel(classifieddataTree)
    for s, tree in relationResult:
        print(s+" has something to do with:  "+tree)
        
 #---------------------below I will read text from file to detect NER from all file..
#----------- in below code I write code to detect and diplay NER from desired file or files
numOfFiles=1 # user can change from how many file, want to detect NER.
for c in range(numOfFiles):
    fileHandler=open('sport\\'+str(c).zfill(3)+'.txt',"r") # here I will read all files and tokenize the all sentences
    fileText=fileHandler.read()
    fileHandler.close()
    listOfLinesInFile=fileText.split('.')
    #uncomment the below loop if want to detect line by line
    #for line in listOfLinesInFile:
    detectNERViaTrainedAlgo(fileText) # detect nouns ad respective NEr from all sentence one by one

#***** if I want to check the accuracy I will uncomment teh cde, this is commented to save time here
#score = chunker.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]])
#print(score.accuracy())

