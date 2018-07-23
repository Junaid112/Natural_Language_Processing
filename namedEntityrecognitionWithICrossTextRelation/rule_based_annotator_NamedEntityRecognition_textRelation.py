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
from PyDictionary import PyDictionary
from nltk.sem import relextract

# below is to train own POS tagger
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

#inputTextLine=input("Enter a sentence to tag POS:")
#-------------------below 4 lines when I have to use my iwn trained pos tagger--------
# output as tagg words
#result=posTagSentence(inputTextLine,clasifier)
#resultList=list(result)
#print(resultList)

def detectDisplayNamedEntities(sentence):
            ##############if we use buit in pos result owuld be better ######
    # I have trained my own POS tagger as well butits accuracy is around 80 to 90 % I can use that as well
    tokens=nltk.word_tokenize(sentence)
    resultList2=list(nltk.pos_tag(tokens))
    print(resultList2)
    #grammar = "NP: {<DT>?<JJ>*<NN>}" # for desired resutlt we can update the grammer
    grammar2 = r"""
      NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
          {<NNP>+}                # chunk sequences of proper nouns
          {<NNS>+}                # chunk sequences of Noun plural
          {<NNPS>+}                # chunk sequences of Proper noun, plural
          {<LS>+}                # chunk sequences of List item marker
    """
    
    cp = nltk.RegexpParser(grammar2)
    nounPhraseTree = cp.parse(resultList2)
    print(nounPhraseTree)
    
    print("**** -below is the output from code to extract realtion between entities by library--****")
    relationResult = relextract.tree2semi_rel(nounPhraseTree)
    for s, tree in relationResult:
         print(str(s)+" has something to do with:  "+str(tree))
        
    # uncomment line below ehn you want to see the tree structure of tags as well
    #nounPhraseTree.draw()
     
    nounList=[]
    for node in nounPhraseTree:
        if isinstance(node, nltk.tree.Tree):               
            if node.label() == 'NP':
                NP =  node.leaves()
                print(NP)
                for x in NP:
                    if x[1]=='NN' or x[1]=='NNP' or x[1]=='NNPS' or x[1]=='NNS':
                        nounList.append(x[0])
    
    print("*****-----------------------------------------------------------*****")                    
    print("list of all nouns detected in the text is result as below:")
    print(nounList)
    dictionary={}
    dictionary['coutries']=[]
    
#    with open('countries.txt') as openfileobject:
#        for line in openfileobject:
#            dictionary['coutries'].append(line.rstrip())
#        openfileobject.closed
            
    fileHandler= open('countries.txt')
    allCountries=fileHandler.read()
    fileHandler.close()
    dictionary['coutries']=allCountries.split("\n")
    
    fileHandler= open('months.txt')
    allCountries=fileHandler.read()
    fileHandler.close()
    dictionary['months']=allCountries.split("\n")
    
    fileHandler= open('days.txt')
    allCountries=fileHandler.read()
    fileHandler.close()
    dictionary['days']=allCountries.split("\n")
   ### same way we can use different dictionalries to tag detail with our detected nouns
    #print(dictionary['coutries'][1])
    finalNamedEntityWithEntityTags=[]
    
    for n in nounList:  # here by n I mean one noun from the list of nouns
        if n in dictionary['coutries']:
            finalNamedEntityWithEntityTags.append((n,'name of Country'))
        if n in dictionary['months']:
            finalNamedEntityWithEntityTags.append((n,'name of Month'))
        if n in dictionary['days']:
            finalNamedEntityWithEntityTags.append((n,'Day of the week'))
    
    for resultLine in finalNamedEntityWithEntityTags:         
        print(resultLine)
    
    finalNERWithDetail=[]
    dictionary=PyDictionary()
    for n in nounList:
        # this will help user to understand detected NER
        try: #try block if dictionary has no synonyn then its a name
            finalNERWithDetail.append((n,dictionary.synonym(n))) 
        except :
            finalNERWithDetail.append((n,"it is a name of something or a person")) 
            
    print("=> Detected NER with synonym detail that help to understand these NER: ") 
    for resultLine in finalNERWithDetail:
        print(resultLine)   
    #print(finalNERWithDetail)
    #-----------------------------I have tunned Grammar according to data as well then receive different resutl 
    # according to data
    #grammar = r"""
    #   Person: {<NE-Cap>(<NE.*>|<FM>)*<NE-Cap>}
    #   Musiker: {<NN-mus><Person>}
    #            {<Person><\$,><[^\$]*>*<NN-mus><\$,>}
    #"""
    #
    #cp = nltk.RegexpParser(grammar)
    #tree = cp.parse(result)
    
    
 #---------------------below I will read text from file to detect NER from all file..
#----------- in below code I write code to detect and diplay NER from desired file or files
numOfFiles=1 # user can change from how many file, want to detect NER.
for c in range(numOfFiles):
    fileHandler=open('sport\\'+str(c).zfill(3)+'.txt',"r") # here I will read all files and tokenize the all sentences
    fileText=fileHandler.read()
    fileHandler.close()
    listOfLinesInFile=fileText.split('.')
    print("********------------ Result of FIle No: "+str(c).zfill(3)+" ----------**********")
    #for line in listOfLinesInFile:
    detectDisplayNamedEntities(fileText) # detect nouns ad respective NEr from all sentence one by one
        
        
        