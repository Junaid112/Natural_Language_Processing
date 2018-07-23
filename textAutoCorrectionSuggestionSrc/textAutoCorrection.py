# -*- coding: utf-8 -*-
"""
Created on Thu May 26 12:33:28 2017

@author: JUNAID AHMED GHAURI (277220)
"""

#import numpy as np
#mport math as mth
from nltk.corpus import udhr
import re
from operator import itemgetter




dictNGram={}
ngram=2


# In below function I am computing n ngrams
def findNGrams(text,n):
    ngrams=list()
    text=text.lower()   
    for i in range(len(text)-n+1):
        token=text[i:i+n]
        ngrams.append(token)    
    return ngrams

# below funciton use with in  edit distance 
def substitutionError(c1,c2):
    if c1 == c2:
        return 0
    else:
        return 1

# below function is used to compute edit distance between two words
def editDistance(v, w):
    matrix = [[0 for j in range(len(w) + 1)] for i in range(len(v) + 1)]
    for i in range(len(v)+1):
        for j in range(len(w)+1):
            if i > 0 and j > 0:
                val1 = matrix[i-1][j] + 1
                val2 = matrix[i][j-1] + 1
                val3 = matrix[i-1][j-1] + substitutionError(v[i-1],w[j-1]) 
                matrix[i][j] = min(val1, val2, val3)
            elif i > 0:
                matrix[i][j] = matrix[i-1][j] + 1
            elif j > 0:
                matrix[i][j] = matrix[i][j-1] + 1
            else:
                matrix[i][j] = 0 


    return matrix[len(v)][len(w)] # return the edit distance between two words

def main():
    english=udhr.raw("English-Latin1") # loading data for English from UDHR
    textTokens=re.split(" |\n",english) # split text based on space or next line feen \n
    for i in range(len(textTokens)):
        listNgrams=findNGrams(textTokens[i],ngram)
        for j in range(len(listNgrams)):
            if(listNgrams[j] not in dictNGram):
                dictNGram[listNgrams[j]]=list()
                # each ngram in dictionary contain a list of words that contain that ngram
            dictNGram[listNgrams[j]].append(textTokens[i]) 
            
    ### now I have all list of words that contain possible ngrams ##
    testLine=input("Enter a sentence for suggestions: ")
    nSuggestions=int(input("Number of suggestions you want: "))
    testWords=testLine.split(" ")
    for i in range(len(testWords)):
        candidateWordsDist=list() # cadidate words with edit distance
        candidateWords=list() # unique candidate word list
        testNgrams=findNGrams(testWords[i],ngram)
    
        for ng in testNgrams:
            if(ng in dictNGram):
                for wr in dictNGram[ng]:
                    if(wr not in candidateWords):
                        candidateWords.append(wr)
                        candidateWordsDist.append([wr,editDistance(testWords[i],wr)]) # compute the edit distance test word and train data
        candidateWordsDist.sort(key=itemgetter(1))
        #if(candidateWordsDist[0][1]==0):
        #   print("The word: "+testWords[i]+" is correct no need suggestions")
        #else:
                # line below display possible suggestions for all words in sentence
        print("possible correct spellings for word => "+testWords[i]+" are given below: ")
        print(candidateWordsDist[:nSuggestions]) # print possible corrections
        
    
main()