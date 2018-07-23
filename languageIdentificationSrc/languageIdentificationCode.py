# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:33:28 2017

@author: JUNAID AHMED GHAURI (277220)
"""

import numpy as np
import math as mth
from nltk.corpus import udhr




# below function is computing cosine similarity between two frequency vectors
def cosineSim(dt1,dt2):
    cosSim=0.0
    sumdtt1xdt2=0
    sumdtt1xdt1=0
    sumdtt2xdt2=0
    for key in dt1:
        sumdtt1xdt2=sumdtt1xdt2+dt1[key]*dt2.get(key,0)
    for key in dt1:
        sumdtt1xdt1=sumdtt1xdt1+dt1[key]*dt1[key]
    for key in dt2:
        sumdtt2xdt2=sumdtt2xdt2+dt2[key]*dt2[key]
        
    cosSim=sumdtt1xdt2/(mth.sqrt(sumdtt1xdt1)*mth.sqrt(sumdtt2xdt2))
    return cosSim
    

# In below function I am computing ngrams up to n
def nGrams(text,n):
    ngramFr={}
    # this will generate all 1, 2 & 3 ngrams an compute frequencies
    text=text.lower()
    for i in range(1,n+1):   
        for x in range(len(text)-i+1):
            token=text[x:x+i]
            if token in ngramFr:
                ngramFr[token]+=1
            else:
                ngramFr[token]=1
    return ngramFr

def main():
    ###################### in this block I load train data from text files
    #f = open("eng.txt")
    #alltext=f.read()
    #nGramsEng=nGrams(alltext,3)
    #f = open("ger.txt")
    #alltext=f.read()
    #nGramsGer=nGrams(alltext,3)
    #f = open("spn.txt")
    #alltext=f.read()
    #nGramsSpn=nGrams(alltext,3)
    #f = open("itn.txt")
    #alltext=f.read()
    #nGramsItn=nGrams(alltext,3)
    #f = open("frn.txt")
    #alltext=f.read()
    #nGramsFrn=nGrams(alltext,3)
    #f = open("danish.txt")
    #alltext=f.read()
    #nGramsDanish=nGrams(alltext,3)
    #f = open("swedish.txt")
    #alltext=f.read()
    #nGramsSwedin=nGrams(alltext,3)
    ################# in this block train data load from nltk.corpus.udhr (Universal Declaration of Human Rights)
  
    english=udhr.raw("English-Latin1")
    french=udhr.raw("French_Francais-Latin1")
    german=udhr.raw("German_Deutsch-Latin1")
    italian=udhr.raw("Italian-Latin1")
    spanish=udhr.raw("Spanish-Latin1")
    swedish=udhr.raw("Swedish_Svenska-Latin1")
    danish=udhr.raw("Danish_Dansk-Latin1")
    
    # nGrams() will generate all up to 3 (1-3) ngrams an compute frequencies of all ngrams
    # User can vary the up to ngrams value and can see difference in output
    upToNgrams=3
    nGramsEng=nGrams(english,upToNgrams)
    nGramsGer=nGrams(german,upToNgrams)
    nGramsSpn=nGrams(spanish,upToNgrams)
    nGramsItn=nGrams(italian,upToNgrams)
    nGramsFrn=nGrams(french,upToNgrams)
    nGramsDanish=nGrams(danish,upToNgrams)
    nGramsSwedin=nGrams(swedish,upToNgrams)
    # in above part we compute ngrams up to 3 and we get best matching result till 1- 3
    
    # here input from user, whatever string user want to test
    inputStr=input("Write a string to detect language (larger string gives good result): ")
    ngramsOfInput=nGrams(inputStr,3)
    
    # below part computer similarities of test string from all languages
    result={}
    result["English"]=cosineSim(nGramsEng,ngramsOfInput)
    result["German"]=cosineSim(nGramsGer,ngramsOfInput)
    result["Spanish"]=cosineSim(nGramsSpn,ngramsOfInput)
    result["Italian"]=cosineSim(nGramsItn,ngramsOfInput)
    result["french"]=cosineSim(nGramsFrn,ngramsOfInput)
    result["Danish"]=cosineSim(nGramsDanish,ngramsOfInput)
    result["Swedish"]=cosineSim(nGramsSwedin,ngramsOfInput)
    
    return result

# in below function on the basis of max similarity I declare that is the resultant language
def detectlanguage(results):
    maximum=0
    tag=""
    for key in results:
        if(results[key]>maximum):
            maximum=results[key]
            tag=key
    return {tag:maximum}


results=main()
# here just printing the putput tesults
print("Similarities of all Languages: "+str(results))
print("Final Language with max similarity is : "+str(detectlanguage(results)))
