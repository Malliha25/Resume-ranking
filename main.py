#word embeddeding vectors
from gensim.models import Word2Vec
import string
import re
import streamlit as st
from nltk.corpus import stopwords
st.title("Resume Ranking")
with open('skills.txt',encoding='utf-8') as f:
    content=f.readlines()
content=[x.strip() for x in content]

import nltk 

from nltk.tokenize import word_tokenize
import gensim 
from gensim.models.phrases import Phraser,Phrases
x=[]
for i in content:
    tokens=word_tokenize(i)
    tok=[w.lower() for w in tokens]
    table=str.maketrans('','',string.punctuation)
    str1=[w.translate(table) for w in tok]
    words=[word for word in str1 if word.isalpha()]
    stop_words=set(stopwords.words("english"))
    words=[w for w in words if not w in stop_words]
    x.append(words)

texts=x
with open("common.txt",encoding='utf-8') as f:
    content2=f.read()
ntexts=[]
l=len(texts)
for j in range(l):
    s=texts[j]
    res=[k for k in s if k not in content2]
    ntexts.append(res)
texts=ntexts
content=texts

common_terms=["of",'with','without','many','using','and','or','the','a','an']
x=ntexts
phrases=Phrases(x,common_terms=common_terms)
bigram=Phraser(phrases)
all_sentences=list(bigram[x])
model=gensim.models.Word2Vec(all_sentences,size=5000,min_count=2,workers=4,window=4)
model.save('final.model')

z=model.wv.most_similar('machine_learning')

import os
from os import listdir
from os.path import isfile,join
import pandas as pd
from io import StringIO
from collections import Counter
import en_core_web_sm
nlp=en_core_web_sm.load()
from spacy.matcher import PhraseMatcher
def save_up(newf):
    with open(os.path.join("Dir",newf.name),"wb") as f:
        f.write(newf.getbuffer())
    return 
mypath='resumes'
only_files=[os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f))]
m1="Dir"
newf=st.file_uploader("Upload CV/Resume",type=['pdf'])
if newf is not None:
    save_up(newf)
only_files=[os.path.join(m1,f) for f in os.listdir(m1) if os.path.isfile(os.path.join(m1,f))]
from pdfminer.high_level import extract_text
import collections
def pdf_extract(file):
    text=extract_text(file)
    return text

def create_candidate_profile(file):
    model=Word2Vec.load('final.model')
    text=str(pdf_extract(file))
    text=text.replace('\n','')
    text=text.lower()
    #print(text)
    stats=[nlp(text[0]) for text in model.wv.most_similar("statistics")]
    NLP=[nlp(text[0]) for text in model.wv.most_similar("language")]
    ML=[nlp(text[0]) for text in model.wv.most_similar("machine_learning")]
    DL=[nlp(text[0]) for text in model.wv.most_similar("deep")]
    python=[nlp(text[0]) for text in model.wv.most_similar("python")]
    DE=[nlp(text[0]) for text in model.wv.most_similar("data")]
    matcher=PhraseMatcher(nlp.vocab)
    matcher.add('stats',None,*stats)
    matcher.add('NLP',None,*NLP)
    matcher.add('ML',None,*ML)
    matcher.add('DL',None,*DL)
    matcher.add('python',None,*python)
    matcher.add('DE',None,*DE)
    doc=nlp(text)
    d=[]
    matchers=matcher(doc)
    for match_id,start,end in matchers:
        r=nlp.vocab.strings[match_id]
        span=doc[start:end]
        
        d.append((r,span.text))
        #print(d)
    keywords="\n".join(f'{i[0]},{i[1]},({j})' for i,j in Counter(d).items())
    score=0
    k=keywords.split('\n')
    
    #print(k)
    for i in k:
        num=i.split("(")
        if len(num)==0:
            score+=0
        else:
            #print(num[-1][:-1])
            score+=int(num[-1][:-1])
    fname=file.split("\\")
    fname[-1]=fname[-1].split(".")
    print("file name:",fname[1][0])
    print("score:",score)
    #st.write("file name:",fname[1][0])
    #st.write("score:",score)
    return [fname[1][0],score]
    
i=0
d1=[]
while i<len(only_files):
    file=only_files[i]
    #print(i)
    d1.append(create_candidate_profile(file))
    #print(i)
    i+=1
d1.sort(key=lambda x:x[1])
for i in range(len(d1)-1,-1,-1):
    st.write("Candidate Name:",(d1[i][0]))
    st.write("Score:",d1[i][1])

