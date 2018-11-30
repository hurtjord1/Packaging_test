

from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import sys
import en_core_web_sm
nlp = en_core_web_sm.load()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def lemmatize(Doc):
    return [
        token.lemma_ for token in Doc
        if not token.is_punct and not token.is_space
        and not token.lower_ in STOP_WORDS
        and (not token.tag_ == "POS" or token.text == "'s") and not token.text == "US"
    ]

def tf(x, Doc):
    l = lemmatize(Doc)
    return l.count(' '.join((lemmatize(nlp(x)))))


# In[5]:


def idf(x, Docs):
    doc_number = 0
    for elem in Docs:
        l = lemmatize(elem)
        if l.count(' '.join(lemmatize(nlp(x)))) > 0:
            doc_number += 1
    if doc_number>0:
        return 1/doc_number
    else:
        return 0
    #return 1 / doc_number if doc_number else 0


def tf_idf(x, Doc, Docs):
    t_f = tf(x, Doc)
    i_d_f = idf(x, Docs)
    return t_f * i_d_f


# In[9]:





# In[10]:


def all_lemmas(Docs):
    all = set()
    for Doc in Docs:
        all |= set(lemmatize(Doc))
        #all.update(set(lemmatize(docs)))
        #all = lemmas.union(set(lemmatize(doc)))
    return all


def tf_idf_doc(Doc, Docs):
    lemmas = all_lemmas(Docs)
    lemma_freq = {x : tf_idf(x, Doc, Docs) for x in lemmas}
    return lemma_freq


# In[13]:




def tf_idf_scores(Docs):
    data = pd.DataFrame(tf_idf_doc(Doc, Docs) for Doc in Docs)
    return data


# In[15]: