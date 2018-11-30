from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import sys
import en_core_web_sm
nlp = en_core_web_sm.load()
import matplotlib.pyplot as plt
import seaborn as sns

from data import t0, t1, t2, t3, t4, t5, t6
from processing import tf_idf_scores

Docs = [nlp(x) for x in [t0, t1, t2, t3, t4, t5, t6]]

res = tf_idf_scores(Docs)

sns.set()

fig, ax =plt.subplots(figsize=(15,3))
sns.heatmap(res,ax=ax)
plt.savefig('tf_idf_scores.png')