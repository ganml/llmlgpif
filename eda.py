# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 22:32:33 2025

@author: gjgan
"""

import pandas as pd
import numpy as np
import os, pdb, sys, pickle
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import umap
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

tr = pd.read_csv('peril.training.csv')
te = pd.read_csv('peril.validation.csv')
df = tr
prefix = "tr"

# Assign Label 
cols = list(df.columns)[:9]
df['ClaimType'] = df[cols].idxmax(axis=1)
df = df.sort_values(by = 'Loss', ascending = True).reset_index(drop = True)

df.head()

  

columns, ct, avgLoss, minLoss, maxLoss = [], [], [], [], []
firstQ, median, thirdQ = [], [], []

for col in ['Vandalism', 'Fire', 'Lightning', 'Wind', 'Hail', 'Vehicle', 'WaterNW', 'WaterW', 'Misc']: 
    sub = df[df[col] == 1].reset_index(drop = True)
    columns.append(col)
    ct.append(len(sub))
    minLoss.append(np.min(sub['Loss']))
    firstQ.append(np.quantile(sub['Loss'], 0.25))
    avgLoss.append(np.mean(sub['Loss']))
    median.append(np.quantile(sub['Loss'], 0.5))
    thirdQ.append(np.quantile(sub['Loss'], 0.75))
    maxLoss.append(np.max(sub['Loss']))
columns.append('Overall')
ct.append(df.shape[0])
minLoss.append(np.min(df['Loss']))
firstQ.append(np.quantile(df['Loss'], 0.25))
avgLoss.append(np.mean(df['Loss']))
median.append(np.quantile(df['Loss'], 0.5))
thirdQ.append(np.quantile(df['Loss'], 0.75))
maxLoss.append(np.max(df['Loss']))

res = pd.DataFrame({'col': columns, 'Count': ct, 'Min': minLoss, '1st Q': firstQ, 'Mean': avgLoss, 'Median': median, '3rd Q': thirdQ, 'Max': maxLoss})
res.to_csv(prefix+"summary.csv", index=False)

loss = df["Loss"]
stat = [loss.min().item(), loss.quantile(0.25).item(), loss.mean().item(), loss.quantile(0.5).item(), loss.quantile(0.75).item(), loss.max().item()]
print(stat)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.hist(np.log1p(loss), bins=100, color='skyblue', edgecolor='black')
ax.set_xlabel("Logloss")
ax.set_ylabel("Count")
fig.savefig(prefix+"hist.pdf", bbox_inches='tight')

model = SentenceTransformer('all-mpnet-base-v2') #('all-MiniLM-L6-v2')
embeddings = model.encode(df['Description'].fillna(''), show_progress_bar=True)

reducer = umap.UMAP(n_neighbors=15, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

# Log loss for size or color
loss = df['Loss'].replace(0, np.nan).fillna(0.01)
log_loss = np.log1p(loss)

loss_scaled = df['Loss'].fillna(0)
sizes = (loss_scaled / loss_scaled.max()) * 15000
claim_types = df['ClaimType'].astype('category')
plt.figure(figsize=(6,6))
scatter = plt.scatter(
    embedding_2d[:,0],
    embedding_2d[:,1],
    c = df['ClaimType'].astype('category').cat.codes,
    s = sizes,
    cmap = 'tab10',
    alpha = 0.6
)
# Add legend
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      label=label,
                      markerfacecolor=scatter.cmap(scatter.norm(code)),
                      markersize=8)
           for label, code in zip(claim_types.cat.categories, claim_types.cat.codes.unique())]

plt.legend(handles=handles, title='Claim Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("UMAP Embedding Colored by Claim Type")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.tight_layout()
plt.savefig(prefix+'umap.pdf', format='pdf')


nltk.download('stopwords')

# Preprocess
stop_words = list(set(stopwords.words('english')))
vectorizer = CountVectorizer(stop_words=stop_words, max_features=25)
X = vectorizer.fit_transform(df['Description'].fillna(''))

# Sum word counts
word_counts = X.toarray().sum(axis=0)
words = vectorizer.get_feature_names_out()
freq_df = pd.DataFrame({'word': words, 'count': word_counts}).sort_values(by='count', ascending=False)

# Plot
plt.figure(figsize=(6, 6))
plt.barh(freq_df['word'], freq_df['count'], color='steelblue')
plt.xlabel('Frequency')
plt.title('Top 25 Keywords in Claim Descriptions')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(prefix+'top_keywords.pdf', format='pdf')


# summary of text description
from nltk.corpus import stopwords
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))


from lexicalrichness import LexicalRichness
df = te
lst = [desc.strip() if isinstance(desc, str) else '' for desc in df['Description']]
lst = [i.replace('-', ' ').replace('/', '') for i in lst]

# number of empty descriptions
lst.count("")

vocab = set()
numW = [] # number of words in each description
numSW = [] # number of stopwords

blob = ''
for i in range(len(lst)): 
    text = lst[i]
    blob += text + ' '
    lexi = LexicalRichness(text)
    nW = 0
    nSW = 0
    for t in lexi.wordlist: 
        vocab.add(t)
        if t in stopwords_set:
            nSW += 1
        else:
            nW += 1
    if nW == 0:
        print(text)
    numW.append(nW)
    numSW.append(nSW)

vL = []
for w in vocab:
    vL.append(len(w))

lex = LexicalRichness(blob)
print(lex.words) # word count
print(lex.terms) # unique word count
print(np.mean(vL)) # average word length
print((np.array(numSW)+np.array(numW)).mean()) # average words per description
print(np.nanmean(np.array(numSW)/(np.array(numSW)+np.array(numW)))) # average ratio of stopwords
print(lex.yulek) # yule's k
print(lex.Herdan)
