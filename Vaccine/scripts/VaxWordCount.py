__author__ = 'rblankenhorn'

import pandas as pd

##### import data #####
data_file = "../Data/r.csv"
dataset = pd.read_csv(data_file, sep=",")

text_file = open('../Data/r.txt', 'r')
text = text_file.read()

#cleaning
words = text.split()
words = [word.strip('.,!;()[]') for word in words]
words = [word.replace("'s", '') for word in words]

#finding unique
unique = []
for word in words:
    if word not in unique:
        unique.append(word)

#sort
unique.sort()

#print
print(unique)
print(len(unique))
uniqueC=[]
for u in unique:
    if len(u)>3:
        uniqueC.append(u)
print(uniqueC)
print(len(uniqueC))
for c in uniqueC:
    dataset[c] = dataset['stripped_text'].map(lambda x: x.count(c))
print(dataset)

# Prepare tweets in terms of word count
dataset.to_csv("Tweetswords.csv")