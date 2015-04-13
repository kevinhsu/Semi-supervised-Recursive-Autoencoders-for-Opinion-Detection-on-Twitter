import os
import csv
import random

from data import Datapoint
from settings import DATA_PATH

def _iter_data_file(filename):
    path = os.path.join(DATA_PATH, filename)
    it = csv.reader(open(path).read().splitlines(), delimiter="\t")
    row = next(it)  # Drop column names

    while '#' in row[0]: # skip comments at the beginning of the data file
        row=next(it)

    if " ".join(row[:13]) != "tweet.id pub.date.GMT content author.name author.nickname rating.1 rating.2 rating.3 rating.4 rating.5 rating.6 rating.7 rating.8":
        raise ValueError("Input file has wrong column names: {}".format(path))

    for row in it:
        ratings=row[5:]
        data=row[:5]
        data.append(getLabel(ratings))
        yield Datapoint(*data)

# combine the ratings to generate a target label
def getLabel(ratings):
    ratings=[int(r) for r in ratings if not r=='']
    labels=list(set(ratings))
    cnt=[0]*len(labels)
    for rate in ratings:
        cnt[labels.index(rate)]+=1
        if cnt[labels.index(rate)]>len(ratings)/2.0:
            return rate
    maxL=max(cnt)
    randomset=[]
    for i in range(len(cnt)):
        c=cnt[i]
        if c==maxL:
             randomset.append(labels[i])

    return random.sample(randomset,1)[0]
    # else:
    #     return 3

def iter_corpus(__cached=[]):
    """
    Returns an iterable of `Datapoint`s with the contents of trainset
    """
    if not __cached:
        __cached.extend(_iter_data_file("trainset")) # file name
    return __cached

def iter_test_corpus(tagged=False):
    """
    Returns an iterable of `Datapoint`s with the contents of testset
    """
    return list(_iter_data_file("testset"))

def make_train_test_split(seed, proportion=0.9):
    """
    Makes a randomized train/test split of the train.tsv corpus with
    `proportion` fraction of the elements going to train and the rest to test.
    The `seed` argument controls a shuffling of the corpus prior to splitting.
    The same seed should always return the same train/test split and different
    seeds should always provide different train/test splits.

    Return value is a (train, test) tuple where train and test are lists of
    `Datapoint` instances.
    """
    data = list(iter_corpus())
    ids = list(sorted(set(x.id for x in data)))
    if len(ids) < 2:
        raise ValueError("Corpus too small to split")
    N = int(len(ids) * proportion)
    if N == 0:
        N += 1
    rng = random.Random(seed)
    rng.shuffle(ids)
    test_ids = set(ids[N:])
    train = []
    test = []
    for x in data:
        if x.id in test_ids:
            test.append(x)
        else:
            train.append(x)
    return train, test

def exportToFile(proportion):
    path = os.path.join(DATA_PATH, 'rawdata.tsv')
    it = csv.reader(open(path).read().splitlines(), delimiter="\t")
    row = next(it)  # Drop column names

    while '#' in row[0]: # skip comments at the beginning of the data file
        row=next(it)

    if " ".join(row[:13]) != "tweet.id pub.date.GMT content author.name author.nickname rating.1 rating.2 rating.3 rating.4 rating.5 rating.6 rating.7 rating.8":
        raise ValueError("Input file has wrong column names: {}".format(path))
    data=[]
    for row in it:
        data.append(row)
        #ratings=row[5:]
        #data=row[:5]
        #data.append(getLabel(ratings))
        #yield Datapoint(*data)
    #ids=range(len(data))
    N = range(len(data))
    if N == 0:
        N += 1
    rng = random.Random('fighter')
    rng.shuffle(N)
    test_ids = N[int(proportion*len(data)):]
    train = []
    test = []
    for i in range(len(data)):
        if i in test_ids:
            test.append(data[i])
        else:
            train.append(data[i])
   # (train,test)=make_train_test_split(seed='fighter',proportion=0.7)
    fieldnames=['tweet.id','pub.date.GMT','content','author.name','author.nickname','rating.1','rating.2','rating.3','rating.4','rating.5','rating.6','rating.7','rating.8']
    with open('./data/trainset','wb') as f1:
        wr=csv.writer(f1,delimiter='\t')
        wr.writerow(fieldnames)
        for datapoint in train:
            wr.writerow(datapoint)

    with open('./data/testset','wb') as f2:
        wr=csv.writer(f2,delimiter='\t')
        wr.writerow(fieldnames)
        for datapoint in test:
            wr.writerow(datapoint)

    print 'finish export!'

#exportToFile(proportion=0.7)
