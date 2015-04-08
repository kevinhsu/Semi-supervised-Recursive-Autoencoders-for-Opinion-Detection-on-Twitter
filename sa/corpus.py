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
        if cnt[labels.index(rate)]>len(labels)/2.0:
            return rate
    maxL=max(cnt)
    # n=0
    # for c in cnt:
    #     if c==maxL:
    #         n+=1
    # if n==1:
    return labels[cnt.index(maxL)]
    # else:
    #     return 3

def iter_corpus(__cached=[]):
    """
    Returns an iterable of `Datapoint`s with the contents of train.tsv.
    """
    if not __cached:
        __cached.extend(_iter_data_file("test.tsv")) # file name
    return __cached

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
