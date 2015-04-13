"""
This module implements several scikit-learn compatible transformers, see
scikit-learn documentation for the convension fit/transform convensions.
"""

from twokenize import simpleTokenize


class StatelessTransform:
    """
    Base class for all transformations that do not depend on training (ie, are
    stateless).
    """
    def fit(self, X, y=None):
        return self

class ExtractAuthor(StatelessTransform):
    def transform(self,X):
        return [datapoint.name for datapoint in X]

class ExtractDate(StatelessTransform):
    def transform(self,X):
        return [self.getMin(datapoint.date.split()[1]) for datapoint in X]

    def getMin(self,x):
        t=x.split(':')
        h=t[0]-1
        m=t[1]
        return h*60+m

class ExtractText(StatelessTransform):
    """
    This should be the first transformation on a pipeline, it extracts
    the phrase text from the richer `Datapoint` class.
    """
    def __init__(self, lowercase=False):
        self.lowercase = lowercase

    def transform(self, X):
        """
        `X` is expected to be a list of `Datapoint` instances.
        Return value is a list of `str` instances in which words were tokenized
        and are separated by a single space " ". Optionally words are also
        lowercased depending on the argument given at __init__.
        """
        #nltk.WordPunctTokenizer().tokenize
        #for datapoint in X:
        #    a=(" ".join(simpleTokenize(datapoint.content))+' @'+datapoint.name+' @'+datapoint.nickname+' #'+self.getMin(datapoint.date.split()[1]))
        it = (" ".join(simpleTokenize(datapoint.content))+' @'+datapoint.name+' #'+self.getMin(datapoint.date.split()[1]) for datapoint in X)
        if self.lowercase:
            return [x.lower() for x in it]
        return list(it)

    def getMin(self,x):
        t=x.split(':')
        h=int(t[0])-1
        m=int(t[1])
        return str(h*60+m)
    # def fit(self, X, y=None):
    #     return X

class EncodingText():
    def __init__(self,vocabulary):
        from sklearn.preprocessing import LabelEncoder
        self.le=LabelEncoder()
        self.vocabulary=vocabulary

    def fit(self,X,y=None):
        self.le.fit(self.vocabulary)
        return self

    def transform(self,X):
        for x in X:
            y=self.le.transform(x.split())
        return [self.le.transform(x.split()) for x in X]
        #return [self.getSparseM(x) for x in X]

    def getSparseM(self,x):
        from scipy.sparse import coo_matrix
        import numpy as np
        sent=x.split()
        ind=self.le.transform(sent)
        a=coo_matrix((np.ones([len(sent)]),(ind,range(len(sent)))),shape=(len(self.vocabulary),len(sent)))
        return a

class Densifier(StatelessTransform):
    """
    A transformation that densifies an scipy sparse matrix into a numpy ndarray
    """
    def transform(self, X, y=None):
        """
        `X` is expected to be a scipy sparse matrix.
        It returns `X` in a (dense) numpy ndarray.
        """
        return X.todense()
