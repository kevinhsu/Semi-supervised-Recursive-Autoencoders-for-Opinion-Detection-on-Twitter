"""
Run a 10-fold cross validation evaluation given by a json
configuration file.
"""
import time


def fix_json_dict(config):
    new = {}
    for key, value in config.items():
        if isinstance(value, dict):
            value = fix_json_dict(value)
        elif isinstance(value, str):
            if value == "true":
                value = True
            elif value == "false":
                value = False
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
        new[key] = value
    return new


class PrintPartialCV:
    def __init__(self):
        self.last = time.time()
        self.i = 0

    def report(self, score):
        new = time.time()
        self.i += 1
        print("individual {}-th fold score={}% took {} seconds".format(self.i, score * 100, new - self.last))
        self.last = new


if __name__ == "__main__":
    import argparse
    import json
    from evaluation import analyse
    from predictor import PhraseSentimentPredictor

    # get vocabulary
    from corpus import iter_corpus
    import csv,os
    from transformations import ExtractText

    if not os.path.exists('./data/vocabulary'):
        datapoints=list(iter_corpus())
        vocabulary=set()
        et=ExtractText()
        X=et.transform(datapoints)
        for datap in X:
            for w in datap.split():
                vocabulary.add(w)
        vocabulary=list(vocabulary)
        vocabulary.sort()
        with open('./data/vocabulary','wb') as f:
            wr=csv.writer(f)
            for voc in vocabulary:
                wr.writerow([voc])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename")
    config = parser.parse_args()
    config = json.load(open(config.filename))

    factory = lambda: PhraseSentimentPredictor(**config)
    factory()  # Run once to check config is ok

    report = PrintPartialCV()
    analyse(factory)

    print "Analysis finished!"
