"""
train and test set
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


if __name__ == "__main__":
    import argparse
    import json
    import csv
    import sys

    from corpus import iter_corpus, iter_test_corpus
    from predictor import PhraseSentimentPredictor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename")
    config = parser.parse_args()
    config = json.load(open(config.filename))

    start=time.time()
    predictor = PhraseSentimentPredictor(**config)
    predictor.fit(list(iter_corpus()))
    print "fitting takes "+str(time.time()-start)
    test = list(iter_test_corpus())
    #prediction = predictor.predict(test)
    score = predictor.score(test,'test')
    print("test score {}%".format(score * 100))
    print 'programme finished!'
