from corpus import make_train_test_split,iter_corpus
import matplotlib.pyplot as plt
from transformations import ExtractText,EncodingText
import numpy as np
import matplotlib.patches as mpatches

def cross_validation(factory, seed, K=10, callback=None):
    seed = str(seed)
    scores = []
    for k in range(K):
        train, test = make_train_test_split(seed + str(k),proportion=1.0-1.0/K)
        predictor = factory()
        predictor.fit(train)
        score = predictor.score(test,k)
        if callback:
            callback(score)
        scores.append(score)
    return sum(scores) / len(scores)

def analyse(factory):
    data=iter_corpus()
    predictor=factory()
    predictor.fit(data)
    p1=ExtractText()
    X1=p1.transform(data)
    p2=EncodingText(predictor.vocabulary)
    p2.fit(X1)
    X=p2.transform(X1)
    y=[a.rating for a in data]
    (v1,v2,score,words)=predictor.classifier.analyse(X,predictor.vocabulary)

    labels=['neg','pos','mix','other']
    counter=[]

    for i in range(len(set(y))):
        counter.append([0.0]*len(predictor.vocabulary))

    for i in range(np.size(X,0)):
        x=X[i]
        label=y[i]
        for w in x:
            counter[label-1][w]+=1.0

    counter=np.array(counter)
    cl=[]
    for i in range(len(predictor.vocabulary)):
        cl_max=max(counter[:,i])
        for j in range(len(set(y))):
            if counter[j,i]==cl_max:
                cl.append(j)

    visualise(v1,predictor.vocabulary,cl)
    visualise(v2,predictor.vocabulary,cl)
    for i in range(len(score)):
        print 'sentiment - '+str(labels[i])
        for j in range(len(score[i])):
            print words[i][j]+' : '+str(score[i][j])

def visualise(v,vocabulary,lb):
    colors=['blue','yellow','red','green']
    cl=[colors[y1] for y1 in lb]
    labels=['neg','pos','mix','other']
    x=[a[0] for a in v]
    y=[a[1] for a in v]

    # draw
    plt.scatter(x,y,c=cl)
    for label, a, b in zip(vocabulary, x, y):
        plt.annotate(
            label,
            xy = (a, b), xytext = (-5, 5),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.25),size=10)
    blue_patch = mpatches.Patch(color='blue', label='neg')
    yellow_patch = mpatches.Patch(color='yellow', label='pos')
    red_patch = mpatches.Patch(color='red', label='mix')
    green_patch = mpatches.Patch(color='green', label='others')
    plt.legend([blue_patch,yellow_patch,red_patch,green_patch],labels)
    plt.show()

    # print to console
    for i in range(len(x)):
        print str(x[i])+","+str(y[i])+',"'+str(vocabulary[i])+'",'+str(lb[i])

