__author__ = 'Yazhe'

import numpy as np
import rnntree
from util import sigmoid_prime,sigmoid,tanh,norm1tanh_prime,norm,softmax,softmax_prime
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

class RNN:
    '''
    Initialise all the parameters of the recursive neural network
    inputs: hyperparameters are given by the user
    d: word vector length, usually 25-1000
    cat: number of class in the supervised situation
    vocab: number of vocabulary
    alpha: parameter that balancing the reconstruction cost and labelling cost
    lambdas: degree of regularization
    '''
    def __init__(self, d,cat,vocab,alpha,words_vectors,lambdaW,lambdaCat,lambdaL):
        # initialse parameters to be uniform distribution [-r,r]
        # where r is a small number 0.01
        r=np.sqrt(6)/np.sqrt(2*d+1)
        # W1 and W2 are dx2d matrices
        # W1 for learning W2 for reconstruction
        # 2d+1 for the bias
        self.W1=np.random.rand(d,d)*2*r-r
        self.W2=np.random.rand(d,d)*2*r-r
        #self.b1=np.random.rand(d,1)*2*r-r
        self.b1=np.zeros([d,1])
        self.W3=np.random.rand(d,d)*2*r-r
        self.W4=np.random.rand(d,d)*2*r-r
        self.b2=np.zeros([d,1])
        self.b3=np.zeros([d,1])
        #self.b2=np.random.rand(d,1)*2*r-r
        #self.b3=np.random.rand(d,1)*2*r-r
        # Wlab for learning sentiment labels
        self.Wlab=np.random.rand(cat,d)*2*r-r
        self.blab=np.zeros([cat,1])
        #self.blab=np.random.rand(cat,1)*2*r-r
        # Wrep for learning the word vector representation
        self.WL=(np.random.rand(d,vocab)*2*r-r)*10**(-3)

        self.alpha=alpha
        self.cat=cat
        self.d=d
        self.vocab=vocab
        self.words_vectors=words_vectors
        self.lambdaW=lambdaW
        self.lambdaCat=lambdaCat
        self.lambdaL=lambdaL
        self.postClassifier=LogisticRegression(penalty='l2',multi_class='multinomial',C=10**6,solver='lbfgs')

    def combineParams(self):
        d=self.d
        cat=self.cat
        vocab=self.vocab
        return np.hstack([np.reshape(self.W1,d*d),np.reshape(self.W2,d*d),np.reshape(self.W3,d*d),np.reshape(self.W4,d*d),
                          np.reshape(self.Wlab,cat*d),np.reshape(self.b1,d),np.reshape(self.b2,d),np.reshape(self.b3,d),
                          np.reshape(self.blab,cat),np.reshape(self.WL,d*vocab)])

    def forwardProp(self,allKids,words_embedded,updateWlab,label,theta,freq):
        (W1,W2,W3,W4,Wlab,b1,b2,b3,blab,WL)=self.getParams(theta)
        sl=np.size(words_embedded,1)
        sentree=rnntree.rnntree(self.d,sl,words_embedded)
        collapsed_sentence = range(sl)
        if updateWlab:
            temp_label=np.zeros(self.cat)
            temp_label[label-1]=1.0
            nodeUnder = np.ones([2*sl-1,1])

            for i in range(sl,2*sl-1): # calculate n1, n2 and n1+n2 for each node in the sensentree and store in nodeUnder
                kids = allKids[i]
                n1 = nodeUnder[kids[0]]
                n2 = nodeUnder[kids[1]]
                nodeUnder[i] = n1+n2

            cat_size=self.cat
            sentree.catDelta = np.zeros([cat_size, 2*sl-1])
            sentree.catDelta_out = np.zeros([self.d,2*sl-1])

            # classifier on single words
            for i in range(sl):
                sm = softmax(np.dot(Wlab,words_embedded[:,i]) + blab)
                lbl_sm = (1-self.alpha)*(temp_label - sm)
                sentree.nodeScores[i] = 1.0/2.0*(np.dot(lbl_sm,(temp_label- sm)))
                sentree.catDelta[:, i] = -np.dot(lbl_sm,softmax_prime(sm))

            # sm = sigmoid(self.Wlab*words_embedded + self.blab)

            #lbl_sm = (1-self.alpha)*(label[:,np.ones(sl,1)] - sm)
            #sentree.nodeScores[:sl] = 1/2*(lbl_sm.*(label(:,ones(sl,1)) - sm))
            #sentree.catDelta[:, :sl] = -(lbl_sm).*sigmoid_prime(sm)

            for i in range(sl,2*sl-1):
                kids = allKids[i]

                c1 = sentree.nodeFeatures[:,kids[0]]
                c2 = sentree.nodeFeatures[:,kids[1]]

                # Eq. [2] in the paper: p = f(W[1][c1 c2] + b[1])
                p = tanh(np.dot(W1,c1) + np.dot(W2,c2) + b1)

                # See last paragraph in Section 2.3
                p_norm1 = p/norm(p)

                # Eq. (7) in the paper (for special case of 1d label)
                #sm = sigmoid(np.dot(Wlab,p_norm1) + blab)
                sm=softmax(np.dot(Wlab,p_norm1) + blab)
                beta=0.5
                #lbl_sm = beta * (1.0-self.alpha)*(label - sm)
                lbl_sm = beta * (1.0-self.alpha)*(temp_label - sm)
                #lbl_sm = beta * (1.0-self.alpha) * (temp_label-sm)
                #sentree.catDelta[:, i] = -softmax_prime(sm)[:,label-1]
                #J=-(1.0-self.alpha)*np.log(sm[label-1])
                #sentree.catDelta[:, i] = -np.dot(lbl_sm,sigmoid_prime(sm))
                sentree.catDelta[:, i] = -np.dot(lbl_sm,softmax_prime(sm))
                #J = 1.0/2.0*(np.dot(lbl_sm,(label - sm)))
                J = 1.0/2.0*(np.dot(lbl_sm,(temp_label - sm)))

                sentree.nodeFeatures[:,i] = p_norm1
                sentree.nodeFeatures_unnormalized[:,i] = p
                sentree.nodeScores[i] = J
                sentree.numkids = nodeUnder

            sentree.kids = allKids
        else:
            # Reconstruction Error
            for j in range(sl-1):
                size2=np.size(words_embedded,1)
                c1 = words_embedded[:,0:-1]
                c2 = words_embedded[:,1:]

                freq1 = freq[0:-1]
                freq2 = freq[1:]

                p = tanh(np.dot(W1,c1) + np.dot(W2,c2) + np.reshape(b1,[self.d,1])*([1]*(size2-1)))
                p_norm1 =p/np.sqrt(sum(p**2))

                y1_unnormalized = tanh(np.dot(W3,p_norm1) + np.reshape(b2,[self.d,1])*([1]*(size2-1)))
                y2_unnormalized = tanh(np.dot(W4,p_norm1) + np.reshape(b3,[self.d,1])*([1]*(size2-1)))

                y1 = y1_unnormalized/np.sqrt(sum(y1_unnormalized**2))
                y2 = y2_unnormalized/np.sqrt(sum(y2_unnormalized**2))

                y1c1 = self.alpha*(y1-c1)
                y2c2 = self.alpha*(y2-c2)

                # Eq. (4) in the paper: reconstruction error
                J = 1.0/2.0*sum((y1c1)*(y1-c1) + (y2c2)*(y2-c2))

                # finding the pair with smallest reconstruction error for constructing sentree
                J_min= min(J)
                J_minpos=np.argmin(J)

                sentree.node_y1c1[:,sl+j] = y1c1[:,J_minpos]
                sentree.node_y2c2[:,sl+j] = y2c2[:,J_minpos]
                sentree.nodeDelta_out1[:,sl+j] = np.dot(norm1tanh_prime(y1_unnormalized[:,J_minpos]) , y1c1[:,J_minpos])
                sentree.nodeDelta_out2[:,sl+j] = np.dot(norm1tanh_prime(y2_unnormalized[:,J_minpos]) , y2c2[:,J_minpos])

                words_embedded=np.delete(words_embedded,J_minpos+1,1)
                words_embedded[:,J_minpos]=p_norm1[:,J_minpos]
                sentree.nodeFeatures[:, sl+j] = p_norm1[:,J_minpos]
                sentree.nodeFeatures_unnormalized[:, sl+j]= p[:,J_minpos]
                sentree.nodeScores[sl+j] = J_min
                sentree.pp[collapsed_sentence[J_minpos]] = sl+j
                sentree.pp[collapsed_sentence[J_minpos+1]] = sl+j
                sentree.kids[sl+j,:] = [collapsed_sentence[J_minpos], collapsed_sentence[J_minpos+1]]
                sentree.numkids[sl+j] = sentree.numkids[sentree.kids[sl+j,0]] + sentree.numkids[sentree.kids[sl+j,1]]


                freq=np.delete(freq,J_minpos+1)
                freq[J_minpos] = (sentree.numkids[sentree.kids[sl+j,0]]*freq1[J_minpos] + sentree.numkids[sentree.kids[sl+j,1]]*freq2[J_minpos])/(sentree.numkids[sentree.kids[sl+j,0]]+sentree.numkids[sentree.kids[sl+j,1]])

                collapsed_sentence=np.delete(collapsed_sentence,J_minpos+1)
                collapsed_sentence[J_minpos]=sl+j
        return sentree

    def backProp(self,sentree,updateWcat,words_embedded,gradW1,gradW2,gradW3,gradW4,gradWlab,gradb1,gradb2,gradb3,gradblab,gradL,theta):
        (W1,W2,W3,W4,Wlab,b1,b2,b3,blab,WL)=self.getParams(theta)
        sl=np.size(words_embedded,1)
        toPopulate = np.array([[2*sl-2],[0],[0]])
        nodeFeatures = sentree.nodeFeatures
        nodeFeatures_unnormalized = sentree.nodeFeatures_unnormalized
        W0 = np.zeros([self.d,self.d])
        W = np.zeros([self.d,self.d,3])
        W[:,:,0] = W0
        W[:,:,1] = W1
        W[:,:,2] = W2
        DEL = [np.zeros([self.d,1]), sentree.node_y1c1, sentree.node_y2c2]

        while np.size(toPopulate,1)!=0:
            parentNode = toPopulate[:,0].copy()
            mat = W[:,:,parentNode[1]]
            delt = DEL[parentNode[1]][:,parentNode[2]]

            if parentNode[0]>sl-1: # Non-leaf?

                kids = sentree.kids[parentNode[0],:]
                kid1 = [kids[0], 1, parentNode[0]]
                kid2 = [kids[1], 2, parentNode[0]]

                #toPopulate = np.array([kid1, kid2, toPopulate[:, 1:]])
                toPopulate[:,0]=kid2
                toPopulate=np.insert(toPopulate,0,kid1,1)
                a1_unnormalized = nodeFeatures_unnormalized[:,parentNode[0]] # unnormalized feature of pp
                a1 = nodeFeatures[:,parentNode[0]] # normalized feature of pp

                nd1 = sentree.nodeDelta_out1[:,parentNode[0]] # grad c1
                nd2 = sentree.nodeDelta_out2[:,parentNode[0]] # grad c2
                pd = sentree.parentDelta[:,parentNode[0]]


                if updateWcat:
                    smd = sentree.catDelta[:,parentNode[0]];
                    gradblab =gradblab + smd
                    parent_d = np.dot(norm1tanh_prime(a1_unnormalized) , (np.dot(W3,nd1) +
                                                                   np.dot(W4,nd2) +
                                                                   np.dot(mat,pd) + np.dot(np.transpose(Wlab),smd) - delt))
                    gradWlab = gradWlab + np.outer(smd,a1)

                else:
                    parent_d = np.dot(norm1tanh_prime(a1_unnormalized) , (np.dot(W3,nd1) +
                                                                   np.dot(W4,nd2) +
                                                                   np.dot(mat,pd) - delt))

                gradb1 = gradb1 + parent_d
                gradb2 = gradb2 + nd1
                gradb3 = gradb3 + nd2

                sentree.parentDelta[:,toPopulate[0][0]] = parent_d

                sentree.parentDelta[:,toPopulate[0][1]] = parent_d

                gradW1 = gradW1 + np.outer(parent_d,nodeFeatures[:,toPopulate[0][0]])
                gradW2 = gradW2 + np.outer(parent_d,nodeFeatures[:,toPopulate[0][1]])
                gradW3 = gradW3 + np.outer(nd1,a1)
                gradW4 = gradW4 + np.outer(nd2,a1)
            else: # leaf
                if updateWcat:
                    gradWlab = gradWlab + np.outer(sentree.catDelta[:, parentNode[0]], nodeFeatures[:,parentNode[0]])
                    gradblab = gradblab + sentree.catDelta[:,parentNode[0]]
                    gradL[:,toPopulate[0][0]] = gradL[:,toPopulate[0][0]] +(np.dot(mat,sentree.parentDelta[:,toPopulate[0][0]]) + np.dot(np.transpose(Wlab),sentree.catDelta[:,toPopulate[0][0]]) - delt)
                else:
                    gradL[:,toPopulate[0][0]] = gradL[:,toPopulate[0][0]] +(np.dot(mat,sentree.parentDelta[:,toPopulate[0][0]]) - delt)

                toPopulate=np.delete(toPopulate,0,1)
        return (gradW1,gradW2,gradW3,gradW4,gradWlab,gradb1,gradb2,gradb3,gradblab,gradL)

    def computeGrad(self,theta,X,y,allKids,updateWcat,alpha,freq):
        (W1,W2,W3,W4,Wlab,b1,b2,b3,blab,WL)=self.getParams(theta)
        cost_total = 0
        gradW1 = np.zeros([self.d,self.d])
        gradW2 = np.zeros([self.d,self.d])
        gradW3 = np.zeros([self.d,self.d])
        gradW4 = np.zeros([self.d,self.d])
        gradb1 = np.zeros(self.d)
        gradb2 = np.zeros(self.d)
        gradb3 = np.zeros(self.d)
        gradblab = np.zeros(self.cat)
        gradWlab = np.zeros([self.cat,self.d])
        gradblab_total = np.zeros(self.cat)
        gradWlab_total = np.zeros([self.cat,self.d])

        #gradTheta=np.zeros(self.d*self.d*4+3*self.d+self.cat+self.cat*self.d+self.d*self.vocab)
        grad_We_total = np.zeros([self.d, self.vocab])
        num_nodes = 0
        num_sent=np.size(X,0)

        if allKids==None:
            allKids=[[]]*num_sent

        for i in range(num_sent):
            x=X[i]
            sl=len(x)
            L=WL[:,x]
            grad_We=np.zeros([self.d, self.vocab])
            gradL=np.zeros([self.d, sl])
            words_embedded = self.words_vectors[:,x]+L
            if sl>1:
                if updateWcat:
                    tree_sent = self.forwardProp(allKids[i],words_embedded,updateWcat,y[i],theta,freq)
                    cost = sum(tree_sent.nodeScores)  # Include leaf node error (supervised)
                    #num_nodes = num_nodes + 1
                else:
                    tree_sent = self.forwardProp(allKids[i],words_embedded,updateWcat,y[i],theta,freq)
                    allKids[i]=tree_sent.kids
                    cost = sum(tree_sent.nodeScores[sl:])  # Include leaf node error (supervised)
                    #num_nodes = num_nodes + sl

                (gradW1,gradW2,gradW3,gradW4,gradWlab,gradb1,gradb2,gradb3,gradblab,gradL)=\
                    self.backProp(tree_sent,updateWcat,words_embedded,gradW1,gradW2,gradW3,gradW4,gradWlab,gradb1,gradb2,gradb3,gradblab,gradL,theta)

                for l in range(sl):
                     grad_We[:, x[l]] = grad_We[:, x[l]] + gradL[:,l]
                #data=np.reshape(grad_We[:, x]+gradL,self.d*sl)
                #row=np.reshape(np.transpose([[1]]*sl*np.array(range(self.d))),self.d*sl)
                #col=np.array([x[l] for l in range(sl)]*self.d)
                #grad_We=csc_matrix((data,(row,col)),shape=(self.d, self.vocab))
                grad_We_total = grad_We_total + grad_We

                cost_total = cost_total + cost
            #else:
                #num_nodes+=1

        # if not updateWcat:
        #     num_nodes = num_nodes - num_sent
        if not updateWcat:
            num_nodes=np.sum(map(len,X))-num_sent
        else:
            num_nodes=num_sent

        cost_total = 1.0/num_nodes*cost_total + self.lambdaW*alpha/2.0 * ( np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))

        gradW1_total=1.0/num_nodes*(gradW1)+self.lambdaW*alpha*W1
        gradW2_total=1.0/num_nodes*(gradW2)+self.lambdaW*alpha*W2
        gradW3_total=1.0/num_nodes*(gradW3)+self.lambdaW*alpha*W3
        gradW4_total=1.0/num_nodes*(gradW4)+self.lambdaW*alpha*W4
        gradb1_total=1.0/num_nodes*(gradb1)
        gradb2_total=1.0/num_nodes*(gradb2)
        gradb3_total=1.0/num_nodes*(gradb3)

        #gradWlab_total=None
        #gradblab_total=None
        if updateWcat:
            gradWlab_total=1.0/num_nodes *  (gradWlab+self.lambdaCat*alpha*Wlab)
            gradblab_total=1.0/num_nodes *  (gradblab)
            cost_total = cost_total + self.lambdaCat*alpha/2.0 * np.sum(Wlab**2)

        cost_total = cost_total +  self.lambdaL*alpha/2.0 * np.sum(WL**2)
        gradL_total = 1.0/num_nodes*grad_We_total+ self.lambdaL*alpha * WL
        return (gradW1_total,gradW2_total,gradW3_total,gradW4_total,gradWlab_total,gradb1_total,gradb2_total,gradb3_total,gradblab_total,gradL_total,cost_total,allKids)

    def RAECost(self,theta,X,y,freq):
        (gradW1,gradW2,gradW3,gradW4,gradWlab,gradb1,gradb2,gradb3,gradblab,gradL,cost,allKids)=\
            self.computeGrad(theta,X,y,None,False,self.alpha,freq)

        (gradW1_1,gradW2_1,gradW3_1,gradW4_1,gradWlab_1,gradb1_1,gradb2_1,gradb3_1,gradblab_1,gradL_1,cost_1,allKids_1)=\
            self.computeGrad(theta,X,y,allKids,True,1-self.alpha,freq)

        gradW1_total=gradW1+gradW1_1
        gradW2_total=gradW2+gradW2_1
        gradW3_total=gradW3+gradW3_1
        gradW4_total=gradW4+gradW4_1
        if gradWlab == None:
            gradWlab_total=gradWlab_1
            gradblab_total=gradblab_1
        else:
            gradWlab_total=gradWlab+gradWlab_1
            gradblab_total=gradblab+gradblab_1
        gradb1_total=gradb1+gradb1_1
        gradb2_total=gradb2+gradb2_1
        gradb3_total=gradb3+gradb3_1

        gradL_total=gradL+gradL_1
        cost_total=cost+cost_1

        grad_total=np.hstack([np.reshape(gradW1_total,self.d*self.d),np.reshape(gradW2_total,self.d*self.d),np.reshape(gradW3_total,self.d*self.d),
                              np.reshape(gradW4_total,self.d*self.d),np.reshape(gradWlab_total,self.cat*self.d),np.reshape(gradb1_total,self.d),
                              np.reshape(gradb2_total,self.d),np.reshape(gradb3_total,self.d),np.reshape(gradblab_total,self.cat),
                              np.reshape(gradL_total,self.d*self.vocab)])
        return(cost_total,grad_total)

    def getParams(self,theta):
        W1=np.reshape(theta[:self.d*self.d],[self.d,self.d])
        W2=np.reshape(theta[self.d*self.d:2*self.d*self.d],[self.d,self.d])
        W3=np.reshape(theta[2*self.d*self.d:3*self.d*self.d],[self.d,self.d])
        W4=np.reshape(theta[3*self.d*self.d:4*self.d*self.d],[self.d,self.d])
        Wlab=np.reshape(theta[4*self.d*self.d:self.d*(4*self.d+self.cat)],[self.cat,self.d])
        b1=np.reshape(theta[self.d*(4*self.d+self.cat):self.d*(4*self.d+self.cat+1)],self.d)
        b2=np.reshape(theta[self.d*(4*self.d+self.cat+1):self.d*(4*self.d+self.cat+2)],self.d)
        b3=np.reshape(theta[self.d*(4*self.d+self.cat+2):self.d*(4*self.d+self.cat+3)],self.d)
        blab=np.reshape(theta[self.d*(4*self.d+self.cat+3):self.d*(4*self.d+self.cat+3)+self.cat],self.cat)
        WL=np.reshape(theta[self.d*(4*self.d+self.cat+3)+self.cat:],[self.d,self.vocab])
        return(W1,W2,W3,W4,Wlab,b1,b2,b3,blab,WL)

    def fit(self,X,y):
        from sklearn.feature_extraction.text import CountVectorizer
        cv=CountVectorizer(vocabulary=[str(r) for r in range(self.vocab)],tokenizer=lambda x:x.split())
        all_text=[]
        for x in X:
            all_text.extend([str(w) for w in x])
        freq=np.sum(cv.transform(" ".join(all_text)).toarray(),0).astype(float)
        freq=freq/sum(freq)

        #res=fmin_l_bfgs_b(func=self.RAECost,x0=self.combineParams(),args=(X,y,freq),approx_grad=False,disp=1)
        res=minimize(fun=self.RAECost,x0=self.combineParams(),args=(X,y,freq),method='L-BFGS-B',jac=True,options = {'maxiter' :70,'disp':True})
        theta=res.x
        self.theta=theta
        (self.W1,self.W2,self.W3,self.W4,self.Wlab,self.b1,self.b2,self.b3,self.blab,self.WL)=self.getParams(theta)
        X_trans=self.getTopNodeRep(X,freq,True)
        self.postClassifier.fit(X_trans,y)

    def getTopNodeRep(self,X,freq,train):
        rtn=[]
        for x in X:
            if train:
                L=self.WL[:,x]+self.words_vectors[:,x]
            else:
                L=self.WL[:,x]
            sl=len(x)
            sentree=self.forwardProp([],L,False,None,self.combineParams(),freq)
            topF=sentree.nodeFeatures[:,sl-1]
            avgF=np.mean(sentree.nodeFeatures,1)
            rtn.append(np.hstack([topF,avgF]))
        return rtn

    def predict(self,X):
        cv=CountVectorizer(vocabulary=[str(r) for r in range(self.vocab)],tokenizer=lambda x:x.split())
        all_text=[]
        for x in X:
            all_text.extend([str(w) for w in x])
        freq=np.sum(cv.transform(" ".join(all_text)).toarray(),0).astype(float)
        freq=freq/sum(freq)

        x_trans=self.getTopNodeRep(X,freq,False)
        pred=self.postClassifier.predict(x_trans)
        return pred

    def unsupAnalyseTSNE(self):
        from sklearn.manifold import TSNE
        model=TSNE(n_components=2)
        return model.fit_transform(np.transpose(self.WL))

    def unsupAnalysePCA(self):
        from sklearn.decomposition import PCA
        model=PCA(n_components=2,copy=True,whiten=False)
        return model.fit_transform(np.transpose(self.WL))

    def supAnalyser(self,X,freq,vocabulary,top=20):
        result_score=[]
        result_word=[]
        for i in range(self.cat):
            result_score.append([0.0]*top)
            result_word.append(['']*top)

        num_sent=np.size(X,0)
        allKids=[[]]*num_sent

        for i in range(num_sent):
            x=X[i]
            sl=len(x)
            words_embedded=self.WL[:,x]
            unsup_tree = self.forwardProp([],words_embedded,False,None,self.theta,freq)
            allKids[i]=unsup_tree.kids

            sup_tree=rnntree.rnntree(self.d,sl,words_embedded)

            nodeUnder = np.ones([2*sl-1,1])

            for j in range(sl,2*sl-1): # calculate n1, n2 and n1+n2 for each node in the sensentree and store in nodeUnder
                kids = allKids[i][j]
                n1 = nodeUnder[kids[0]]
                n2 = nodeUnder[kids[1]]
                nodeUnder[j] = n1+n2

            #sentree.catDelta = np.zeros([cat_size, 2*sl-1])
            #sentree.catDelta_out = np.zeros([self.d,2*sl-1])

            for j in range(2*sl-1):
                kids = allKids[i][j]

                c1 = sup_tree.nodeFeatures[:,kids[0]]
                c2 = sup_tree.nodeFeatures[:,kids[1]]

                # Eq. [2] in the paper: p = f(W[1][c1 c2] + b[1])
                p = tanh(np.dot(self.W1,c1) + np.dot(self.W2,c2) + self.b1)

                # See last paragraph in Section 2.3
                p_norm1 = p/norm(p)

                # Eq. (7) in the paper (for special case of 1d label)
                #sm = sigmoid(np.dot(Wlab,p_norm1) + blab)
                sm=softmax(np.dot(self.Wlab,p_norm1) + self.blab)
                max_score=max(sm)
                ind=list(sm).index(max_score)
                min_score=min(result_score[ind])
                if max_score>min_score:
                    min_ind=result_score[ind].index(min_score)
                    result_score[ind][min_ind]=max_score
                    if j<sl:
                        result_word[ind][min_ind]=vocabulary[x[j]]
                    else:
                        stk=[]
                        stk.extend(list(kids))
                        stk.reverse()
                        words=[]
                        while len(stk)!=0:
                            current=stk.pop()
                            if current<sl:
                                words.append(vocabulary[x[current]])
                            else:
                                toExtend=[]
                                toExtend.extend(list(allKids[i][current]))
                                toExtend.reverse()
                                stk.extend(toExtend)

                        result_word[ind][min_ind]=' '.join(words)
        return (result_score,result_word)


    def analyse(self,X,vocabulary):
        cv=CountVectorizer(vocabulary=[str(r) for r in range(self.vocab)],tokenizer=lambda x:x.split())
        all_text=[]
        for x in X:
            all_text.extend([str(w) for w in x])
        freq=np.sum(cv.transform(" ".join(all_text)).toarray(),0).astype(float)
        freq=freq/sum(freq)
        vectors1=self.unsupAnalysePCA()
        vectors2=self.unsupAnalysePCA()

        (score,words)=self.supAnalyser(X,freq,vocabulary,10)
        return (vectors1,vectors2,score,words)





