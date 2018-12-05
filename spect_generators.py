import pprint
import sys
import numpy as np
from itertools import product
from sklearn.metrics import f1_score, fbeta_score, confusion_matrix

attrib_names = [
'class'
] + ['F'+str(i) for i in range(1,23)] 


def make_intent(example):
    global attrib_names
    attribs = set([i+':'+str(k) for i,k in zip(attrib_names,example)])
    return attribs
    
def check_hypothesis(context_plus, context_minus, example):
  #  print example
    eintent = make_intent(example)
  #  print eintent
    eintent.discard('class:1')
    eintent.discard('class:0')
    labels = {}
    global cv_res
    for e in context_plus:
        ei = make_intent(e)
        candidate_intent = ei & eintent
        closure = [ make_intent(i) for i in context_minus if make_intent(i).issuperset(candidate_intent)]
        closure_size = len([i for i in closure if len(i)])
    #    print closure
        #print closure_size * 1.0 / len(context_minus)
        res = reduce(lambda x,y: x&y if x&y else x|y, closure ,set())
        for cs in ['1','0']:
            if 'class:'+cs in res:
                # labels[cs] = True
                labels[cs+'_res'] = candidate_intent
                labels[cs+'_total_weight'] = labels.get(cs+'_total_weight',0) +closure_size * 1.0 / len(context_minus) / len(context_plus)
    for e in context_minus:
        ei = make_intent(e)
        candidate_intent = ei & eintent
        closure = [ make_intent(i) for i in context_plus if make_intent(i).issuperset(candidate_intent)]
        closure_size = len([i for i in closure if len(i)])
        #print closure_size * 1.0 / len(context_plus)
        res = reduce(lambda x,y: x&y if x&y else x|y, closure, set())
        for cs in ['1','0']:
            if 'class:'+cs in res:
                # labels[cs] = True
                labels[cs+'_res'] = candidate_intent
                labels[cs+'_total_weight'] = labels.get(cs+'_total_weight',0) +closure_size * 1.0 / len(context_plus) / len(context_minus)
    # print eintent
    return [float(example[0]), labels.get("0_total_weight", 0), labels.get("1_total_weight", 0)]

def limited_acc(x, *args):
    m = args[0]
    # print m
    inferred = np.logical_or(m[:,1]< x[0], m[:,2]> x[1])
    # return inferred
    return np.sum((m[:,0] > 0) == inferred)*1.0/len(inferred)



def limited_f05(x, *args):
    m = args[0]
    inferred = np.logical_or(m[:,1]< x[0], m[:,2]> x[1])
    true_vals = m[:,0] > 0
    return fbeta_score(true_vals, inferred, 0.5)

def reversed_f1(x, *args):
    m = args[0]
    inferred = np.logical_or(m[:,1]< x[0], m[:,2]> x[1])
    true_vals = m[:,0] > 0
    return fbeta_score(np.logical_not(true_vals), np.logical_not(inferred), 2)

def limited_confusion_matrix(x, *args):
    m = args[0]
    inferred = np.logical_or(m[:,1]< x[0], m[:,2]> x[1])
    true_vals = m[:,0] > 0
    return confusion_matrix(true_vals, inferred)


#sanity check:
#check_hypothesis(plus_examples, minus_examples, plus_examples[3])

def load_spect_fold(i=2):
    index = str(i)

    q=open("spect.tsv_train_"+index+".txt","r")
    train = [ a.strip().split(",") for a in q]
    plus_elems = [a for a in train if a[0]=="1"]
    minus_elems = [a for a in train if a[0]=="0"]

    #print t
    q.close()
    w=open("spect.tsv_validation_"+index+".txt","r")
    validation = [a.strip().split(",") for a in w]
    w.close()
    return plus_elems, minus_elems, validation

def load_spect_official():
    q=open("SPECT.train","r")
    train = [ a.strip().split(",") for a in q]
    plus_elems = [a for a in train if a[0]=="1"]
    minus_elems = [a for a in train if a[0]=="0"]

    #print t
    q.close()
    w=open("SPECT.test","r")
    validation = [a.strip().split(",") for a in w]
    w.close()
    return plus_elems, minus_elems, validation

def pretty_acc(limits, scores):
    alltrue_acc = np.sum(scores[:,0] > 0)*1.0/len(scores)
    return "{:.3f} ({} elems, 'alltrue' gives {:3f}) with limits {}".format(
        limited_acc(limits, scores), len(scores), alltrue_acc, limits
    )

def pretty_f05(limits, scores):
    alltrue_f05 = fbeta_score(scores[:,0] > 0, [True]*len(scores), 0.5)
    return "{:.3f} ({} elems, 'alltrue' gives {:3f}) with limits {}".format(
        limited_f05(limits, scores), len(scores), alltrue_f05, limits
    )

def calc_scores(plus, minus, unknown):
    scores =[]
    for elem in unknown:
        scores.append(check_hypothesis(plus, minus, elem))
    return scores

if __name__ == '__main__':
    cv_scores = []
    FIND_BEST = True
    for i in range(7):
        cv_scores += calc_scores(*load_spect_fold(i))
    cv_scores_matr = np.array(cv_scores)
    if FIND_BEST:
        maxacc = 0
        for x, y in product(np.arange(0, 1, 0.01), np.arange(0, .2, 0.001)):
            acc = limited_f05([x, y], cv_scores_matr)
            if acc > maxacc:
                maxacc = acc
                best = [x, y]
        print "Best accuracy over all {} CV folds: ".format(i+1) + pretty_acc(best, cv_scores_matr)
        print "f05 over all CV folds: " + pretty_f05(best, cv_scores_matr)
        print limited_confusion_matrix(best, cv_scores_matr)
    else:
        print limited_acc([0.19, 0.05], cv_scores_matr)

    off_scores_matr = np.array(calc_scores(*load_spect_official()))
    if FIND_BEST:
        print "Accuracy for the official test: " + pretty_acc(best, off_scores_matr)
        print "f05 for the official test: " + pretty_f05(best, off_scores_matr)
        print limited_confusion_matrix(best, off_scores_matr)
    else:
        print "Accuracy for official test: {}".format(limited_acc([0.19, 0.05], off_scores_matr))
