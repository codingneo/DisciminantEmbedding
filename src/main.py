import numpy
import time
import sys
import subprocess
import os
import random
from math import log

from util import data
from dem import model

def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

if __name__ == '__main__':

    train_file = './data/balloons/train.csv'
    valid_file = './data/balloons/valid.csv'
    test_file = './data/balloons/test.csv'

    s = {'lr':0.0627142536696559,
         'verbose':1,
         'decay':True, # decay on the learning rate if improvement stops
         'ne':8, # number of embedding dictionary
         'cs':4, # number of columns
         'nhidden':10, # number of hidden units
         'seed':345,
         'emb_dimension': 20, # dimension of word embedding
         'nepochs':50}

    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)


    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    dem = model(    nh = s['nhidden'],
                    ne = s['ne'],
                    de = s['emb_dimension'],
                    cs = s['cs'] )

    # train with early stopping on validation set
    best_valid_score = numpy.inf
    s['clr'] = s['lr']
    for e in xrange(s['nepochs']):
        # shuffle

        s['ce'] = e
        tic = time.time()

        count = 0
        for X, y in data.generate(train_file):
            # convert input to index
            #print 'X=%s' % (str(X).strip('[]'))
            #print 'shape of X: '


            new_X = numpy.asarray(X).astype('int32')

            old_pred = dem.classify(new_X)
            dem.train(new_X, y, s['clr'])
            new_pred = dem.classify(new_X)

            # print 'old pred: %f, label=%d, new pred: %f' % (old_pred, y, new_pred)
            # sys.stdout.flush()

            count += 1

        if s['verbose']:
            print '[learning] epoch %i >> %i'%(e,count),'completed in %.2f (sec) <<\r'%(time.time()-tic),
        #     sys.stdout.flush()
            
        print dem.Wx.get_value()


        # evaluation // back into the real world : idx -> words
        valid_loss = .0
        for X, y in data.generate(valid_file):
            pred = dem.classify(X)

            print 'label=%d, pred: %f' % (y, pred)
            sys.stdout.flush()

            valid_loss += logloss(pred, y)

        test_loss = .0
        for X, y in data.generate(test_file):
            pred = dem.classify(X)
            test_loss += logloss(pred, y)

        print 'valid score: ', valid_loss
        sys.stdout.flush()        

        if valid_loss<best_valid_score:
            #dem.save(folder)
            best_valid_score = valid_loss

            if s['verbose']:
                print 'NEW BEST: epoch', e, 'valid score', valid_loss, 'best test score', test_loss, ' '*20
            s['valid_loss'] = valid_loss
            s['test_loss'] = test_loss
            s['be'] = e
            #subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            #subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        else:
            print ''
        
        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5 
        if s['clr'] < 1e-5: break

    print 'BEST RESULT: epoch', e, 'valid loss', s['valid_loss'], 'best test loss', s['test_loss'], 'with the model', folder
