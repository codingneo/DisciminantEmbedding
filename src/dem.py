import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict


class model(object):

	def __init__(self, nh, ne, de, cs):
		'''
			Model Initialization

			nh :: dimension of the hidden layer
      ne :: number of word embeddings in the vocabulary
      de :: dimension of the word embeddings
      cs :: column size 
		'''
		self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
						 	 (ne, de)).astype(theano.config.floatX))

		self.Wx = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
							(de*cs, nh)).astype(theano.config.floatX))

		self.Wh = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
							(nh, 1)).astype(theano.config.floatX))

		self.b   = theano.shared(numpy.zeros(1, dtype=theano.config.floatX))


		self.params = [ self.emb, self.Wx, self.Wh, self.b ]
		self.names = [ 'embeddings', 'Wx', 'Wh', 'b' ]

		# bundle
		idxs = T.ivector()
		x = self.emb[idxs].reshape((1, de*cs))
		y = T.iscalar('y')

		h = T.nnet.sigmoid(T.dot(x, self.Wx))
		p_y = T.nnet.sigmoid(T.dot(h, self.Wh) + self.b)

		# cost and gradients
		lr = T.scalar('lr')
		log_loss = T.mean(-y*T.log(p_y)-(1-y)*T.log(1-p_y))
		gradients = T.grad( log_loss, self.params )
		updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))

		# theano functions
		self.train = theano.function( inputs = [idxs, y, lr],
																	outputs = log_loss,
																	updates =  updates )

		self.classify = theano.function( inputs = [idxs],
																		 outputs = p_y )


	def save(self, foler):
		for param, name in zip(self.params, self.names):
			numpy.save(os.path.join(folder, name + '.npy'), param.get_value())


