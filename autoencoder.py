import theano
import theano.tensor as T
import numpy as np

class autoencoder:
    def __init__(self, data, num_subfunctions_ratio=0.05, ntrain=1e6, 
        patch_width=8, sigma=0.1, overcomplete=2, batch_size=200, rng=np.random.RandomState(23455)):
        self.name = 'Denoising Autoencoder with vanHat'

        # standard deviation of noise to use for denoising autoencoder
        self.sigma = sigma

        # load data
        self.data = data
        self.patch_size = data.patch_size

        # Factor of overcompleteness for hidden units
        self.overcomplete = overcomplete
        self.rng = rng

        # create objective function, and initialize parameters
        self.build_model()

        # break the data up into minibatches

        self.N = int(np.ceil(np.sqrt(batch_size)*num_subfunctions_ratio))
        self.subfunction_references = []
        samples_per_subfunction = int(np.floor(batch_size/self.N))
        for mb in range(self.N):
            start_idx = mb*samples_per_subfunction
            end_idx = (mb+1)*samples_per_subfunction
            self.subfunction_references.append((start_idx, end_idx,))
        #self.full_objective_references = self.subfunction_references
        #self.full_objective_references = self.rng.sample(self.subfunction_references, int(num_subfunctions/100.))


    def f_df(self, theta, irange, holdout=False):
        X = self.data.get_data_range(irange[0], irange[1], holdout=holdout)

        # generate noise to add to X in a deterministic fashion based on irange
        N = self.rng.randn(X.shape[0], X.shape[1])*self.sigma

        results = self.f_df_Theano(*(theta + [X, N]))
        return results[0], results[1:]

    def build_model(self):
        # repeatable experiments
        rng = self.rng

        X = T.matrix('x')
        N = T.matrix('n')

        params = []

        L1 = HiddenLayer(rng=rng, input=(X+N),
                                       n_in=self.patch_size, n_out=self.overcomplete*self.patch_size,
                                       activation=T.nnet.sigmoid, b_mean=-2., scl=1.)
        params += L1.params
        L2 = HiddenLayer(rng=rng, input=L1.output,
                                       n_in=(L1.n_out), n_out=(L1.n_out),
                                       activation=T.nnet.sigmoid, b_mean=-2., scl=1.)
        params += L2.params
        L3 = HiddenLayer(rng=rng, input=L2.output,
                                       n_in=(L2.n_out), n_out=self.patch_size,
                                       activation=T.nnet.sigmoid, b_mean=-2., scl=1.)
        params += L3.params

        # reconstruction error
        err = ((L3.output - X)**2).mean()

        # compute the gradient of cost with respect to theta (sorted in params)
        # the resulting gradients will be stored in a list gparams
        gparams = []
        for param in params:
            gparam = T.grad(err, param)
            gparams.append(gparam)

        def convert_variable(x):
            if x.ndim == 1:
                return T.vector(x.name, dtype=x.dtype)
            else:
                return T.matrix(x.name, dtype=x.dtype)
        symbolic_params = [convert_variable(param) for param in params]
        givens = dict(zip(params, symbolic_params))
        self.f_df_Theano = theano.function(inputs=symbolic_params + [X, N], outputs=[err] + gparams, givens=givens ) #,  on_unused_input='warn')

        self.theta_init = [param.get_value() for param in params]
        self.theta = [param.get_value() for param in params]


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b_mean=0.,
                 activation=T.tanh, scl=1.):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        self.n_in = n_in
        self.n_out = n_out

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(1. / (n_in)),
                    high=np.sqrt(1. / (n_in)),
                    size=(n_in, n_out)), dtype=theano.config.floatX) * scl
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = b_mean*np.ones((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]
