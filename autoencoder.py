class autoencoder:
    def __init__(self, data, num_subfunctions_ratio=0.05, ntrain=1e6, patch_width=8, sigma=0.1):
        self.name = 'Denoising Autoencoder with vanHat'

        # standard deviation of noise to use for denoising autoencoder
        self.sigma = sigma

        # load data
        self.data = data
        self.patch_size = data.patch_size

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
        self.full_objective_references = random.sample(self.subfunction_references, int(num_subfunctions/100.))


	def f_df(self, theta, irange, holdout=False):
		X = self.data.get_data_range(irange[0], irange[1], holdout=holdout)

		# generate noise to add to X in a deterministic fashion based on irange
		rand_state = np.random.get_state()
		np.random.seed(irange)
		N = np.random.randn(X.shape[0], X.shape[1])*self.sigma
		np.random.set_state(rand_state)

		results = self.f_df_Theano(*(theta + [X, N]))
		return results[0], results[1:]

	def build_model(self):
		# repeatable experiments
		rng = np.random.RandomState(1234)

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
		err = ((L3 - X)**2).mean()

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
