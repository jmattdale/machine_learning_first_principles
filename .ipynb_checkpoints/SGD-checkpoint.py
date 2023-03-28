import numpy as np

class SGD:
    def __init__(self,
                 gradient,
                 dtype,
                 random_state):
        
        self.gradient = gradient
        self.dtype_ = np.dtype(dtype)
        self.learn_rate = np.array(learn_rate, dtype=dtype_)
        self.decay_rate = np.array(decay_rate, dtype=dtype_)
        self.batch_size = int(batch_size)
        self.n_iter = int(n_iter)
        self.tolerance = np.array(tolerance, dtype=dtype_)
        self.validate_params()
        
        # Initializing the random number generator
        self.seed = None if random_state is None else int(random_state)
        self.rng = np.random.default_rng(seed=seed)

        # Initializing the values of the variables
        self.weights_vector = (
            rng.normal(size=int(n_vars)).astype(dtype_)
            if start is None else
            np.array(start, dtype=dtype_)
        )
    
    def validate_init_params(self):
        # gradient
        if not callable(self.gradient):
            raise TypeError("'gradient' must be callable")
        
        # learn rate
        if np.any(self.learn_rate <= 0):
            raise ValueError("'learn_rate' must be greater than zero")
            
        # decay rate for momentum
        if np.any(self.decay_rate < 0) or np.any(self.decay_rate > 1):
            raise ValueError("'decay_rate' must be between zero and one")
        
        # iterations     
        if self.n_iter <= 0:
            raise ValueError("'n_iter' must be greater than zero")
        
        # tolerance
        if np.any(self, tolerance <= 0):
            raise ValueError("'tolerance' must be greater than zero")
            
    def format_inputs(self, x,y):
        # Converting x and y to NumPy arrays
        x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
        n_obs = x.shape[0]
        if n_obs != y.shape[0]:
            raise ValueError("'x' and 'y' lengths do not match")
        xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
    
        return xy
    
    def train(x, y, batch_size):
        #===================================
        # validate params and prepare inputs
        #===================================
        n_samples = x.shape[0]
        xy = self.format_inputs(x,y)
        
        if not 0 < batch_size <= n_samples
            raise ValueError(
                "'batch_size' must be greater than zero and less than "
                "or equal to the number of observations"
            )
            
        #===================================
        # Run training
        #===================================
        diff = 0

        # Performing the gradient descent loop
        for _ in range(self.n_iter):
            # Shuffle x and y
            self.rng.shuffle(xy)

            # Performing minibatch moves
            for start in range(0, n_samples, batch_size):
                stop = start + batch_size
                x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

                # Recalculating the difference
                grad = np.array(self.gradient(x_batch, y_batch, self.weights_vector), self.dtype_)
                diff = decay_rate * diff - learn_rate * grad

                # Checking if the absolute difference is small enough
                if np.all(np.abs(diff) <= self.tolerance):
                    break

                # Updating the values of the variables
                self.weights_vector += diff

        return self.weights_vector if self.weights_vector.shape else self.weights_vector.item()