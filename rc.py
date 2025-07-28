import torch as pt

class RC:
    # device : cuda or cpu 
    # dim_i : dimension of dynamical system
    # dim_r : dimension/size of reservoir
    # rho : spectral radius of reservoir weight matrix
    # sparsity : sparsity of reservoir weight matrix
    # sigma : input strength/scaling of input x(t)
    # bias : strength of input bias
    def __init__(self, device, seed, dim_i, dim_r, rho, sparsity, sigma, bias):
        self.device = device
        pt.manual_seed(seed)
        self.dim_i = dim_i
        self.dim_r = dim_r
        self.bias = bias * pt.ones((dim_r, 1), device = device)
        # Each element of W_i chosen uniformly from [-sigma, sigma]
        self.W_i = pt.Tensor(dim_r, dim_i).uniform_(-sigma, sigma)
        self.W_i = self.W_i.to(device)
        # Each element of W_r chosen uniformly from [-1, 1]
        W_r_ = pt.Tensor(dim_r, dim_r).uniform_(-1, 1)
        # create a mask to setup dropout with P(dropout) = 1 - sparsity
        mask = pt.Tensor(dim_r, dim_r).bernoulli(1 - sparsity)
        # multiply W_r_ and mask to obtain W_r with sparsity as specified
        W_r = W_r_ * mask
        # scaling matrix W_r such that rho becomes the spectral radius of W_r
        eigvals = pt.linalg.eigvals(W_r)
        max_eigval = pt.max(pt.abs(eigvals))
        if(max_eigval != 0):
            W_r = (W_r / max_eigval)  
        W_r = W_r * rho
        self.W_r = W_r.to(device)
        # reservoir and states at some time t
        self.r_t = self.r_T1 = self.r_train = pt.empty((dim_r, 1), device = device)
        # output weight matrix
        self.W_o = pt.empty((dim_i, dim_r), device = device)
        
    def __next_r(self, x_t):
        self.test = self.r_t
        # r(t + 1) = tanh(W_r * r(t) + W_i * x(t) + bias)
        r_ = pt.tanh(self.W_r @ self.r_t + self.W_i @ x_t + self.bias)
        self.r_t = r_
            
    def train(self, x_transient, x_train, r0):
          # ignore transient x values and only update r_t
          self.r_t = r0
          x_transient_ = pt.hsplit(x_transient, x_transient.shape[1])
          for i in range(x_transient.shape[1]):
              self.__next_r(x_transient_[i])
          self.r_train = self.r_t
          # training over non transint
          x_train_ = pt.hsplit(x_train, x_train.shape[1])
          for i in range(x_train.shape[1] - 1):
            self.__next_r(x_train_[i])     
            self.r_train = pt.column_stack((self.r_train, self.r_t))
          self.r_T1 = self.r_t
          # X = W_o R
          # X R.T = W_o R R.T   normal eq
          # W_o = X R.T (R R.T)^(-1)
          self.W_o = x_train @ self.r_train.T @ pt.linalg.inv(self.r_train @ self.r_train.T)
    
    def test_training(self):
        # X = W_o Q
        return self.W_o @ self.r_train
         
    def one_step_prediction(self, x_one_step, r0 = None):
        # r(t + 1) = tanh(W_r * r(t) + W_i * x(t) + bias)
        # x_pred(t + 1) = # W_o r(t + 1)
          self.r_t = self.r_T1 if r0 == None else r0
          R = self.r_t
          x_t = pt.hsplit(x_one_step, x_one_step.shape[1])
          for i in range(x_one_step.shape[1] - 1):
              self.__next_r(x_t[i])
              R = pt.column_stack((R, self.r_t))
          return self.W_o @ R, R

    def reservoir_prediction(self, num_pred, r0 = None):
        # r(t + 1) = tanh(W_r * r(t) + W_i * (W_o * r(t)) + bias)
        # x_pred(t + 1) = # W_o r(t + 1)
        self.r_t = self.r_T1 if r0 == None else r0
        R = r = self.r_t
        x_pred = pt.empty((self.dim_i, num_pred + 1), device = self.device)
        x_t = self.W_o @ r
        x_pred[:, 0] = x_t.T
        for i in range(num_pred):
            self.__next_r(x_t)
            r = self.r_t
            x_t = self.W_o @ r
            x_pred[:, i + 1] = x_t.T
            R = pt.column_stack((R, r))
        return x_pred, R
