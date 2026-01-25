import torch
import torch.nn as nn


class ESNLayer(nn.Module):

    def __init__(self, n_in,n_res,n_out,spectral_radius=0.92,leak_rate=0.79,reg=1e-8,device='cpu',seed=None,input_scale=1.0, density=None):

        super().__init__()

        self.n_in= n_in # input dimension
        self.n_res= n_res # hidden (reservoir) dimension
        self.n_out= n_out # output dimension
        self.spectral_radius= spectral_radius # spectral radius
        self.leak_rate= leak_rate # leaking rate
        self.reg= reg # regularization coefficient(Ridge regression)
        self.device= torch.device(device)

        if seed is not None:
            torch.manual_seed(seed) # very important for reproducibility of reservoir weights

        self.W_in= self._init_input_weights(n_in,n_res,input_scale)
        self.W = self._init_reservoir_weights(n_res, density, spectral_radius)
        self.bias = (torch.rand(n_res, device=self.device) * 2 - 1) * input_scale
        self.W_out= None # output weights to be learned during training


    def _init_input_weights(self,n_in,n_res,input_scale):
        return (torch.rand(n_res, n_in, device=self.device) * 2 - 1) * input_scale # see research paper for details


    def _init_reservoir_weights(self, n_res, density, rho):
        """W with spectral radius"""
        if density is None:
            W = torch.rand(n_res, n_res, device=self.device) * 2 - 1  # full connectivity
        else:
            n_conn = int(n_res * n_res * density)       # sparse connectivity
            W = torch.zeros(n_res, n_res, device=self.device) # initialize to zero
            idx = torch.randperm(n_res * n_res, device=self.device)[:n_conn] # random indices
            W.view(-1)[idx] = torch.rand(n_conn, device=self.device) * 2 - 1  # assign random weights to those indices

        eigs = torch.linalg.eigvals(W.cpu())
        radius = torch.max(torch.abs(eigs)).item()

        if radius > 0:
            return W * (rho / radius)
        else:
            return W


    def forward(self, u, x):
        return (1-self.leak_rate) * x + self.leak_rate * torch.tanh(self.W_in @ u + self.W @ x + self.bias)  # reservoir update equation


    # input → reservoir → collect x(t)
    # X=[x(1),x(2),...,x(T)]

    # applying washout in next step before applying ridge regression to learn W_out

    def run(self, inputs, washout=0, x0=None):
        """Collect states"""
        states = []

        if x0 is None:
            x = torch.zeros(self.n_res, device=self.device) #[0,0,...,0]
        else:
            x = x0.clone()

        for t in range(len(inputs)):
            x = self.forward(inputs[t], x) # update reservoir state
            if t >= washout:
                states.append(x)

        return torch.stack(states), x


    def fit(self,inputs, targets, washout=0):

        X,_= self.run(inputs, washout) # collect states after washout
        # Wout​=(XTX+λI)−1XTY

        X = torch.cat([torch.ones(len(X), 1, device=self.device), X], dim=1)
        Y = targets[washout:len(inputs)]

        self.W_out = torch.linalg.solve(
            X.T @ X + self.reg * torch.eye(X.shape[1], device=self.device),
            X.T @ Y
        )


    def predict(self, inputs, x0=None):
        """y(t) = W_out * [1, x(t)]"""
        X, _ = self.run(inputs, 0, x0)
        X = torch.cat([torch.ones(len(X), 1, device=self.device), X], dim=1)
        return X @ self.W_out
