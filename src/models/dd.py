import torch
from torch import Tensor
from torch.nn import Sigmoid, Linear
from torch.nn import functional as F


def hill_fn(x: torch.Tensor, n: float, k: float):
    return (x ** n) / (k + x ** n)


class Hill():
    """Hill cooperativity function with parameters n (Hill coefficient) and k (half saturation).
    
    Usage:
        hill = Hill(n=2.0, k=0.01)
        hill(x)

    Args:

            n: Hill coefficient
            k: half saturation
    """

    def __init__(self, n: int, k: float):
        self.n = n
        self.k = k

    def __call__(self, x: torch.Tensor):
        return hill_fn(x, self.n, self.k)


class DendriticFullyConnected(Linear):
    """Dendritic Fully Connected Layer. This extends the classical Linear layer to 
    include a 'clustering' mechanism. Groups adjascent synapses through a convolution
    with a fixed filter. The 'state' of the neuron is the usual weighted sum of inputs.
    The result of the convolution is added to the state and passed
    through a non-linear function to mimic cooperativity.

    Usage:
        dendritic = DendriticFullyConnected(in_features=10, out_features=20)
        inputs = torch.randn(32, 10)
        dendritic(inputs)
        # output is a tensor of size 32 x 20

    Args:

        in_features: size of each input feature (last dimension of input tensor)
        out_features: size of each output features (last dimension of output tensor)
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        conv_filter: the convolution filter to group adjacent synapses
            Default: ``torch.tensor([[[0.5, 0.5]]])``
        stride: stride of the convolution filter
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        conv_filter: Tensor = torch.tensor([[[0.5, 0.5]]]),
        stride: int = 1,
        **kwargs
    ):
        super().__init__(in_features, out_features, bias, **kwargs)     
        # just for info, the below is in the parent class constructor
        # factory_kwargs = {'device': device, 'dtype': dtype}
        # super().__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        # self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # if bias:
        #     self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()
        self.cluster_act_fn = Sigmoid()  # Tanh()  # Hill(n=2.0, k=0.01)  # Hill function instead of Tanh or Sigmoid?
        self.conv_filter = conv_filter  # conv filter is out_channels x in_channels x kernel
        self.stride = stride
        self.kernel_size = self.conv_filter.size(-1)
        self.padding = 0   #  to simplify for now
        conv_output_width = (self.in_features - self.kernel_size + 2 * self.padding) // self.stride + 1
        post_dim = conv_output_width
        # using same stride, kernel size and pooling for max pool as for conv to simplify
        # if conv_output_width >= self.kernel_size:
        #     self.max_pool_kernel_size = self.kernel_size
        #     maxpool_output_width = ((conv_output_width - self.kernel_size + 2 * self.padding) // self.stride) + 1
        # else:
        #     self.max_pool_kernel_size = conv_output_width
        #     maxpool_output_width = 1
        # post_dim = maxpool_output_width
        self.post_dim = post_dim

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.size(-1) == self.in_features, f"input has wrong number of in_features (dimension -1) (found {inputs.size(-1)} instead of required {self.in_features})"
        # input is typically a matrix of inputs, structured into batches of arrays of input features
        # with dimension B_atch x L_ength x in_F_eatures (B x L x in_F)
        # output will be B_atch x L_ength x out_F_eatures (B x L x out_F)
        # it could have more dimensions, so to simplify we flatten all dimensions except the last two
        original_shape = inputs.size()
        inputs = inputs.view(-1, self.in_features)  # --> ... x in_F, noted B x in_F in the following

        # The state is the linear weighted sum of inputs (the result of the classical linear fully connected layer)
        state = F.linear(inputs, self.weight, bias=self.bias)  # (inputs @ self.weight.T) + self.bias --> B x out_F

        # We now include a 'clustering' mechanism that introduces an aggressively
        # non-linear read out of the synaptic activities, gated by the state.
        # To identify clusters, we access the individual synaptic activities
        # via an element-wise multiplication to obtain a matrix of neurons x synapses
        # This will have dimension B x out_F x in_F
        # It requires a bit of tensor gymnastics with broadcasting rules (thank you CoPilot):
        # inputs need to be reshaped to B x 1 x in_F and weights to 1 x out_F x in_F
        x = inputs.view(-1, 1, self.in_features)  # --> B x 1 x in_F
        w = self.weight.unsqueeze(0)  # --> 1 x out_F x in_F
        synapses = torch.mul(x, w)  # element-wise multiplication --> B x out_F x in_F
        # note that we could use this to compute state = synapses.sum(-1)  # --> ... x out_F but F.linear is so fast, no need

        # To scan adjacent synapses along the in_Feature axis
        # we use a fixed convolution filter to aggregate across neighboring synapses
        # conv1D expects B x C x W, where C is the number of channels (in our case only 1, for now)
        # and W is the dimension along which the convolution is applied.
        # In our case the convolution dimension W will thus correspond to the in_F dimension
        # as we apply the convolution along the synaptic inputs for each neuron
        # Since synapses is B x out_F x in_F we need to introduce the channel dimension and reshape to B*out_F x 1 x in_F
        synapses = synapses.view(-1, 1, self.in_features)  # --> B*out_F x 1 x in_F
        cluster = F.conv1d(synapses, self.conv_filter, stride=self.stride).squeeze(1)  # B*out_F x in_F-n
        # cluster = F.max_pool1d(cluster, kernel_size=self.max_pool_kernel_size, stride=self.stride)  # B*out_F x in_F-n-m
        cluster = cluster.view(-1, self.out_features, self.post_dim)  # --> B x out_F x in_F-n

        # apply a non-lin function to represent cluster 'cooperativity', gated by the state
        # here, we refrain from including additional params such as cluster-specific weights
        state_expanded = state.view(-1, self.out_features, 1).repeat(1, 1, self.post_dim)  # --> B x out_F x in_F-n
        assert state_expanded.size() == cluster.size()
        assert state_expanded.size(-1) == self.post_dim
        assert state_expanded.size(1) == self.out_features
        assert state_expanded.size(0) == inputs.size(0), f"state_expanded has wrong number of batches (dimension 0) (found {state_expanded.size(0)} instead of required {inputs.size(0)}; full dims are {state_expanded.size()})"
        cluster_activity = self.cluster_act_fn(cluster + state_expanded)  # --> B x out_F x in_F-n

        # sum cluster activity over the synapsis i.e. over remaining in_F dimension (contracted given convolution)
        cluster_activity = cluster_activity.sum(-1)  # --> B x out_F
        
        # reshape back to original shape
        output = cluster_activity.view(original_shape[:-1] + (self.out_features,))

        # output is cluster activity
        return output  # --> ... x out_F