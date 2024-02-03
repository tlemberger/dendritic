import torch
from torch import Tensor
from torch.nn import Module, Parameter, Sigmoid, Linear, Tanh, ReLU, Conv1d
from torch.nn import functional as F


def hill_fn(x: torch.Tensor, n, k_d):
    # check that x is positive (otherwise we can get NaNs)
    assert all((x >= 0).view(-1)), f"x has negative values (found min {x.min()})"
    return (x ** n) / (k_d + x ** n)


class Hill():
    """Hill cooperativity function with parameters n (Hill coefficient) and k_a (half saturation).

    Usage:
        hill = Hill(n=2.0, k_a=0.01)
        hill(x)

    Args:

            n: Hill coefficient
            k_a: half saturation
    """

    def __init__(self, n: float, k_a: float):
        self.n = n
        self.k_a = k_a

    def __call__(self, x: torch.Tensor):
        # n_modif = self.n * (1+state)  # 1..n*2 since state is -1..0
        k_d = self.k_a ** self.n  # * n_modif
        y = hill_fn(x, n, k_d)
        return y


class MyRELU():

    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor, state: torch.Tensor):
        return F.relu(x) * (1 + state)


# conv filter is out_channels x in_channels x kernel
DEFAULT_CONV_FILTER = torch.tensor(
    [
        [
            [1.0] * 2
        ]
    ]
)


class DendriticFullyConnected(Module):
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
        clustering_frac: float = 0.1,
        conv_filter: Tensor = None,
        stride: int = 1,
        cluster_act_fn=Hill(2, 0.5),
        device=None,
        dtype=None,
        **kwargs
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.clustering_frac = clustering_frac
        self.bias = bias
        assert 0 < self.clustering_frac < 1, f"clustering_frac should be between 0 and 1 (found {self.clustering_frac})"
        assert self.in_features > 1 / self.clustering_frac, f"clustering_frac is too large (found {self.clustering_frac} for {self.in_features} in_features)"
        self.in_features_clustering = int(self.in_features * self.clustering_frac)
        self.in_features_non_clustering = self.in_features - self.in_features_clustering
        self.nmda = Linear(self.in_features_clustering, self.out_features, bias=self.bias, **factory_kwargs)
        self.non_nmda = Linear(self.in_features_non_clustering, self.out_features, bias=self.bias, **factory_kwargs)

        self.cluster_act_fn = cluster_act_fn  #  Sigmoid()  # Tanh()  # Hill(n=2.0, k=0.01)  # Hill function instead of Tanh or Sigmoid?
        conv_filter = conv_filter if conv_filter is not None else DEFAULT_CONV_FILTER
        # conv_filter = conv_filter.to(**factory_kwargs)
        self.conv_filter = conv_filter  # conv filter is out_channels x in_channels x kernel
        self.stride = stride
        self.kernel_size = self.conv_filter.size(-1)
        self.padding = 0  # to simplify for now
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
        self.reset_parameters()

    def forward(self, inputs: Tensor) -> Tensor:
        # convolution filter weights have to be same device and dtype as inputs
        self.conv_filter = self.conv_filter.to(inputs.device, inputs.dtype)
        assert inputs.device == self.conv_filter.device, f"inputs and conv_filter have different devices (found {inputs.device} and {self.conv_filter.device} respectively)"
        assert inputs.dtype == self.conv_filter.dtype, f"inputs and conv_filter have different dtypes (found {inputs.dtype} and {self.conv_filter.dtype} respectively)"
        assert inputs.size(-1) == self.in_features, f"input has wrong number of in_features (dimension -1) (found {inputs.size(-1)} instead of required {self.in_features})"
        # input is typically a matrix of inputs, structured into batches of arrays of input features
        # with dimension B_atch x L_ength x in_F_eatures (B x L x in_F)
        # output will be B_atch x L_ength x out_F_eatures (B x L x out_F)
        # it could have more dimensions, but first one is batch B and last one is in_F
        original_shape = inputs.size()

        # split the input into nmda (clustering) vs non nmda (non-clustering) inputs
        nmda_inputs = inputs[..., :self.in_features_clustering]  # --> B x L x in_F_c
        non_nmda_inputs = inputs[..., self.in_features_clustering:]  # --> B x L x in_F_nc

        # nmda synapses should be more rare than non-nmda synapses
        # constraints on the nmda and non-nmda weights
        # Clamping weights to the range [-1, 1]
        # self.non_nmda.weight.data = torch.clamp(self.non_nmda.weight.data, -1, 1)
        # nmda_weight = self.nmda.weight.data
        # nmda_weight = torch.clamp(nmda_weight, -1, 1)
        # sum_nmda = nmda_weight.sum(dim=1)
        # max_nmda = max(1, nmda_weight.size(1) / 100)  # at least 1 synapse per neuron otherwise 1% of synapses
        # # make sure that sum_nmda is smaller or equal to max_nmda
        # delta = F.relu(sum_nmda - max_nmda)  # if sum_ndma is smaller than max_nmda, delta is 0  --> out_F
        # delta = delta.unsqueeze(-1).repeat(1, self.in_features)  # --> out_F x in_F
        # assert delta.size() == self.nmda.weight.size(), f"delta has wrong shape (found {delta.size()} instead of required {self.nmda.weight.size()})"
        # nmda_weight = nmda_weight - (delta / nmda_weight.size(1))
        # self.nmda.weight.data = nmda_weight

        # The state is the classical weighted sum of inputs
        # As an approximation, only non NMDA contribute to the state
        # threshoold state to be maximum zero so that ic cannot contribut to positive output
        # nmda cluster activity needed to obtain positive outputs
        state = F.sigmoid(self.non_nmda(non_nmda_inputs)) - 1.0  # --> B x L x out_F
        assert all((state <= 0).view(-1)), f"state has positive values (found max {state.max()})"
        assert state.size() == (original_shape[:-1] + (self.out_features,)), f"state has wrong shape (found {state.size()} instead of required {(original_shape[:-1] + (self.out_features,))}; full dims are {state.size()})"

        # We now include a 'clustering' mechanism for NMDA synapses that introduces an aggressively
        # non-linear read out of the synaptic activities, gated by the state.
        # To identify clusters, we access the individual synaptic activities
        # via an element-wise multiplication to obtain a matrix of neurons x synapses
        # This will have dimension ... x out_F x in_F
        # It requires a bit of tensor gymnastics with broadcasting rules (thank you CoPilot):
        # inputs need to be reshaped to B x 1 x in_F and weights to 1 x out_F x in_F
        nmda_inputs = nmda_inputs.view(-1, 1, self.in_features_clustering)  # --> B*L x 1 x in_F_c
        nmda_syn = torch.mul(nmda_inputs,  self.nmda.weight) # element-wise multiplication --> B*L x out_F x in_F_c
        assert nmda_syn.size() == (nmda_inputs.size(0), self.out_features, self.in_features_clustering), f"synapses has wrong shape (found {synapses.size()} instead of required {(x.size(0), self.out_features, self.in_features)}; full dims are {synapses.size()})"

        # To scan adjacent synapses along the in_Feature axis
        # we use a fixed convolution filter to aggregate across neighboring synapses
        # conv1D expects B x C x W, where C is the number of channels (in our case only 1, for now)
        # and W is the dimension along which the convolution is applied.
        # In our case the convolution dimension W has to correspond to the in_F dimension
        # as we apply the convolution along the synaptic inputs for each neuron
        nmda_syn = nmda_syn.view(-1, 1, self.in_features_clustering)  # --> B*L*out_F x 1 x in_Fc
        # we apply the convolution filter (out_channels x in_channels x kernel_size)
        # in our canse out_channels=1 and in_channels=1 
        cluster = F.conv1d(nmda_syn, self.conv_filter, stride=self.stride)  # B*L*out_F x 1 x in_Fc-n
        cluster = cluster.view(-1, self.out_features, cluster.size(-1))  # --> B*L x out_F x in_Fc-n
        cluster_activity = cluster.sum(-1)  # --> B*L x out_F
        cluster_activity = cluster_activity.view(original_shape[:-1] + (self.out_features,))  # --> B x L x out_F
        # cluster_activity = F.relu(cluster_activity)  # threshold to only positive, for example with Hill function?
        # cooperativity and 'gating'?
        # state is B x L x out_F too
        cluster_activity = self.cluster_act_fn(cluster_activity + state)  # --> B x L x out_F

        output = cluster_activity # + state

        assert output.size(-1) == self.out_features, f"output has wrong number of out_features (dimension -1) (found {output.size(-1)} instead of required {self.out_features})"
        assert output.size(0) == inputs.size(0)
        return output  # --> B x L x out_F

    # see nn.Linear.reset_parameters()
    def reset_parameters(self) -> None:
        self.nmda.reset_parameters()
        self.non_nmda.reset_parameters()


if __name__ == "__main__":

    # profile and check it works

    # set deterministic behavior
    torch.manual_seed(0)

    # create a dendritic layer with 2 synaptic channels, 10 inputs and 3 outputs
    dendritic = DendriticFullyConnected(
        in_features=768,
        out_features=3,
        bias=True,
        clustering_frac=0.1,
        conv_filter=torch.tensor([[[1.0] * 5]]),
        stride=4,
        cluster_act_fn=Tanh(),
    )
    print(dendritic)

    # create a random input tensor of size 32 x 10
    inputs = torch.randn(50, 10, 768)
    print(inputs.size())

    # profile timing
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        output = dendritic(inputs)
    if torch.cuda.is_available():
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print(output.size())

    assert output.size() == (50, 10, 3), f"output has wrong shape (found {output.size()})"
    print("Success!")
