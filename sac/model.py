import mxnet as mx
from mxnet.gluon import Block, nn
from mxnet import nd
from gluonts.distribution import Gaussian

LOG_MAX = 2
LOG_MIN = -20
EPSILON = 1e-6


class QNetwork(Block):
    def __init__(self, num_inputs, num_actions, hidden_dim, device):
        super(QNetwork, self).__init__()
        self.device = device
        mx.Context.default_ctx = mx.Context(device, 0)

        with self.name_scope():
            self.layer0 = nn.Dense(units=hidden_dim, in_units=num_inputs + num_actions, activation="relu")
            self.layer1 = nn.Dense(units=hidden_dim, in_units=hidden_dim, activation="relu")
            self.layer2 = nn.Dense(units=1, in_units=hidden_dim, activation="relu")

            self.initialize(mx.init.Xavier())
            self.cast("float64")

    def forward(self, state, action):

        x = nd.concat(state, action, dim=1)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class GaussianPolicy(Block):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space, device):

        super(GaussianPolicy, self).__init__()
        mx.Context.default_ctx = mx.Context(device, 0)

        with self.name_scope():

            self.layer0 = nn.Dense(hidden_dim, in_units=num_inputs, activation="relu")
            self.layer1 = nn.Dense(hidden_dim, in_units=hidden_dim, activation="relu")

            self.output_mean = nn.Dense(num_actions, in_units=hidden_dim)
            self.output_log_std = nn.Dense(num_actions, in_units=hidden_dim)

            self.initialize(mx.init.Xavier())

        # action rescaling
        if action_space is None:
            self.action_scale = nd.array(1)
            self.action_bias = nd.array(0)
        else:
            self.action_scale = nd.array((action_space.high - action_space.low) / 2.0, dtype="float64")
            self.action_bias = nd.array((action_space.high + action_space.low) / 2.0, dtype="float64")

        self.cast("float64")

    def forward(self, state):

        state = nd.array(state, dtype="float64")
        x = self.layer0(state)
        x = self.layer1(x)
    
        mean = self.output_mean(x)
        log_std = self.output_log_std(x)
        log_std = nd.clip(log_std, a_min=LOG_MIN, a_max=LOG_MAX)

        return mean, log_std

    def sample(self, mean, log_std):
        std = log_std.exp()

        distribution = Gaussian(mu=mean, sigma=std)
        sample = distribution.sample(dtype="float64")  # for reparameterization trick (mu + std * N(0,1))    
        sample_log_prob = distribution.log_prob(sample)

        return  self.scale_and_bound(sample, sample_log_prob, mean)
    
    def scale_and_bound(self, sample, log_prob, mean):
        action_bounded = sample.tanh() # bound action
        action_scaled = action_bounded * self.action_scale + self.action_bias # scale action

        mean_bounded = mean.tanh() * self.action_scale + self.action_bias # bound and scale mean

        log_prob_bounded = log_prob - (self.action_scale * (1 - nd.power(action_bounded, 2)) + EPSILON).log()

        return action_scaled, log_prob_bounded, mean_bounded






