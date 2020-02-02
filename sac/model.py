import mxnet as mx
from mxnet.gluon import Block, nn
from mxnet import nd
from gluonts.distribution import Gaussian

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class ValueNetwork(Block):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()
        # use name_scope to give child Blocks appropriate names.
        with self.name_scope():
            self.linear1 = mx.gluon.nn.Dense(hidden_dim, in_units=num_inputs)
            self.linear2 = mx.gluon.nn.Dense(hidden_dim, in_units=hidden_dim)
            self.linear3 = mx.gluon.nn.Dense(1, in_units=hidden_dim)

        self.initialize(mx.init.Xavier())

    def forward(self, state):
        x = nd.relu(self.linear1(state))
        x = nd.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(Block):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = mx.gluon.nn.Dense(
            hidden_dim, in_units=num_inputs + num_actions, dtype="float64"
        )
        self.linear2 = mx.gluon.nn.Dense(
            hidden_dim, in_units=hidden_dim, dtype="float64"
        )
        self.linear3 = mx.gluon.nn.Dense(1, in_units=hidden_dim, dtype="float64")

        self.initialize(mx.init.Xavier())

    def forward(self, state, action):

        xu = nd.concat(state, action, dim=1)
        x1 = nd.relu(self.linear1(xu))
        x1 = nd.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1


class GaussianPolicy(Block):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = mx.gluon.nn.Dense(hidden_dim, in_units=num_inputs)
        self.linear2 = mx.gluon.nn.Dense(hidden_dim, in_units=hidden_dim)

        self.mean_linear = mx.gluon.nn.Dense(num_actions, in_units=hidden_dim)
        self.log_std_linear = mx.gluon.nn.Dense(num_actions, in_units=hidden_dim)
        self.initialize(mx.init.Xavier())

        # action rescaling
        if action_space is None:
            self.action_scale = nd.array(
                1, dtype="float64"
            )  # have to check if this is right
            self.action_bias = nd.array(0, dtype="float64")
        else:
            self.action_scale = nd.array(
                (action_space.high - action_space.low) / 2.0, dtype="float64"
            )
            self.action_bias = nd.array(
                (action_space.high + action_space.low) / 2.0, dtype="float64"
            )

        self.cast("float64")

    def forward(self, state):
        state = nd.array(state, dtype="float64")
        x_1 = nd.relu(self.linear1(state))
        x_2 = nd.relu(self.linear2(x_1))
        mean = self.mean_linear(x_2)
        log_std = self.log_std_linear(x_2)
        log_std = nd.clip(log_std, a_min=LOG_SIG_MIN, a_max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()

        distribution = Gaussian(mu=mu, sigma=std)
        x_t = distribution.sample(
            dtype="float64"
        )  # for reparameterization trick (mu + std * N(0,1))
        y_t = x_t.tanh()
        action = y_t * self.action_scale + self.action_bias

        # Enforcing Action Bound
        log_prob_1 = (
            distribution.log_prob(x_t)
            - (self.action_scale * (1 - nd.power(y_t, 2)) + epsilon).log()
        )
        log_prob_2 = log_prob_1.sum(1, keepdims=True)

        mean = mu.tanh() * self.action_scale + self.action_bias
        return action, log_prob_2, mean
