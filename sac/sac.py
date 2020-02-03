import os
import pathlib
import pickle
import mxnet as mx

from mxnet import nd, gluon
from .model import GaussianPolicy, QNetwork
from .utils import MemoryBuffer, hard_update, soft_update


class SAC:
    """
    Soft-Actor Critic model
    Implementation based on: Haarnoja, et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." (2018)

    Args:
        num_inputs: int
            the number of inputs the neural networks receive
        action_space: gym space
            environment's action space
        gamma: float (default: .99)
            discount factor for reward
        tau: float (default: 5e-3)
            polyak smoothing coefficient (τ)
        lr: float (default: 3e-4)
            learning rate
        alpha: float (default: .2)
            Relative importance of the entropy term against the reward
        automatic_entropy_tuning: bool (default: False)
            Automatically adjust α
        batch_size: int (default: 256)
            batch size
        hidden_size: int (default: 64)
            size of hidden layers of neural network models
        target_update_interval: int (default: 1)
            Value target update per no. of updates per step
        gpu: bool (default: False)
            run on GPU
    """

    def __init__(
        self,
        num_inputs,
        action_space,
        gamma=0.99,
        tau=5e-3,
        lr=3e-4,
        alpha=0.2,
        automatic_entropy_tuning=False,
        batch_size=256,
        hidden_size=64,
        target_update_interval=1,
        gpu=False,
        **kwargs
    ):

        # discount factor
        self.gamma = gamma

        # entropy configuration
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # factor for updating the target network
        self.tau = tau
        # interval to update the critic target network wrt the critic
        self.target_update_interval = target_update_interval

        # parameter for casting variables
        self.device = mx.gpu() if gpu else mx.cpu()
        mx.Context.default_ctx = mx.Context(self.device, 0)

        # ==========================================================================================
        # We use a particular variation of SAC that uses Q-networks instead of a value network
        # ==========================================================================================

        self.critic = [
            QNetwork(num_inputs, action_space.shape[0], hidden_size, self.device),
            QNetwork(num_inputs, action_space.shape[0], hidden_size, self.device),
        ]

        self.critic_optim = [
            gluon.Trainer(
                self.critic[0].collect_params(), "adam", {"learning_rate": lr}
            ),
            gluon.Trainer(
                self.critic[1].collect_params(), "adam", {"learning_rate": lr}
            ),
        ]

        self.critic_target = [
            QNetwork(num_inputs, action_space.shape[0], hidden_size, self.device),
            QNetwork(num_inputs, action_space.shape[0], hidden_size, self.device),
        ]

        # the critic target doesn't need a optimizer, since it uses the update mechanism
        hard_update(self.critic_target[0], self.critic[0])
        hard_update(self.critic_target[1], self.critic[1])

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.target_entropy = -nd.prod(nd.array(action_space.shape)).asscalar()
            self.log_alpha = mx.gluon.Parameter(
                "log_alpha", shape=(1,), init=mx.init.Zero(), dtype="float64"
            )
            self.log_alpha.initialize(ctx=self.device)
            self.alpha_optim = gluon.Trainer(
                [self.log_alpha],
                optimizer="adam",
                optimizer_params={"learning_rate": lr},
            )

        self.policy = GaussianPolicy(
            num_inputs, action_space.shape[0], hidden_size, action_space, self.device
        )

        self.policy_optim = gluon.Trainer(
            self.policy.collect_params(),
            optimizer="adam",
            optimizer_params={"learning_rate": lr},
        )

    def select_action(self, state, eval=False):
        state = nd.array(state, ctx=self.device).expand_dims(0)
        mean, log_std = self.policy(state)
        if eval == False:
            # if not evaluating take the an action sampled from the distribution (stochastic policy)
            action, _, _ = self.policy.sample(mean, log_std)
        else:
            # otherwise take the mean of the distribution to properly evaluate policy
            _, _, action = self.policy.sample(mean, log_std)
        return nd.array(action.detach(), ctx=mx.cpu()).asnumpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory buffer
        (states, actions, rewards, next_states, masks,) = memory.sample(
            batch_size=batch_size
        )

        # cast variables
        states = nd.array(states, dtype="float64", ctx=self.device)
        next_states = nd.array(next_states, dtype="float64", ctx=self.device)
        actions = nd.array(actions, dtype="float64", ctx=self.device)
        rewards = nd.array(rewards, dtype="float64", ctx=self.device).expand_dims(1)
        masks = nd.array(masks, dtype="float64", ctx=self.device).expand_dims(1)

        # ===============================
        #    Minimizing the q-networks
        # ===============================

        # get next states based on current policy
        mean, log_std = self.policy(next_states)
        next_state_action, next_state_log_pi, _ = self.policy.sample(mean, log_std)

        # get q-values from dual-critic
        qf1_next_target = self.critic_target[0](
            next_states, nd.array(next_state_action, ctx=self.device)
        )
        qf2_next_target = self.critic_target[1](
            next_states, nd.array(next_state_action, ctx=self.device)
        )

        # select the minimum between the two q-values
        min_qf_next_target = self.min_between(qf1_next_target, qf2_next_target)

        min_qf_next_target -= self.alpha * next_state_log_pi
        next_q_value = rewards + masks * self.gamma * (min_qf_next_target)

        loss_fn = gluon.loss.L2Loss()

        with mx.autograd.record():

            qf1 = self.critic[0](states, actions) # Q_1(st,at)
            qf1_loss = loss_fn(qf1, next_q_value).mean() # JQ = E_(st,at)~D [ 0.5 ( Q1_(st,at) - r_(st,at) - γ ( E_(st+1)~p [V(st+1)] )) ^2]
            qf2 = self.critic[1](states, actions) # Q_2(st,at)
            qf2_loss = loss_fn(qf2, next_q_value).mean() # JQ = E_(st,at)~D [ 0.5 ( Q1_(st,at) - r_(st,at) - γ ( E_(st+1)~p [V(st+1)] )) ^2]

        mx.autograd.backward([qf1_loss, qf2_loss])
        self.critic_optim[0].step(1)
        self.critic_optim[1].step(1)

        # ===============================
        #    Minimizing the policy
        # ===============================

        with mx.autograd.record():
            mean, log_std = self.policy(states)
            pi, log_pi, _ = self.policy.sample(mean, log_std)
            qf1_pi = self.critic[0](states, pi)
            qf2_pi = self.critic[1](states, pi)
            min_qf_pi = self.min_between(qf1_pi, qf2_pi)
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        policy_loss.backward()
        self.policy_optim.step(1)

        if self.automatic_entropy_tuning:
            with mx.autograd.record():
                alpha_loss = -(
                    self.log_alpha.data() * (log_pi + self.target_entropy).detach()
                ).mean()

            alpha_loss.backward()
            self.alpha_optim.step(1)

            self.alpha = self.log_alpha.data().exp()
            alpha_tlogs = self.alpha.copy()  # For TensorboardX logs

        else:
            alpha_loss = nd.array(0.0, ctx=self.device)
            alpha_tlogs = nd.array(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target[0], self.critic[0], self.tau)
            soft_update(self.critic_target[1], self.critic[1], self.tau)

        return (
            qf1_loss.asscalar(),
            qf2_loss.asscalar(),
            policy_loss.asscalar(),
            alpha_loss,
            alpha_tlogs,
        )

    @staticmethod
    def min_between(arr1, arr2):
        return nd.cast(
            nd.min(nd.array([arr1.asnumpy(), arr2.asnumpy()]), axis=0), dtype="float64"
        )

    # Save model parameters
    def save_model(self, filename):
        name = "{}.pkl".format(filename.replace(".pkl", ""))
        pathlib.Path(name.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
        with open(name, "wb") as picklefile:
            pickle.dump(self, picklefile)
        print(
            "File saved at : {}".format(
                os.path.join(os.getcwd(), "{}.pkl".format(filename))
            )
        )

    # Load model parameters
    @classmethod
    def load_model(cls, filename):
        name = "{}.pkl".format(filename.replace(".pkl", ""))
        with open(name, "r+b") as picklefile:
            data = pickle.load(picklefile)
        return data

    def set_context(self, device):
        self.device = device
        print(device)
        mx.Context.default_ctx = mx.Context(self.device, 0)
