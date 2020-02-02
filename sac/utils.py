import collections
import random
import numpy as np

def soft_update(target, source, tau):
    for target_param, param in zip(
        target.collect_params().items(), source.collect_params().items()
    ):
        target_param[1].set_data(
            target_param[1].data() * (1.0 - tau) + param[1].data() * tau
        )


def hard_update(target, source):
    for target_param, param in zip(
        target.collect_params().items(), source.collect_params().items()
    ):
        target_param[1].set_data(param[1].data())


class MemoryBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = collections.deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) == self.capacity:
            if not mask:
                self.buffer.appendleft((state, action, reward, next_state, mask))
            else:
                self.buffer[-1] = (state, action, reward, next_state, mask)
        else:
            self.buffer.append((state, action, reward, next_state, mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask = map(np.stack, zip(*batch))
        return state, action, reward, next_state, mask

    def __len__(self):
        return len(self.buffer)
