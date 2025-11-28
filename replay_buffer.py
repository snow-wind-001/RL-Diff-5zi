import numpy as np


class DiffReplayBuffer:
    """扩散经验回放缓冲区"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.players = []
        self.advantages = []

    def add(self, state, action, player, advantage):
        """添加经验到缓冲区"""
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.players.pop(0)
            self.advantages.pop(0)
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.int64))
        self.players.append(int(player))
        self.advantages.append(float(advantage))

    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.states)

    def sample(self, batch_size):
        """从缓冲区中采样一个批次的数据"""
        idxs = np.random.randint(0, len(self.states), size=batch_size)
        batch_states = [self.states[i] for i in idxs]
        batch_actions = [self.actions[i] for i in idxs]
        batch_players = [self.players[i] for i in idxs]
        batch_adv = [self.advantages[i] for i in idxs]
        return (
            np.stack(batch_states, axis=0),
            np.stack(batch_actions, axis=0),
            np.array(batch_players, dtype=np.int64),
            np.array(batch_adv, dtype=np.float32),
        )