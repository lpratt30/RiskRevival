"""Double DQN with bounded replay and a softly updated target network."""

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_layers=5,
        hidden_dim_max=512,
        hidden_dim_min=128,
    ):
        super().__init__()
        if num_layers < 2 or not 0 < hidden_dim_min <= hidden_dim_max:
            raise ValueError(
                "Use at least two layers and 0 < hidden_dim_min <= hidden_dim_max"
            )
        dim = hidden_dim_max
        layers = [nn.Linear(input_size, dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            next_dim = dim // 2 if dim // 2 >= hidden_dim_min else dim
            layers.extend([nn.Linear(dim, next_dim), nn.ReLU()])
            dim = next_dim
        layers.append(nn.Linear(dim, output_size))
        # Keep parameter names/shapes compatible with the original checkpoints.
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        batch_size=64,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        tau=0.9,
        num_layers=5,
        hidden_dim_max=512,
        hidden_dim_min=128,
        target_update_freq=1000,
    ):
        if batch_size < 1 or target_update_freq < 1:
            raise ValueError("Batch size and target update frequency must be positive")
        if not 0 <= gamma <= 1 or not 0 <= tau <= 1:
            raise ValueError("gamma and tau must be between zero and one")
        if not 0 <= epsilon_min <= epsilon <= 1:
            raise ValueError("Require 0 <= epsilon_min <= epsilon <= 1")
        if epsilon_decay is not None and not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.memory_limit = 100000
        self._memory_index = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.steps = 0
        architecture = (
            state_size,
            action_size,
            num_layers,
            hidden_dim_max,
            hidden_dim_min,
        )
        self.model = DQN(*architecture).to(self.device)
        self.target_model = DQN(*architecture).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.requires_grad_(False)
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def remember(self, state, action, reward, next_state, terminated, truncated):
        transition = (
            np.array(state, copy=True),
            action,
            reward,
            np.array(next_state, copy=True),
            terminated,
            truncated,
        )
        if len(self.memory) < self.memory_limit:
            self.memory.append(transition)
        else:
            self.memory[self._memory_index] = transition
        self._memory_index = (self._memory_index + 1) % self.memory_limit

    def act(self, state, eval=False):
        if not eval and np.random.rand() <= self.epsilon:
            return int(np.random.choice(self.action_size))
        with torch.no_grad():
            state = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return self.model(state).argmax(dim=1).item()

    def optimize_network(self):
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, terminated, _ = zip(
            *random.sample(self.memory, self.batch_size)
        )
        states = torch.as_tensor(
            np.array(states), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.array(next_states), dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            terminated, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            # Double DQN: online network selects; target network evaluates.
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_values = (
                self.target_model(next_states).gather(1, next_actions).squeeze(1)
            )
            # Time limits truncate a continuing task, so bootstrap through them.
            targets = rewards + (1 - terminated) * self.gamma * next_values
        values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            with torch.no_grad():
                for target, online in zip(
                    self.target_model.parameters(), self.model.parameters()
                ):
                    target.lerp_(online, self.tau)
        if self.epsilon_decay is not None:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    def load(self, filepath):
        state = torch.load(filepath, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.target_model.load_state_dict(state)

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
