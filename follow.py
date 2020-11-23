import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cpommerman
from plan import FDTS, JointSimpleAgent


class Net(nn.Module):
    def __init__(self, n_channels=32, n_conv_layers=4):
        super().__init__()
        layers = [nn.Conv2d(14, n_channels, 3, padding=1), nn.ReLU()]
        for _ in range(n_conv_layers-1):
            layers.extend([nn.Conv2d(n_channels, n_channels, 3, padding=1), nn.ReLU()])
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(11*11*n_channels, 6)
        print(self.conv)
        print(self.fc)

    def forward(self, obs):
        out = self.conv(obs)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def act(self, obs, legal_mask, sample=False):
        pred = self.forward(obs)
        theta = F.softmax(pred, -1).detach().cpu().numpy()
        theta = theta * legal_mask
        theta = theta / theta.sum(-1)[:, None]
        if sample:
            return np.array([np.random.choice(6, p=p) if not np.isnan(p[0]) else 0 for p in theta], dtype=np.uint8)
        else:
            return theta.argmax(1).astype(np.uint8)

    def compute_loss(self, features, targets):
        pred = self.forward(features)
        targets = targets[:, :, 0] / (targets[:, :, 0] + targets[:, :, 1])
        loss = F.cross_entropy(pred, targets.argmax(-1))
        return loss


class Policy:
    def __init__(self):
        self.env = cpommerman.make()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.net = Net().to(self.device)
        self.net.eval()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)

        self.oracle = FDTS(100, 20)
        self.buffer = deque(maxlen=10**5)

    def add_data(self, features, policies, alive):
        for features_p, policy_p, alive_p in zip(features, policies, alive):
            self.latest_data = []
            if alive_p:
                targets = torch.ones(6, 2)
                targets[:, 1] *= 100
                targets[policy_p.actions, 0] = torch.FloatTensor(policy_p.win_count)
                targets[policy_p.actions, 1] = torch.FloatTensor(policy_p.loss_count)
                self.buffer.append((features_p, targets))
                self.latest_data.append((features_p, targets))

    def train(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return None
        
        self.net.train()

        batch = random.sample(self.buffer, batch_size-len(self.latest_data))
        batch = batch + self.latest_data
        features, targets = [torch.stack(t).to(self.device) for t in zip(*batch)]

        loss = self.net.compute_loss(features, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.net.eval()
        return loss.item()

    def get_legal_mask(self, legal_actions):
        legal_mask = np.zeros((4, 6))
        for i, actions in enumerate(legal_actions):
            legal_mask[i][actions] = 1.
        return legal_mask

    def rollout(self, train=True, render=False):
        self.env.reset()
        self.oracle.reset()
        losses, length = [], 0
        oracle_diff, oracle_last = deque(maxlen=10), np.zeros(4)
        bomb_last = 0
        while not self.env.get_done():
            # Oracle
            legal_actions = self.env.get_legal_actions()
            oracle_actions = self.oracle.step(self.env, legal_actions)

            # Follower
            features = self.env.get_features()
            features = torch.FloatTensor(features).permute(0, 3, 1, 2).to(self.device)
            legal_mask = self.get_legal_mask(legal_actions)
            actions = self.net.act(features, legal_mask, sample=True)
            rewards = self.env.get_rewards()
            alive = rewards == 0
            actions[~alive] = 0

            # Step
            self.env.step(actions)

            # Train
            self.add_data(features, self.oracle.root_policy.policies, alive)
            loss = self.train() if train else None
            if loss is not None:
                losses.append(loss)
            length += 1

            # Early termination
            oracle_diff.append(oracle_actions-oracle_last)
            oracle_last = oracle_actions
            if len(oracle_diff) == 10 and np.array(oracle_diff).sum() == 0:
                print("Terminating episode: same oracle action in past 10 steps")
                break
            bomb_last = 0 if np.any(actions==5) else bomb_last + 1
            if bomb_last > 40:
                print("Terminating episode: no bomb action in past 40 steps")
                break

            #print(length, rewards, actions, oracle_actions, loss, len(self.buffer))

        loss = np.mean(losses) if losses else None
        rewards = self.env.get_rewards()
        print(f"Rewards: {rewards}, Loss: {loss}")
        return loss, length

    def evaluate(self, n_games=100):
        outcomes = [0, 0, 0] # wins, draws, losses
        for _ in range(n_games):
            self.env.reset()
            policy_id = np.random.randint(4)
            simple = JointSimpleAgent(policy_id)
            while not self.env.get_done():
                obses = self.env.get_observations()
                actions = simple.step(obses)

                features = self.env.get_features()
                features = torch.FloatTensor(features).permute(0, 3, 1, 2).to(self.device)
                legal_mask = self.get_legal_mask(self.env.get_legal_actions())
                policy_actions = self.net.act(features, legal_mask)
                actions[policy_id] = policy_actions[policy_id]

                self.env.step(actions)

            rewards = self.env.get_rewards()
            if rewards[policy_id] == 1: # win
                idx = 0
            elif sum(rewards) == -4: # draw
                idx = 1
            else: # loss
                idx = 2
            outcomes[idx] += 1
        return outcomes

def run():
    policy = Policy()
    for i in range(1000):
        loss, length = policy.rollout(render=False)
        if (i+1) % 10 == 0:
            outcomes = policy.evaluate()
            print(i, outcomes)
            print("===")

if __name__ == '__main__':
    run()
