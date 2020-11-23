import multiprocessing as mp
import argparse

import cpommerman
import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants

import numpy as np
from gym.spaces import Discrete
action_space = Discrete(6)

def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(b), where=b!=0)

def sample_argmax(a):
    return np.random.choice(np.arange(len(a))[a == a.max()])


class UCB:
    def __init__(self, actions, c=2):
        self.actions = actions
        self.value_sum = np.zeros(len(self.actions))
        self.visits = np.zeros(len(self.actions))
        self.c = c

    def select(self):
        Q = safe_div(self.value_sum, self.visits)
        U = self.c * np.sqrt(safe_div(np.log(self.visits.sum()), self.visits))
        U[self.visits==0] = np.inf
        ucb = Q + U
        self.idx = sample_argmax(ucb)
        return self.actions[self.idx]

    def update(self, value):
        self.value_sum[self.idx] += value
        self.visits[self.idx] += 1

    def act(self):
        idx = sample_argmax(self.visits)
        return self.actions[idx]


class TS:
    def __init__(self, actions):
        self.actions = actions
        self.win_count, self.loss_count = np.ones(len(self.actions)), np.ones(len(self.actions))

    def select(self):
        probs = [np.random.beta(wins, losses) for wins, losses in zip(self.win_count, self.loss_count)]
        self.idx = np.argmax(probs)
        return self.actions[self.idx]

    def update(self, value):
        if value == 0:
            self.win_count[self.idx] += 0.5
            self.loss_count[self.idx] += 0.5
        elif value > 0:
            self.win_count[self.idx] += 1
        elif value < 0:
            self.loss_count[self.idx] += 1

    def mean(self):
        return self.win_count / (self.win_count + self.loss_count)

    def act(self):
        self.idx = sample_argmax(self.mean())
        return self.actions[self.idx]


class DecoupledMAB:
    def __init__(self, legal_actions, mab):
        if mab == 'ts':
            self.policies = [TS(actions) for actions in legal_actions]
        elif mab == 'ucb':
            self.policies = [UCB(actions) for actions in legal_actions]
        else:
            raise Exception('Unknown MAB algorithm')

    def select(self):
        return tuple([policy.select() for policy in self.policies])

    def update(self, values):
        for policy, value in zip(self.policies, values):
            policy.update(value)

    def act(self):
        return np.array([policy.act() for policy in self.policies], dtype=np.uint8)


class MCS:
    def __init__(self, n_simulations, horizon, mab='ts', value_fn=None):
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.mab = mab
        self.value_fn = value_fn
        if self.value_fn is None:
            self.value_fn = lambda env: env.get_rewards()

    def reset(self):
        pass

    def step(self, env, legal_actions):
        root_state = env.get_json_info()
        policy = DecoupledMAB(legal_actions, self.mab)
        for _ in range(self.n_simulations):
            env.set_json_info(root_state)
            actions = policy.select()
            env.step(np.array(actions, dtype=np.uint8))
            for _ in range(self.horizon-1):
                actions = [np.random.choice(actions) for actions in env.get_legal_actions()]
                env.step(np.array(actions, dtype=np.uint8))
                if env.get_done():
                    break

            values = self.value_fn(env)
            policy.update(values)

        env.set_json_info(root_state)

        actions = policy.act()
        self.root_policy = policy
        return actions


class MCTS:
    def __init__(self, n_simulations, mab='ts', value_fn=None):
        self.n_simulations = n_simulations
        self.mab = mab
        self.value_fn = value_fn
        if self.value_fn is None:
            self.value_fn = lambda env: env.get_rewards()

    def reset(self):
        self.tree = {}

    def get_state_str(self, json_info):
        return str({k:v for k,v in json_info.items() if k not in "step_count"})

    def step(self, env, legal_actions):
        root_state = env.get_json_info()
        root_state_str = self.get_state_str(root_state)
        if root_state_str not in self.tree:
            self.tree[root_state_str] = DecoupledMAB(legal_actions, self.mab)
        for _ in range(self.n_simulations):
            env.set_json_info(root_state)
            state_str = root_state_str
            search_path = []
            while True:
                policy = self.tree[state_str]
                actions = policy.select()
                env.step(np.array(actions, dtype=np.uint8))
                search_path.append(policy)

                state_str = self.get_state_str(env.get_json_info())
                if state_str not in self.tree:
                    legal_actions = env.get_legal_actions()
                    self.tree[state_str] = DecoupledMAB(legal_actions, self.mab)
                    break

                if env.get_done():
                    break

            values = self.value_fn(env)

            for policy in search_path:
                policy.update(values)

        env.set_json_info(root_state)

        actions = self.tree[root_state_str].act()
        self.root_policy = self.tree[root_state_str]
        return actions


class FDTS:
    def __init__(self, n_simulations, horizon, mab='ts', value_fn=None):
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.mab = mab
        self.value_fn = value_fn
        if self.value_fn is None:
            self.value_fn = lambda env: env.get_rewards()

    def reset(self):
        self.tree = {}

    def get_state_str(self, json_info):
        return str({k:v for k,v in json_info.items() if k not in ["step_count", "board_size", "items"]})

    def step(self, env, legal_actions):
        root_state = env.get_json_info()
        root_state_str = self.get_state_str(root_state)
        if root_state_str not in self.tree:
            self.tree[root_state_str] = DecoupledMAB(legal_actions, self.mab)
        for _ in range(self.n_simulations):
            env.set_json_info(root_state)
            state_str = root_state_str
            search_path = []
            for _ in range(self.horizon):
                policy = self.tree[state_str]
                actions = policy.select()
                env.step(np.array(actions, dtype=np.uint8))
                search_path.append(policy)

                state_str = self.get_state_str(env.get_json_info())
                if state_str not in self.tree:
                    legal_actions = env.get_legal_actions()
                    self.tree[state_str] = DecoupledMAB(legal_actions, self.mab)

                if env.get_done():
                    break

            values = self.value_fn(env)

            for policy in search_path:
                policy.update(values)

        env.set_json_info(root_state)

        actions = self.tree[root_state_str].act()
        self.root_policy = self.tree[root_state_str]
        return actions


class JointSimpleAgent:
    def __init__(self, mcts_id=None):
        self.agents = [SimpleAgent(), SimpleAgent(), SimpleAgent(), SimpleAgent()]
        if mcts_id is not None:
            self.agents[mcts_id] = None
        self.mcts_id = mcts_id

    def step(self, obses):
        actions = []
        for i, obs in enumerate(obses):
            if i+10 in obs['alive'] and i != self.mcts_id:
                actions.append(self.agents[i].act(obs, action_space))
            else:
                actions.append(constants.Action.Stop.value)
        return np.array(actions, dtype=np.uint8)


def runner(id, num_episodes, args, fifo):
    env = cpommerman.make()

    if args.planner == 'fdts':
        planner = FDTS(args.n_simulations, args.horizon, args.mab)
    elif args.planner == 'mcts':
        planner = MCTS(args.n_simulations, args.mab)
    elif args.planner == 'mcs':
        planner = MCS(args.n_simulations, args.horizon, args.mab)
    else:
        raise Exception('Unknown planner')

    outcomes = [0, 0, 0] # wins, draws, losses
    for i in range(num_episodes):
        env.reset()
        planner.reset()
        planner_id = np.random.randint(4)
        simple = JointSimpleAgent(planner_id)

        while not env.get_done():
            obses = env.get_observations()
            actions = simple.step(obses)
            planner_actions = planner.step(env, env.get_legal_actions())
            actions[planner_id] = planner_actions[planner_id]

            state_str = planner.get_state_str(env.get_json_info())

            env.step(actions)

        rewards = env.get_rewards()
        if rewards[planner_id] == 1: # win
            idx = 0
        elif sum(rewards) == -4: # draw
            idx = 1
        else: # loss
            idx = 2
        outcomes[idx] += 1
        fifo.put(idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--planner', default='fdts')
    parser.add_argument('--mab', default='ts')
    parser.add_argument('--n_simulations', default=100, type=int)
    parser.add_argument('--horizon', default=20, type=int)

    parser.add_argument('--n_threads', default=5, type=int)
    parser.add_argument('--n_episodes', default=100, type=int)

    args = parser.parse_args()

    mp.set_start_method('spawn')
    fifo = mp.Queue()
    for i in range(args.n_threads):
        process = mp.Process(target=runner, args=(i, args.n_episodes//args.n_threads, args, fifo))
        process.start()

    outcomes = [0, 0, 0] # wins, draws, losses
    for i in range(args.n_episodes):
        idx = fifo.get()
        outcomes[idx] += 1
        print(f'Played {i} games. Wins: {outcomes[0]}. Draws: {outcomes[1]}. Losses: {outcomes[2]}.')
