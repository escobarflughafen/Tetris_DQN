import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
from tetris import Tetris
from collections import deque
from deep_q_network import DeepQNetwork1 as DeepQNetwork
import matplotlib.pyplot as plt
import matplotlib
import argparse
import datetime
import dateutil


class Agent:
    def __init__(self, opt):
        self.memory = deque(maxlen=opt.memory_size)
        self.memory_size = opt.memory_size
        self.batch_size = opt.batch_size
        self.policy_net = DeepQNetwork()
        self.target_net = DeepQNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_net_update_period = opt.target_net_update_period
        self.game = Tetris(block_size=20)
        self.epsilon = opt.initial_epsilon
        self.final_epsilon = opt.final_epsilon
        self.epsilon_decay_rate = opt.epsilon_decay_rate
        self.discount_factor = opt.gamma
        self.log_path = opt.log_path
        self.model_path = opt.model_path
        self.results = {
            'scores': [],
            'cleared_lines': [],
            'dropped_tetrominoes': []
        }

        self.state = self.game.reset()
        self.action = None
        self.next_state = None
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=opt.lr)

        self.epoch = 1

        if (not os.path.isdir(self.log_path)):
            os.mkdir(os.path.join(os.getcwd(), self.log_path))

        if (not os.path.isdir(self.model_path)):
            os.mkdir(os.path.join(os.getcwd(), self.model_path))

    def reset(self):
        pass

    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))

    def policy_q_value(self, state):
        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(state)[:, 0]
        self.policy_net.train()

        return q

    def target_q_value(self, state):
        self.target_net.eval()
        with torch.no_grad():
            q = self.target_net(state)[:, 0]
        self.target_net.train()

        return q

    def get_actions(self):
        actions, next_states = list(zip(*self.game.get_next_states().items()))

        return actions, next_states

    def _epsilon_greedy_play(self):
        state = self.state
        actions, next_states = self.get_actions()
        state_action_pairs = torch.stack(
            [torch.cat((state, torch.tensor(action))) for action in actions])

        u = random.random()

        # updating epsilon value
        if (self.epsilon > self.final_epsilon):
            self.epsilon = max(
                self.epsilon - self.epsilon_decay_rate, self.final_epsilon)

        if u < self.epsilon:
            index = random.randint(0, len(actions) - 1)

        else:
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state_action_pairs)[:, 0]
            self.policy_net.train()

            index = torch.argmax(action_values).item()

        action = actions[index]
        next_state = next_states[index][:]

        reward, is_game_over = self.game.step(action, render=False)

        next_actions = self.get_actions()[0]

        self.memory.append([state_action_pairs[index], reward,
                           next_state, next_actions, is_game_over])

        if is_game_over:
            self.results['scores'].append(self.game.score)
            self.results['dropped_tetrominoes'].append(self.game.tetrominoes)
            self.results['cleared_lines'].append(self.game.cleared_lines)
            self.state = self.game.reset()
        else:
            self.state = next_state

        return reward, is_game_over

    def epsilon_greedy_play(self, render=False):
        actions, next_states = self.get_actions()

        u = random.random()

        # updating epsilon value
        if (self.epsilon > self.final_epsilon and self.epoch > 1):
            self.epsilon = max(
                self.epsilon - self.epsilon_decay_rate, self.final_epsilon)

        if u <= self.epsilon:
            index = random.randint(0, len(actions) - 1)

        else:
            self.policy_net.eval()
            with torch.no_grad():
                predictions = self.policy_net(torch.stack(next_states))[:, 0]
            self.policy_net.train()

            index = torch.argmax(predictions).item()

        action = actions[index]
        next_state = next_states[index][:]

        reward, is_game_over = self.game.step(action, render)

        self.memory.append(
            [self.state, action, reward, next_state, is_game_over])

        if is_game_over:
            if render:
                self.game.render(save_img=True)
            self.results['scores'].append(self.game.score)
            self.results['dropped_tetrominoes'].append(self.game.tetrominoes)
            self.results['cleared_lines'].append(self.game.cleared_lines)
            self.state = self.game.reset()
        else:
            self.state = next_state

        return reward, is_game_over

    def train(self):
        if len(self.memory) >= self.batch_size * 4:
            if self.epoch % self.target_net_update_period == 0:
                print('update_target_network')
                self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epoch += 1
            batch = random.sample(self.memory, self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, is_game_over_batch = zip(
                *batch)

            q_values = self.policy_net(torch.stack(state_batch))

            self.target_net.eval()
            with torch.no_grad():
                next_q_value_prediction_batch = self.target_net(
                    torch.stack(next_state_batch))

            y_tuple = tuple(torch.tensor([reward]) if is_game_over else reward + self.discount_factor * prediction
                            for reward, is_game_over, prediction in
                            zip(reward_batch, is_game_over_batch, next_q_value_prediction_batch))

            y_batch = torch.stack(y_tuple)

            self.optimizer.zero_grad()
            loss = self.loss(q_values, y_batch)
            loss.backward()
            self.optimizer.step()

            print("epoch: {}, loss: {}".format(
                self.epoch,
                loss
            ))

    def _train(self):
        if len(self.memory) >= self.batch_size * 2:
            self.epoch += 1
            batch = list(
                filter(lambda x: x[-1] == False, random.sample(self.memory, self.batch_size)))
            print('actual batch size:', len(batch))
            state_action_pair_batch, reward_batch, next_state_batch, next_actions_batch, is_game_over_batch = zip(
                *batch)

            q_values = self.policy_net(torch.stack(state_action_pair_batch))
            self.policy_net.eval()
            with torch.no_grad():
                next_state_action_pair_batch = []
                for i in range(len(next_state_batch)):
                    next_state_action_pairs = torch.stack(
                        [torch.cat((next_state_batch[i], torch.tensor(action))) for action in next_actions_batch[i]])

                    pred = self.policy_net(next_state_action_pairs)
                    max_index = torch.argmax(pred).item()

                    next_state_action_pair_batch.append(
                        next_state_action_pairs[max_index])

                next_q_value_prediction_batch = self.policy_net(
                    torch.stack(next_state_action_pair_batch))
            self.policy_net.train()

            y_tuple = tuple(torch.tensor([reward]) if is_game_over else reward + self.discount_factor * prediction
                            for reward, is_game_over, prediction in
                            zip(reward_batch, is_game_over_batch, next_q_value_prediction_batch))

            y_batch = torch.stack(y_tuple)

            self.optimizer.zero_grad()
            loss = self.loss(q_values, y_batch)
            loss.backward()
            self.optimizer.step()

            print("epoch: {}, loss: {}".format(
                self.epoch,
                loss
            ))

    def plot_results(self, filename=None, dpi=300, backend='pdf'):
        matplotlib.use(backend)
        fig, axs = plt.subplots(3)
        for i, key in enumerate(self.results.keys()):
            axs[i].plot(*zip(*list(enumerate(self.results[key]))),
                        label=key, linewidth=0.5)
            axs[i].legend()

        plt.savefig(os.path.join(self.log_path, filename or 'tetris_{}epoch_{}.{}'.format(
            self.epoch, time.strftime('%Y-%m-%d_%H_%M_%S'), backend)), dpi=dpi)
        plt.close('all')

    def save_model(self, filename=None):
        torch.save(self.policy_net, os.path.join(self.model_path, filename or 'policy_net_{}_{}'.format(
            self.epoch, time.strftime('%Y-%m-%d_%H_%M_%S'))))


def load_and_play(model_path, args, round=-1):
    agent = Agent(args)
    agent.policy_net = torch.load(model_path)
    agent.epsilon = agent.final_epsilon

    def play():
        while True:
            done = agent.epsilon_greedy_play(render=True)[-1]
            if done:
                print('results:\t',
                      agent.results['scores'][-1],
                      agent.results['dropped_tetrominoes'][-1],
                      agent.results['cleared_lines'][-1])
                break
            else:
                pass

    if round < 0:
        while True:
            play()
    else:
        for i in range(round):
            play()
