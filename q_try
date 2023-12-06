# Hyperparameters
EPSILON = 0.9
TARGET_REPLACE_ITER = 100
LEARNING_RATE = 1e-2
STATE_SIZE = 16
BATCH_SIZE = 32
BATCH_SHAPE = (BATCH_SIZE, 1, 4, 4)
MEMORY_CAPACITY = 10000
SHAPE = (1, 4, 4)
EPOCHS = 300
GAMMA = 0.99

import torch
from torch.autograd import Variable
import logic2048 as logic
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class QNetwork2048(object):
    """
    Q-learning agent for the 2048 game.
    """
    def __init__(self, action_set):
        self.eval_net, self.target_net = NeuralNetwork(), NeuralNetwork()
        self.loss_function = nn.MSELoss()
        self.actions = action_set
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        self.memory = np.zeros((MEMORY_CAPACITY, STATE_SIZE * 2 + 2))
        self.action_size = len(action_set)
        self.learn_step_counter = 0
        self.memory_counter = 0

    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy policy.
        """
        state = Variable(torch.unsqueeze(state, 0))
        if np.random.uniform() < EPSILON:
            actions_values = self.eval_net.forward(state)
            action_index = actions_values.data.numpy().argmax()
        else:
            action_index = np.random.randint(0, self.action_size)
        return action_index, self.actions[action_index]

    def store_transition(self, state, action, reward, next_state):
        """
        Store the transition in the replay memory.
        """
        transition = np.hstack((state.reshape(STATE_SIZE), [action, reward], next_state.reshape(STATE_SIZE)))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        """
        Update the Q-network based on a batch of experiences.
        """
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_states = batch_memory[:, :STATE_SIZE].reshape(BATCH_SHAPE)
        batch_states = Variable(torch.FloatTensor(batch_states))
        batch_actions = batch_memory[:, STATE_SIZE:STATE_SIZE + 1].astype(int)
        batch_actions = Variable(torch.LongTensor(batch_actions))
        batch_rewards = batch_memory[:, STATE_SIZE + 1:STATE_SIZE + 2]
        batch_rewards = Variable(torch.FloatTensor(batch_rewards))
        batch_next_states = batch_memory[:, -STATE_SIZE:].reshape(BATCH_SHAPE)
        batch_next_states = Variable(torch.FloatTensor(batch_next_states))

        q_eval = self.eval_net(batch_states).gather(1, batch_actions)
        q_next = self.target_net(batch_next_states).detach()
        q_target = batch_rewards + GAMMA * torch.unsqueeze(q_next.max(1)[0], 1)
        loss = self.loss_function(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class NeuralNetwork(nn.Module):
    """
    Neural Network class for Q-learning in 2048 game.
    """
    def __init__(self, input_size=(1, 4, 4), output_size=4, alpha=0.1):
        super(NeuralNetwork, self).__init__()
        self.alpha = alpha
        self.output_size = output_size
        self.input_size = input_size
        h = (input_size[1] - 1) * 2 + 2
        w = (input_size[2] - 1) * 2 + 2
        self.fc3 = nn.Linear(50, output_size)
        self.fc2 = nn.Linear(400, 50)
        self.fc1 = nn.Linear(3 * h * w, 400)
        self.conv_transpose = nn.ConvTranspose2d(self.input_size[0], 3, kernel_size=2, stride=2)
        
        # Initialize weights and biases
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0, 0.1)
                nn.init.constant_(module.bias.data, 0)

    def forward(self, input_vector):
        output0 = self.conv_transpose(input_vector).view(-1, 24 * 8)
        output1 = F.leaky_relu(self.fc1(output0), negative_slope=self.alpha)
        output2 = F.leaky_relu(self.fc2(output1), negative_slope=self.alpha)
        output3 = F.leaky_relu(self.fc3(output2), negative_slope=self.alpha)
        return output3

def train():
    """
    Train the Q-learning agent to play the 2048 game.
    """
    q_network = QNetwork2048([0, 1, 2, 3])
    max_block_values = []

    for e in range(EPOCHS):
        game = logic.Game2048()
        prev_score = 0

        while True:
            state = torch.FloatTensor(game.get_preprocessed_state())
            action_index, action = q_network.choose_action(state)
            game.make_move(action)
            current_score = game.get()
            game_over = game.game_end
            reward = current_score - prev_score
            prev_score = current_score

            q_network.store_transition(state.numpy(), action_index, reward, game.get_preprocessed_state())
            
            if q_network.memory_counter > MEMORY_CAPACITY:
                q_network.learn()

            if game_over:
                max_block_values.append(current_score)
                break

        print(f"Iteration {e}, max_block = {max_block_values[-1]}")
        print(game)

    plt.plot(max_block_values)
    plt.show()

if __name__ == "__main__":
    train()
