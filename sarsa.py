from logic2048 import Game2048
import matplotlib.pyplot as plt
import numpy as np
import random

class SARSA:
    def __init__(self, alpha=0.0001, gamma=0.9, initial_epsilon=1.0, epsilon_decay=0.9, min_epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q_table = {}

    def state_to_key(self, state):
        return str(state.tolist())

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2, 3])
        else:
            key = self.state_to_key(state)
            if key not in self.Q_table:
                return random.choice([0, 1, 2, 3])
            return np.argmax(self.Q_table[key])

    def update_Q_table(self, state, action, reward, next_state, next_action):
        key = self.state_to_key(state)
        next_key = self.state_to_key(next_state)

        if key not in self.Q_table:
            self.Q_table[key] = np.zeros(4)

        if next_key not in self.Q_table:
            self.Q_table[next_key] = np.zeros(4)

        self.Q_table[key][action] += self.alpha * (reward + self.gamma * self.Q_table[next_key][next_action] - self.Q_table[key][action])

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)


# Main loop for training the agent
def train_sarsa(agent, episodes):
    highest_tile_training = []
    for episode in range(episodes):
        game = Game2048()
        state = game.get_preprocessed_state()
        action = agent.choose_action(state)
        total_reward = 0

        while not game.game_end:
            # Perform action and get reward
            prev_score = game.get_merge_score()
            game.make_move(action)
            reward = game.get_merge_score() - prev_score

            # Observe new state and choose next action
            next_state = game.get_preprocessed_state()
            next_action = agent.choose_action(next_state)

            # Update Q-table
            agent.update_Q_table(state, action, reward, next_state, next_action)

            # Update state and action
            state = next_state
            action = next_action
            total_reward += reward

        # Epsilon decay for balance between exploration and exploitation
        agent.update_epsilon()
        highest_tile_training.append(game.max_num())

    return highest_tile_training

def sarsa_test(agent, episodes):
    highest_tiles_testing = []
    for episode in range(episodes):
        game = Game2048()
        state = game.get_preprocessed_state()

        while not game.game_end:
            action = agent.choose_action(state)  # Choose action based on learned policy
            game.make_move(action)
            next_state = game.get_preprocessed_state()
            state = next_state

        highest_tiles_testing.append(game.max_num())

    return highest_tiles_testing

if __name__ == "__main__":
    agent = SARSA()
    training_rewards = train_sarsa(agent, 2000)
    testing_rewards = sarsa_test(agent, 750)
    plt.figure(figsize=(10, 12))

    # Training Score
    plt.subplot(2, 1, 1)
    plt.plot(training_rewards)
    plt.title('Training: Highest Tile per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Highest Tile')

    # Testing Score
    plt.subplot(2, 1, 2)
    plt.plot(testing_rewards)
    plt.title('Testing: Highest Tile per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Highest Tile')

    plt.tight_layout()
    plt.show()