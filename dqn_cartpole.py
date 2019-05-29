import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
from keras.optimizers import Adam


memory_size = 100000
BATCH_SIZE = 20

class DQNSolver():
    def __init__(self):
        self.gamma = 0.9
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=env.observation_space.shape[0], activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(env.action_space.n, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.001))

    def experience_replay(self, batch):
        for state, action, next_state, reward, terminal in batch:
            q_update = reward if terminal else reward + self.gamma*np.amax(self.model.predict(next_state))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)


env = gym.make('CartPole-v1')
total_episode = 1000
epsilon = 0.99
memory = deque([], maxlen=memory_size)
modelNN = DQNSolver()


for episode in range(total_episode):
    state = np.reshape(env.reset(), [1, env.observation_space.shape[0]])
    terminal = False
    count = 0
    epsilon = max(epsilon*np.exp(-episode/50), 0.01)
    while terminal is not True:
        count += 1
        env.render()
        action = np.argmax(modelNN.model.predict((state))) if random.random() >= epsilon else env.action_space.sample()
        next_state, reward, terminal, _ = env.step(action)
        reward = reward if not terminal else -50*reward
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        memory.append((state, action, next_state, reward, terminal))
        state = next_state
        modelNN.experience_replay(random.sample(memory, BATCH_SIZE) if len(memory) > BATCH_SIZE else memory)
    print(count)

