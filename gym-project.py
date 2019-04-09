import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('Taxi-v2')
q_table = np.zeros([env.observation_space.n, env.action_space.n])
learning_rate = 0.9
discount = 0.9
epsilon = 0.1
num_episode = 1000
rewards = np.zeros([num_episode,])

for episode in range(0, num_episode):
	state = env.reset()
	done = False
	while done != True:
		if random.random() >= epsilon: curr_action = np.argmax(q_table[state, :])
		else: curr_action = env.action_space.sample()
		new_state, reward, done, info = env.step(curr_action)
		q_table[state, curr_action] += learning_rate*(reward + discount*max(q_table[new_state, :]) - q_table[state, curr_action])
		state = new_state
		rewards[episode,] += reward 

plt.plot(rewards)
plt.show()