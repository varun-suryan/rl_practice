import random
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel 


epsilon = 0.02
alpha = 0.2
gamma = 0.9
GRID = 5
num_of_episodes = 30
q = {}

states = [(x,y) for x in range(-GRID, GRID + 1) for y in range(-GRID, GRID + 1)]
actions=[(1,0), (-1,0), (0,1), (0,-1)]



term_state = (GRID, GRID)

# train_data_f = np.arange(-10 , 10, 1)[:, np.newaxis]

# train_data_t = np.sin(train_data_f)

# mean = np.zeros(train_data_f.shape)

kernel = C(5.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-5, 1e5))

# sampled = np.random.multivariate_normal( np.squeeze(mean), kernel(train_data_f))


gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 2)


# plt.plot(data, np.sin(data))
# plt.show()


# print gp.get_params(deep=True)

# x = np.arange(-10 , 10, 0.2)[:, np.newaxis]


# y_pred, sigma = gp.predict(x, return_std=True)

# plt.scatter(train_data_f, train_data_t)
# plt.plot(x, y_pred)
# plt.show()

def reward_dynamics(state):
    if state == term_state:
        return 100
    else:
        return -10

def getQ(state, action):
    return q.get((state, action), 0)
    # return self.q.get((state, action), 1.0)

# def learnQ(state, action, reward, value):
#     oldv = q.get((state, action))
#     if oldv is None:
#         q[(state, action)] = oldv + alpha * (value - oldv)
#     else:
#         q[(state, action)] = oldv + alpha * (value - oldv)

def chooseAction(state):
    if random.random() < epsilon:
        action = random.choice(actions)
    else:
        q = [getQ(state, a) for a in actions]
        maxQ = max(q)
        count = q.count(maxQ)
        if count > 1:
            best = [i for i in range(len(actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = actions[i]
    return action

# def learn(state1, action1, reward, state2):
#     maxqnew = max([getQ(state2, a) for a in actions])
#     learnQ(state1, action1, reward, reward + gamma*maxqnew)


curr_state = (-GRID, -GRID)
design_mat_f = []
design_mat_t = []

test = []

for state in states:
	for action in actions:
		test.append([state[0], state[0], action[0], action[1]])



for time_step in range(0, 100):

	action = chooseAction(curr_state)
	design_mat_f.append([curr_state[0], curr_state[1], action[0], action[1]])  
	next_state = (max(min(curr_state[0] + action[0], GRID), -GRID) , max(min(curr_state[1] + action[1], GRID), -GRID))
	design_mat_t.append(reward_dynamics(next_state) + gamma * max([getQ(next_state, a) for a in actions]))
	curr_state = next_state
	

gp.fit(design_mat_f, design_mat_t)

heu = gp.predict(test, return_std = True)


# for i in range(0, num_of_episodes):
#     curr_state = (-GRID, -GRID)
    
#     while curr_state != term_state:
#         action = chooseAction(curr_state)
#         next_state = (max(min(curr_state[0] + action[0], GRID), -GRID) , max(min(curr_state[1] + action[1], GRID), -GRID))
#         learn(curr_state, action, reward_dynamics(next_state), next_state)
#         curr_state = next_state
#     print q
#     print '\n'

# for state in states:
#     final_action = chooseAction(state)
#     plt.quiver(state[0], state[1], final_action[0], final_action[1])

# plt.xlim(-GRID - 1, GRID + 1)
# plt.ylim(-GRID - 1, GRID + 1)
# plt.show()

