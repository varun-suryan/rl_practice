import random
from time import sleep
import matplotlib.pyplot as plt

epsilon = 0.02
alpha = 0.2
gamma = 0.9
GRID = 2
num_of_episodes = 30
q = {}

states = [(x,y) for x in range(-GRID, GRID + 1) for y in range(-GRID, GRID + 1)]
actions=[(1,0), (-1,0), (0,1), (0,-1)]



term_state = (GRID, GRID)

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

for time_step in range(0, 100):
	action = chooseAction(curr_state)
	design_mat_f = [curr_state, action]
	next_state = (max(min(curr_state[0] + action[0], GRID), -GRID) , max(min(curr_state[1] + action[1], GRID), -GRID))
	design_mat_t = reward_dynamics(next_state) + gamma * max([getQ(next_state, a) for a in actions])







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

