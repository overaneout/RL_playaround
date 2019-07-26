import gym
import numpy as np

# First approach
# use simple q-learning
LEARNING_RATE = 0.5
DISCOUNT = 0.999
EPISODES = 700
dimensions = [5, 5, 12, 5]
EPSILON = 0.8
EPSILON_DECAY_STOP_EP = int(4/5 * EPISODES)
EPSILON_DECAY = EPSILON/EPSILON_DECAY_STOP_EP

# set q-table
# env state has formate of
# [cart_position, cart_velocity, pole_angle, pole_tip_velocity]
upper_bounds = np.array([4.8000002e+00, 5, 4.1887903e-01, 5])
lower_bounds = np.array([-4.8000002e+00, -5, -4.1887903e-01, -5])

q_table = np.random.uniform(low=0, high=1, size=dimensions + [2])
discrete_windows_sizes = (upper_bounds - lower_bounds) / dimensions


def get_q_position(state, lower_bounds):
    q_position = (state - lower_bounds) / discrete_windows_sizes
    return tuple(q_position.astype(np.int))


# set hyper-parameters

with gym.make("CartPole-v0") as env:
    for ep in range(EPISODES):
        state = env.reset()
        discrete_state = get_q_position(state, lower_bounds)
        done = False
        reward = 0
        steps = 0
        while not done:
            steps += 1
            if np.random.random() > EPSILON:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(2)
            new_state, reward, done, _ = env.step(action)
            new_discrete_state = get_q_position(new_state, lower_bounds)
            if not done:
                current_q = q_table[discrete_state + (action,)]
                max_future_q = np.max(q_table[new_discrete_state])
                new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward+DISCOUNT * max_future_q)
                q_table[discrete_state+(action,)] = new_q
            elif steps < 200:
                q_table[discrete_state+(action,)] = 0
            discrete_state = new_discrete_state
            if ep > (EPISODES*99)//100:
                env.render()
        print(f"current ep {ep}")
        print(steps)
        EPSILON -= EPSILON_DECAY

