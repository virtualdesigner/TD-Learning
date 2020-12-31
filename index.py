import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')
reward_list = []


def epsilon_greedy_policy(no_of_actions):
    def generate_action(epsilon, action_value_of_s):
        uniform_prob = epsilon/(no_of_actions + 1)
        return np.random.choice([0, 1, 2, 3, 4, 5, np.argmax(action_value_of_s)], p=[uniform_prob, uniform_prob, uniform_prob,
                                                                                     uniform_prob, uniform_prob, uniform_prob, (uniform_prob)+(1-epsilon)])
    return generate_action


def TD(on_policy=False, epsilon=1, total_episodes=500, trajectory_length=500, gamma=0.9, learning_rate=0.6, q_table=np.zeros((500, 6))):

    generate_action = epsilon_greedy_policy(6)

    for episode in range(1, total_episodes):
        observation = env.reset()
        total_reward = 0
        current_action = generate_action(epsilon, q_table[observation])

        for transition in range(trajectory_length):
            env.render()

            old_observation = observation

            action_took = current_action

            observation, reward, done, info = env.step(current_action)

            if(on_policy == True):
                # SARSA
                current_action = generate_action(epsilon, q_table[observation])
            else:
                # Q-Learning
                current_action = np.argmax(q_table[observation])

            q_table[old_observation, action_took] += learning_rate * (reward + (
                gamma * q_table[observation, current_action]) - q_table[old_observation, action_took])

            total_reward += reward

        epsilon = (1/(episode))

        reward_list.append(total_reward)


TD()

env.close()

plt.xlabel('Episode', fontsize=18)
plt.ylabel('Reward', fontsize=18)
plt.plot([j for j in range(1, 500)], reward_list)
plt.show()
