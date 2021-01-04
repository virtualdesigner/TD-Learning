import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')
reward_list = []


def epsilon_greedy_policy(no_of_actions):
    def generate_action(epsilon, action_value_of_s):
        uniform_prob = epsilon/(no_of_actions + 1)
        greedy_action = np.argmax(action_value_of_s)
        random_action = np.random.choice([0, 1, 2, 3, 4, 5, greedy_action], p=[uniform_prob, uniform_prob, uniform_prob,
                                                                               uniform_prob, uniform_prob, uniform_prob, (uniform_prob)+(1-epsilon)])
        if greedy_action == random_action:
            return random_action, True
        else:
            return random_action, False
    return generate_action


def replace_traces(on_policy, eligibility_table, state, action, is_greedy_action, gamma, lamda):
    if(on_policy == True):
        eligibility_table = eligibility_table * (gamma * lamda)
        eligibility_table[state] = 0
        eligibility_table[state, action] = 1
    elif(on_policy == False):
        eligibility_table = eligibility_table * (gamma * lamda)
        if is_greedy_action == True:
            eligibility_table[state, action] = 1
        else:
            eligibility_table = eligibility_table * 0

    return eligibility_table


def accumulate_traces(on_policy, eligibility_table, state, action, is_greedy_action, gamma, lamda):
    if(on_policy == True):
        old_table = eligibility_table
        eligibility_table = eligibility_table * (gamma * lamda)
        eligibility_table[state] = 0
        eligibility_table[state, action] = (
            old_table[state, action] * (gamma * lamda)) + 1
    elif(on_policy == False):
        old_table = eligibility_table
        eligibility_table = eligibility_table * (gamma * lamda)
        if is_greedy_action == True:
            eligibility_table[state, action] = (
                old_table[state, action] * (gamma * lamda)) + 1
        else:
            eligibility_table = eligibility_table * 0

    return eligibility_table


def TD(on_policy=True, epsilon=1, total_episodes=1000, trajectory_length=500, gamma=0.9, learning_rate=0.5, use_eligibility_traces=True, trace_type="accumulate", lamda=0.9, q_table=np.zeros((500, 6))):

    generate_action = epsilon_greedy_policy(6)

    for episode in range(1, total_episodes):
        observation = env.reset()
        total_reward = 0
        current_action, is_greedy_action = generate_action(
            epsilon, q_table[observation])
        eligibility_table = np.zeros(q_table.shape)

        print("---------------- EPISODE STARTS ----------------")

        for transition in range(trajectory_length):
            env.render()

            print("current state: ", observation)

            old_observation = observation

            action_took = current_action

            observation, reward, done, info = env.step(current_action)

            if(use_eligibility_traces == True and trace_type == "replace"):
                # Replacing Traces
                eligibility_table = replace_traces(on_policy, eligibility_table,
                                                   old_observation, action_took, is_greedy_action, gamma, lamda)
            elif(use_eligibility_traces == True and trace_type == "accumulate"):
                # Accumulating Traces
                eligibility_table = accumulate_traces(on_policy, eligibility_table,
                                                      old_observation, action_took, is_greedy_action, gamma, lamda)

            if(on_policy == True):
                # SARSA
                current_action, is_greedy_action = generate_action(
                    epsilon, q_table[observation])
            else:
                # Q-Learning
                current_action, is_greedy_action = generate_action(
                    epsilon, q_table[observation])

            delta = reward + (
                gamma * q_table[observation, current_action]) - q_table[old_observation, action_took]

            if use_eligibility_traces == True:
                delta = delta * eligibility_table[old_observation, action_took]

            q_table[old_observation, action_took] += learning_rate * delta

            total_reward += reward

            if(done):
                print("Total Moves: ", transition)
                print("Reward for this episode: ", total_reward)
                break

        epsilon = (1/(episode))

        reward_list.append(total_reward)

        print("---------------- EPISODE ENDS ----------------\n\n")


TD()

env.close()

plt.xlabel('Episode', fontsize=18)
plt.ylabel('Reward', fontsize=18)
# plt.yt(-1000 * 10, ymax * scale_factor)
plt.plot([j for j in range(1, 1000)], reward_list)
plt.show()
