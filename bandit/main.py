import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.Counts = np.zeros(action_size)

    def update(self, action, reward):
        self.Counts[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.Counts[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)


steps = 1000
epsilon = 0.1

total_reward = 0
total_rewards = []
rates = []

bandit = Bandit()
agent = Agent(epsilon)

for step in range(steps):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
    total_reward += reward
    total_rewards.append(total_reward)
    rates.append(total_reward / (step + 1))

print(total_reward)

plt.ylabel('Total reward')
plt.xlabel('steps')
plt.plot(total_rewards)
plt.show()

plt.ylabel('rates')
plt.xlabel('steps')
plt.plot(rates)
plt.show()
