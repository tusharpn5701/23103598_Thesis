import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class SoccerEnvphase3:#maintaining same env as attacker for dqn
    def __init__(self, goal_width=7.32, goal_height=2.44, field_length=11.0):
        self.goal_width=goal_width
        self.goal_height=goal_height
        self.field_length=field_length
        self.state= self.reset()
    def reset(self,x_position=None y_position=None):
        if x_position is None:
            x_position= np.random.uniform(0,self.field_length)
        if y_position is None:
            y_position=np.random.uniform(0,self.goal_height)
        self.state=np.array([x_position,y_position])
        return self.state
    def step(self,action):
        x, y=np.clip(action,0,[self.goal_width,self.goal_height])
        reward=self._calculate_reward(x, y)
        return self.state,reward
    def _calculate_reward(self, x, y):
        goal_center_x=self.goal_width / 2
        distance_to_center=np.abs(x - goal_center_x)
        max_distance=self.goal_width / 2
        reward_center=1-(distance_to_center / max_distance + 1e-9)
        distance_to_goal=np.sqrt((x-goal_center_x)**2+y**2)
        max_distance_to_goal=np.sqrt((goal_center_x-self.goal_width)**2 + self.goal_height**2)
        intermediate_reward=(1-(distance_to_goal/(max_distance_to_goal+1e-9)))*5
        theta=np.arctan(y / (distance_to_center + 1e-9))
        angle_reward=(1-(np.abs(theta)/(np.pi/2)))*5
        total_reward=(reward_center*5)+intermediate_reward+angle_reward
        total_reward=np.clip(total_reward,-10,10)
        return total_reward
    def is_goal(self, x, y):
        return 0 <= x <= self.goal_width and 0 <= y <= self.goal_height
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1=nn.Linear(state_dim, 64)
        self.fc2=nn.Linear(64, 64)
        self.fc3=nn.Linear(64, action_dim)
    def forward(self, state):
        x=torch.relu(self.fc1(state))
        x=torch.relu(self.fc2(x))
        action=torch.tanh(self.fc3(x))
        return action
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(state_dim + action_dim, 64)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,1)
    def forward(self,state,action):
        x=torch.relu(self.fc1(torch.cat([state, action], 1)))
        x=torch.relu(self.fc2(x))
        q_value=self.fc3(x)
        return q_value
class DDPGAgent:
    def __init__(self,env):
        self.env=env
        self.state_dim=2
        self.action_dim=2
        self.actor=Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target=Actor(self.state_dim, self.action_dim).to(device)
        self.actor_optimizer=optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic=Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target=Critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer=optim.Adam(self.critic.parameters(), lr=0.001)
        self.memory =deque(maxlen=2000)
        self.batch_size= 32
        self.gamma =0.99
        self.tau =0.005#guassian noise for exploration here.
        self.noise_std= 0.2
        
        self.update_target_networks(tau=1.0) #Initialize target networks
    def update_target_networks(self, tau=None):
        if tau is None:
            tau=self.tau
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param,param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau*param.data + (1.0 - tau) * target_param.data)
    def remember(self, state,action,reward, next_state):
        self.memory.append((state, action, reward, next_state))
    def act(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action=self.actor(state).cpu().detach().numpy()[0]
        if noise:
            action+=np.random.normal(0, self.noise_std, size=self.action_dim)
        return np.clip(action,0,1)
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states,actions,rewards,next_states=zip(*minibatch)
        states=torch.FloatTensor(states).to(device)
        actions=torch.FloatTensor(actions).to(device)
        rewards=torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states=torch.FloatTensor(next_states).to(device)
        next_actions=self.actor_target(next_states)
        next_q_values=self.critic_target(next_states, next_actions)
        q_targets=rewards + self.gamma * next_q_values
        q_values = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q_values, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.update_target_networks()
    def train(self,episodes=500, step_limit=10):
        history = []
        goal_counts =[]  
        miss_counts =[]  
        for e in range(episodes):
            state = self.env.reset()
            total_reward= 0
            goals =0
            misses= 0
            for time in range(step_limit):
                action= self.act(state)
                next_state, reward = self.env.step(action)
                self.remember(state, action, reward, next_state)
                state= next_state
                total_reward+= reward
                if self.env.is_goal(action[0], action[1]):
                    goals +=1
                else:
                    misses += 1
                self.replay()
            history.append(total_reward)
            goal_counts.append(goals)
            miss_counts.append(misses)
            if e % 10 == 0:
                print(f"Episode {e}/{episodes}, Reward: {total_reward:.4f}, Goals: {goals}, Misses: {misses}")
        return history, goal_counts, miss_counts
env_phase3 = SoccerEnvPhase3()
ddpg_agent = DDPGAgent(env_phase3)
history_ddpg, goal_counts_ddpg, miss_counts_ddpg = ddpg_agent.train(episodes=500)
def evaluate_agent(agent, env, num_shots=100):
    successful_shots = 0
    for _ in range(num_shots):
        state = env.reset()
        action = agent.act(state, noise=False)
        x, y = action
        if env.is_goal(x, y):
            successful_shots += 1
    success_rate = successful_shots / num_shots
    return success_rate
success_rate_ddpg = evaluate_agent(ddpg_agent, env_phase3)
print(f"DDPG Agent Success Rate: {success_rate_ddpg * 100:.2f}%")
# Perform detailed evaluation
def evaluate_agent_detailed(agent, env, positions, num_shots_per_position=10):
    results = []
    for pos in positions:
        successful_shots = 0
        for _ in range(num_shots_per_position):
            state = env.reset(x_position=pos[0], y_position=pos[1])
            action = agent.act(state, noise=False)
            x, y = action
            if env.is_goal(x, y):
                successful_shots += 1
        success_rate = successful_shots / num_shots_per_position
        results.append((pos, success_rate))
    return results
test_positions = [
    (1.0, 1.0),
    (3.0, 1.0),
    (5.0, 1.0),
    (7.0, 1.0),
    (9.0, 1.0),
    (1.0, 2.0),
    (3.0, 2.0),
    (5.0, 2.0),
    (7.0, 2.0),
    (9.0, 2.0),
    (1.0, 3.0),
    (3.0, 3.0),
    (5.0, 3.0),
    (7.0, 3.0),
    (9.0, 3.0),
]
detailed_results_ddpg = evaluate_agent_detailed(ddpg_agent, env_phase3, test_positions)
for pos, success_rate in detailed_results_ddpg:
    print(f"Position {pos}: Success Rate = {success_rate * 100:.2f}%")
plt.plot(history_ddpg, label='DDPG Reward')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('DDPG Agent Performance Over Time')
plt.legend()
plt.show()
window_size = 10
moving_avg_ddpg = np.convolve(history_ddpg, np.ones(window_size)/window_size, mode='valid')
plt.plot(range(window_size-1, len(history_ddpg)), moving_avg_ddpg, label='DDPG Moving Average', color='red')
plt.xlabel('Episodes')
plt.ylabel('Moving Average Reward')
plt.title('DDPG Moving Average Reward Over Time')
plt.legend()
plt.show()
plt.plot(goal_counts_ddpg, label='DDPG Goals')
plt.plot(miss_counts_ddpg, label='DDPG Misses')
plt.xlabel('Episodes')
plt.ylabel('Count')
plt.title('DDPG Goals and Misses Over Time')
plt.legend()
plt.show()
torch.save(ddpg_agent.actor.state_dict(), 'ddpg_actor_model.pth')
torch.save(ddpg_agent.critic.state_dict(), 'ddpg_critic_model.pth')
