import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class SoccerEnvGoalkeeper:
    def __init__(self, goal_width=7.32, goal_height=2.44):
        self.goal_width = goal_width
        self.goal_height = goal_height
        self.state = self.reset()
    def reset(self, ball_x=None, ball_y=None):
        if ball_x is None:
            ball_x= np.random.uniform(0, self.goal_width)
        if ball_y is None:
            ball_y= np.random.uniform(0, self.goal_height)
        self.state = np.array([ball_x, ball_y, self.goal_width / 2, self.goal_height / 2])
        return self.state
    def step(self, action):
        gk_x, gk_y=np.clip(action,[0, 0],[self.goal_width,self.goal_height])
        reward=self._calculate_reward(gk_x, gk_y)
        return np.array([self.state[0], self.state[1], gk_x, gk_y]), reward
    def _calculate_reward(self,gk_x,gk_y):
        ball_x, ball_y=self.state[:2]
        distance_to_ball=np.sqrt((gk_x - ball_x)**2 + (gk_y - ball_y)**2)
        max_distance_to_ball=np.sqrt((self.goal_width)**2 + (self.goal_height)**2)
        reward_distance=(1-(distance_to_ball / max_distance_to_ball)) * 5
        reward_save=0
        if distance_to_ball<0.5:
            reward_save=10#Caught the ball
        elif distance_to_ball < 1.0:
            reward_save=5  #Deflected the ball
        goal_center_x=self.goal_width / 2
        distance_from_center=np.abs(ball_x - goal_center_x)
        max_distance_from_center=self.goal_width / 2
        reward_away_from_center=(distance_from_center / max_distance_from_center) * 5
        
        return reward_distance + reward_save + reward_away_from_center
    def is_save(self,gk_x,gk_y):
        ball_x, ball_y=self.state[:2]
        distance_to_ball=np.sqrt((gk_x - ball_x)**2 + (gk_y - ball_y)**2)
        return distance_to_ball < 1.0 
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
        self.fc2=nn.Linear(64, 64)
        self.fc3=nn.Linear(64, 1)
    def forward(self, state, action):
        x=torch.relu(self.fc1(torch.cat([state, action], 1)))
        x=torch.relu(self.fc2(x))
        q_value=self.fc3(x)
        return q_value
class DDPGAgent:
    def __init__(self, env):
        self.env=env
        self.state_dim=4
        self.action_dim=2
        self.actor=Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target=Actor(self.state_dim, self.action_dim).to(device)
        self.actor_optimizer=optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic=Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target=Critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer=optim.Adam(self.critic.parameters(), lr=0.001)
        self.memory=deque(maxlen=2000)
        self.batch_size=32
        self.gamma=0.99
        self.tau=0.005
        self.noise_std=0.2
        self.update_target_networks(tau=1.0) 
    def update_target_networks(self, tau=None):
        if tau is None:
            tau=self.tau
        for target_param, param in zip(self.actor_target.parameters(),self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 -tau)* target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0-tau)* target_param.data)
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    def act(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action =self.actor(state).cpu().detach().numpy()[0]
        if noise:
            action+= np.random.normal(0, self.noise_std, size=self.action_dim)
        return np.clip(action, -1, 1)
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch= random.sample(self.memory, self.batch_size)
        states,actions, rewards, next_states = zip(*minibatch)
        states =torch.FloatTensor(states).to(device)
        actions= torch.FloatTensor(actions).to(device)
        rewards=torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states=torch.FloatTensor(next_states).to(device)
        next_actions=self.actor_target(next_states)
        next_q_values=self.critic_target(next_states, next_actions)
        q_targets=rewards + self.gamma * next_q_values
        q_values=self.critic(states, actions)
        critic_loss=nn.functional.mse_loss(q_values, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.update_target_networks()
    def train(self, episodes=500, step_limit=10):
        history=[]
        save_counts=[]
        miss_counts=[]
        for e in range(episodes):
            state=self.env.reset()
            total_reward=0
            saves=0
            misses=0
            for time in range(step_limit):
                action=self.act(state)
                next_state, reward = self.env.step(action)
                self.remember(state, action, reward, next_state)
                state=next_state
                total_reward+=reward
                if self.env.is_save(action[0], action[1]):
                    saves+=1
                else:
                    misses+= 1
                self.replay()
            history.append(total_reward)
            save_counts.append(saves)
            miss_counts.append(misses)
            if e % 10==0:
                print(f"Episode {e}/{episodes}, Reward: {total_reward:.4f}, Saves: {saves}, Misses: {misses}")
        return history, save_counts, miss_counts
env_goalkeeper=SoccerEnvGoalkeeper()
goalkeeper_agent=DDPGAgent(env_goalkeeper)
history_goalkeeper,save_counts_goalkeeper,miss_counts_goalkeeper = goalkeeper_agent.train(episodes=500)
#EVALUTION
def evaluate_goalkeeper(agent, env,num_shots=100):
    successful_saves = 0
    for _ in range(num_shots):
        state = env.reset()
        action = agent.act(state, noise=False)
        if env.is_save(action[0], action[1]):
            successful_saves += 1
    success_rate = successful_saves / num_shots
    return success_rate
success_rate = evaluate_goalkeeper(goalkeeper_agent, env_goalkeeper)
print(f"Goalkeeper Success Rate: {success_rate * 100:.2f}%")
plt.plot(history_goalkeeper, label='Reward')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Goalkeeper Agent Performance Over Time')
plt.legend()
plt.show()
window_size = 10
moving_avg_goalkeeper = np.convolve(history_goalkeeper, np.ones(window_size)/window_size, mode='valid')
plt.plot(range(window_size-1, len(history_goalkeeper)), moving_avg_goalkeeper, label='Moving Average', color='red')
plt.xlabel('Episodes')
plt.ylabel('Moving Average Reward')
plt.title('Moving Average Reward Over Time')
plt.legend()
plt.show()
plt.plot(save_counts_goalkeeper, label='Saves')
plt.plot(miss_counts_goalkeeper, label='Misses')
plt.xlabel('Episodes')
plt.ylabel('Count')
plt.title('Saves and Misses Over Time')
plt.legend()
plt.show()
torch.save(goalkeeper_agent.actor.state_dict(), 'goalkeeper_actor_model.pth')
torch.save(goalkeeper_agent.critic.state_dict(), 'goalkeeper_critic_model.pth')
