import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch_optimizer as optim_optim

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")

#Environment for phase 1
class SoccerEnvPhase1:
    def __init__(self, goal_width=7.32,goal_height=2.44):
        self.goal_width=goal_width
        self.goal_height=goal_height
        self.state = self.reset()
    def reset(self):
        self.state=np.array([0,0])
        return self.state
    def step(self,action):
        x,y=action
        reward=self._calculate_reward(x,y)
        return self.state,reward
    def calculate_reward(self,x,y):
        #This is the closeness reward as mentioned in the thesis
        goal_center_x=self.goal_width/2
        distance_to_center=np.abs(x-goal_center_x)
        max_distance=self.goal_width / 2
        reward=1-(distance_to_center/max_distance)
        return reward*10
#Now the class for Qnetwork
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork,self).__init__()
        self.fc1=nn.Linear(2,64)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,32)
        self.fc4=nn.Linear(32,32)
        self.fc5=nn.Linear(32,2)
        self.dropout=nn.Dropout(0.2)
    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=self.dropout(x)
        x= torch.relu(self.fc2(x))
        x= self.dropout(x)
        x= torch.relu(self.fc3(x))
        x=self.dropout(x)
        x=torch.relu(self.fc4(x))
        x=self.dropout(x)
        return self.fc5(x)
#class Agent for attacker
class SoccerAgent:
    def __init__(self,env):
        self.env=env
        self.model=QNetwork().to(device)
        self.model=nn.DataParallel(self.model)
        self.target_model=QNetwork().to(device)
        self.target_model=nn.DataParallel(self.target_model)
        self.update_target_model()
        self.memory=deque(maxlen=2000)
        self.gamma=0.95 #discount rate gamma
        self.epsilon=0.1#setting the minimum epsilon as I am following epsilon increase
        self.epsilon_max=1.0#max episilon
        self.epsilon_increase = 0.005#Increase epsilon gradually
        self.batch_size = 32#
        self.update_freq = 1
        self.optimizer=optim_optim.RAdam(self.model.parameters(),lr=0.00005)
    def update_target_model(self):
        self.target_model.module.load_state_dict(self.model.module.state_dict())
    def remember(self, state,action,reward,next_state):
        self.memory.append((state,action,reward,next_state))
    def act(self,state):
        if np.random.rand()<=self.epsilon:
            return np.random.uniform(self.env.goal_width/2-0.1,self.env.goal_width/2+0.1),np.random.uniform(0,self.env.goal_height)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values=self.model(state)
        return action_values.cpu().numpy()[0]
    def replay(self):
        if len(self.memory)<self.batch_size:
            return
        minibatch=random.sample(self.memory, self.batch_size)
        for state,action,reward,next_state in minibatch:
            state= torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state =torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward=torch.FloatTensor([reward]).to(device)
            target=reward
            if next_state is not None:
                with torch.no_grad():
                    target=reward+self.gamma*self.target_model(next_state).max(1)[0]
            output = self.model(state)
            # Match the output and target sizes
            target_f=output.clone()
            target_f[0][0]=target
            loss=nn.functional.mse_loss(output,target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increase
    def train(self,episodes=100,step_limit=10):
        history=[]
        epsilon_values=[]#for storge of epsilon
        for e in range(episodes):
            state =self.env.reset()
            total_reward= 0
            for time in range(step_limit):
                action =self.act(state)
                next_state, reward =self.env.step(action)
                self.remember(state,action,reward, next_state)
                state =next_state
                total_reward += reward
                self.replay()
            history.append(total_reward)
            epsilon_values.append(self.epsilon)#keeping track of the current value of epsilon
            if e%self.update_freq == 0:
                self.update_target_model()
            if e%10== 0:
                print(f"Episode {e}/{episodes}, Reward: {total_reward:.4f}, Epsilon: {self.epsilon:.2f}")
        return history, epsilon_values
env_phase1 = SoccerEnvPhase1()
agent_phase1 = SoccerAgent(env_phase1)
#training
history_phase1,epsilon_values_phase1=agent_phase1.train(episodes=100)  # Fewer episodes for phase 1
#evaluation
plt.plot(history_phase1,label='Reward')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Agent Performance Over Time (Phase 1)')
window_size = 10
moving_avg_phase1=np.convolve(history_phase1, np.ones(window_size)/window_size, mode='valid')
plt.plot(range(window_size-1,len(history_phase1)),moving_avg_phase1,label='Moving Average',color='red')
plt.legend()
plt.show()
plt.plot(epsilon_values_phase1, label='Epsilon')
plt.xlabel('Episodes')
plt.ylabel('Epsilon')
plt.title('Epsilon Increase Over Time (Phase 1)')
plt.legend()
plt.show()
