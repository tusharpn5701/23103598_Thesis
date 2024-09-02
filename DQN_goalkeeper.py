import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import torch_optimizer as optim_optim
class SoccerEnvGoalkeeper:
    def __init__(self, goal_width=7.32, goal_height=2.44):
        self.goal_width = goal_width
        self.goal_height = goal_height
        self.state = self.reset()
    def reset(self, ball_x=None, ball_y=None):
        if ball_x is None:
            ball_x=np.random.uniform(0, self.goal_width)
        if ball_y is None:
            ball_y=np.random.uniform(0, self.goal_height)
        self.state=np.array([ball_x, ball_y, self.goal_width / 2, 0])
        return self.state
    def step(self, action):
        gk_x,gk_y=np.clip(action,[0, 0],[self.goal_width, self.goal_height])
        reward=self._calculate_reward(gk_x, gk_y)
        return np.array([self.state[0],self.state[1],gk_x,gk_y]),reward
    def _calculate_reward(self,gk_x,gk_y):
        ball_x,ball_y=self.state[:2]
        distance_to_ball=np.sqrt((gk_x-ball_x)**2+(gk_y-ball_y)**2)
        max_distance_to_ball=np.sqrt((self.goal_width)**2+(self.goal_height)**2)
        
        reward_distance=(1-(distance_to_ball/max_distance_to_ball))*5
        reward_save=0
        
        if distance_to_ball<0.5:
            reward_save=10 #ball within 0.5 means caught
        elif distance_to_ball<1.0:
            reward_save = 5#ball within 1 means deflected
        #the following is the rewward which the goalkeeper gets if the ball is away from the center when deflected so that it doesnt land in the danger area
        goal_center_x = self.goal_width/2
        distance_from_center=np.abs(ball_x-goal_center_x)
        max_distance_from_center=self.goal_width/2
        reward_away_from_center=(distance_from_center/max_distance_from_center)*5
        return reward_distance + reward_save + reward_away_from_center
    def is_save(self, gk_x, gk_y):
        ball_x, ball_y = self.state[:2]
        distance_to_ball = np.sqrt((gk_x - ball_x)**2 + (gk_y - ball_y)**2)
        return distance_to_ball < 1.5  # Considered a save if within 0.5 units
env_goalkeeper = SoccerEnvGoalkeeper()
#the q network for gk
class QNetworkGoalkeeper(nn.Module):
    def __init__(self):
        super(QNetworkGoalkeeper,self).__init__()
        self.fc1=nn.Linear(4, 64)
        self.fc2=nn.Linear(64, 64)
        self.fc3= nn.Linear(64, 32)
        self.fc4= nn.Linear(32, 32)
        self.fc5 =nn.Linear(32, 2)
        self.dropout =nn.Dropout(0.2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        return self.fc5(x)
class SoccerGoalkeeperAgent:
    def __init__(self, env):
        self.env = env
        self.model = QNetworkGoalkeeper().to(device)
        self.model = nn.DataParallel(self.model)
        self.target_model = QNetworkGoalkeeper().to(device)
        self.target_model = nn.DataParallel(self.target_model)
        self.update_target_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 0.1
        self.epsilon_max = 1.0
        self.epsilon_increase = 0.005
        self.batch_size = 32
        self.update_freq = 1
        self.optimizer = optim_optim.RAdam(self.model.parameters(), lr=0.00005) 
    def update_target_model(self):
        self.target_model.module.load_state_dict(self.model.module.state_dict())
    def remember(self,state,action,reward,next_state):
        self.memory.append((state,action,reward,next_state))
    def act(self,state):
        if np.random.rand()<=self.epsilon:
            return np.random.uniform(0,self.env.goal_width),np.random.uniform(0,self.env.goal_height)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values=self.model(state)
        return action_values.cpu().numpy()[0]
    def replay(self):
        if len(self.memory)<self.batch_size:
            return
        minibatch=random.sample(self.memory, self.batch_size)
        for state,action,reward,next_state in minibatch:
            state=torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state=torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward=torch.FloatTensor([reward]).to(device)
            target=reward
            if next_state is not None:
                with torch.no_grad():
                    target=reward + self.gamma * self.target_model(next_state).max(1)[0]
        
            output=self.model(state)
            target_f=output.clone()
            target_f[0][0]=target
            loss=nn.functional.mse_loss(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon<self.epsilon_max:
            self.epsilon+=self.epsilon_increase
    def train(self, episodes=1000, step_limit=10):
        history = []
        epsilon_values = []
        save_counts = []
        miss_counts = []
        for e in range(episodes):
            state=self.env.reset()
            total_reward= 0
            saves= 0
            misses= 0
            for time in range(step_limit):
                action= self.act(state)
                next_state,reward= self.env.step(action)
                self.remember(state, action, reward, next_state)
                state = next_state
                total_reward+= reward
                if self.env.is_save(action[0], action[1]):
                    saves+= 1
                else:
                    misses+= 1
                self.replay()
            history.append(total_reward)
            epsilon_values.append(self.epsilon)
            save_counts.append(saves)
            miss_counts.append(misses)
            if e%self.update_freq== 0:
                self.update_target_model()
            if e%10== 0:
                print(f"Episode {e}/{episodes}, Reward: {total_reward:.4f}, Saves: {saves}, Misses: {misses}, Epsilon: {self.epsilon:.2f}")
        return history, epsilon_values, save_counts, miss_counts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
env_goalkeeper = SoccerEnvGoalkeeper()
goalkeeper_agent = SoccerGoalkeeperAgent(env_goalkeeper)
# Train the goalkeeper agent
history_goalkeeper, epsilon_values_goalkeeper, save_counts_goalkeeper, miss_counts_goalkeeper = goalkeeper_agent.train(episodes=1000)
#evalutaion with save percentage
def evaluate_goalkeeper(agent,env,num_shots=100):
    successful_saves=0
    for _ in range(num_shots):
        state=env.reset()
        action=agent.act(state)
        if env.is_save(action[0], action[1]):
            successful_saves+=1
    success_rate=successful_saves / num_shots
    return success_rate
success_rate=evaluate_goalkeeper(goalkeeper_agent, env_goalkeeper)
print(f"Goalkeeper Success Rate: {success_rate * 100:.2f}%")
plt.plot(history_goalkeeper,label='Reward')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Goalkeeper Agent Performance Over Time')
plt.legend()
plt.show()
window_size=10
moving_avg_goalkeeper=np.convolve(history_goalkeeper,np.ones(window_size)/window_size, mode='valid')
plt.plot(range(window_size-1,len(history_goalkeeper)),moving_avg_goalkeeper,label='Moving Average',color='red')
plt.xlabel('Episodes')
plt.ylabel('Moving Average Reward')
plt.title('Moving Average Reward Over Time')
plt.legend()
plt.show()
plt.plot(epsilon_values_goalkeeper, label='Epsilon')
plt.xlabel('Episodes')
plt.ylabel('Epsilon')
plt.title('Epsilon Increase Over Time')
plt.legend()
plt.show()
plt.plot(save_counts_goalkeeper, label='Saves')
plt.plot(miss_counts_goalkeeper, label='Misses')
plt.xlabel('Episodes')
plt.ylabel('Count')
plt.title('Saves and Misses Over Time')
plt.legend()
plt.show()

# Save the trained goalkeeper model
torch.save(goalkeeper_agent.model.state_dict(), 'goalkeeper_model.pth')
