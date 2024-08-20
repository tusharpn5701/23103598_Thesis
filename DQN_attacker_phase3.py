#phase 3 env
class SoccerEnvPhase3:
    def __init__(self, goal_width=7.32,goal_height=2.44,field_length=11.0):
        self.goal_width = goal_width
        self.goal_height = goal_height
        self.field_length = field_length
        self.state = self.reset()
    def reset(self,x_position=None,y_position=None):
        if x_position is None:
            x_position= np.random.uniform(0, self.field_length)
        if y_position is None:
            y_position =np.random.uniform(0, self.goal_height)
        self.state = np.array([x_position, y_position])
        return self.state
    def step(self, action):
        x,y=np.clip(action,0,[self.goal_width,self.goal_height])
        reward=self._calculate_reward(x,y)
        return self.state,reward
    def calculate_reward(self,x,y):
        goal_center_x=self.goal_width/ 2
        distance_to_center= np.abs(x-goal_center_x)
        max_distance =self.goal_width / 2
        reward_center=1-(distance_to_center/max_distance+1e-9)
        
        distance_to_goal=np.sqrt((x-goal_center_x)**2+y**2)
        max_distance_to_goal=np.sqrt((goal_center_x-self.goal_width)**2+self.goal_height**2)
        intermediate_reward=(1-(distance_to_goal/(max_distance_to_goal+1e-9)))*5
        theta = np.arctan(y / (distance_to_center + 1e-9))#angles reward
        angle_reward=(1-(np.abs(theta)/(np.pi/2)))*5
        total_reward=(reward_center*5)+intermediate_reward+angle_reward
        total_reward=np.clip(total_reward,-10,10)
        return total_reward
    def is_goal(self, x, y):
        return 0<=x<= self.goal_width and 0<=y<= self.goal_height#condition for goal
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(2,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,32)
        self.fc5 = nn.Linear(32,2)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=self.dropout(x)
        x=torch.relu(self.fc2(x))
        x=self.dropout(x)
        x=torch.relu(self.fc3(x))
        x= self.dropout(x)
        x=torch.relu(self.fc4(x))
        x=self.dropout(x)
        return self.fc5(x)
class SoccerAgent:
    def __init__(self,env):
        self.env=env
        self.model=QNetwork().to(device)
        self.model=nn.DataParallel(self.model)
        self.target_model=QNetwork().to(device)
        self.target_model= nn.DataParallel(self.target_model)
        self.update_target_model()
        self.memory=deque(maxlen=2000)
        self.gamma=0.95
        self.epsilon = 0.1
        self.epsilon_max=1.0
        self.epsilon_increase=0.005
        self.batch_size=32
        self.update_freq=1
        self.optimizer=optim_optim.RAdam(self.model.parameters(),lr=0.00005)
    
    def update_target_model(self):
        self.target_model.module.load_state_dict(self.model.module.state_dict()) 
    def remember(self,state,action,reward,next_state):
        self.memory.append((state,action,reward,next_state))
    def act(self, state):
        if np.random.rand()<=self.epsilon:
            return np.random.uniform(0,self.env.goal_width),np.random.uniform(0,self.env.goal_height)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.model(state)
        return action_values.cpu().numpy()[0]
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch =random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            state =torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state= torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward =torch.FloatTensor([reward]).to(device)
            target =reward
            if next_state is not None:
                with torch.no_grad():
                    target = reward + self.gamma * self.target_model(next_state).max(1)[0]
            output=self.model(state)
            target_f=output.clone()
            target_f[0][0]=target
            loss=nn.functional.mse_loss(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increase
    def train(self, episodes=10000, step_limit=10):
        history=[]
        epsilon_values=[]
        goal_counts = [] 
        miss_counts = [] 
        for e in range(episodes):
            state=self.env.reset()
            total_reward=0
            goals=0
            misses=0
            for time in range(step_limit):
                action=self.act(state)
                next_state,reward=self.env.step(action)
                self.remember(state,action,reward,next_state)
                state=next_state
                total_reward+=reward
                if self.env.is_goal(action[0],action[1]):
                    goals+=1 
                else:
                    misses+=1
                self.replay()
            history.append(total_reward)
            epsilon_values.append(self.epsilon)
            goal_counts.append(goals)
            miss_counts.append(misses)
            if e % self.update_freq==0:
                self.update_target_model()
            if e%10==0:
                print(f"Episode {e}/{episodes},Reward:{total_reward:.4f},Goals:{goals},Misses:{misses},Epsilon:{self.epsilon:.2f}")
        return history, epsilon_values,goal_counts,miss_counts


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
env_phase3=SoccerEnvPhase3()
agent_phase3=SoccerAgent(env_phase3)
agent_phase3.model.load_state_dict(torch.load('phase2_model.pth'))
history_phase3,epsilon_values_phase3,goal_counts_phase3,miss_counts_phase3=agent_phase3.train(episodes=1000)
def evaluate_agent(agent,env,num_shots=100):
    successful_shots=0
    for _ in range(num_shots):
        state=env.reset()
        action=agent.act(state)
        x,y=action
        if env.is_goal(x,y):
            successful_shots+=1
    success_rate=successful_shots/num_shots
    return success_rate
success_rate = evaluate_agent(agent_phase3, env_phase3)
print(f"Agent Success Rate: {success_rate * 100:.2f}%")
#evaluation this time includes success rate
def evaluate_agent_detailed(agent, env, positions, num_shots_per_position=10):
    results = []
    for pos in positions:
        successful_shots=0
        for _ in range(num_shots_per_position):
            state=env.reset(x_position=pos[0], y_position=pos[1])
            action=agent.act(state)
            x, y=action
            if env.is_goal(x, y):
                successful_shots+=1
        success_rate=successful_shots/num_shots_per_position
        results.append((pos,success_rate))
    return results
test_positions=[
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


detailed_results=evaluate_agent_detailed(agent_phase3,env_phase3,test_positions)


for pos, success_rate in detailed_results:
    print(f"Position {pos}:Success Rate={success_rate*100:.2f}%")


plt.plot(history_phase3,label='Reward')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Agent Performance Over Time (Phase 3)')
plt.legend()
plt.show()

window_size = 10
moving_avg_phase3 = np.convolve(history_phase3, np.ones(window_size)/window_size, mode='valid')
plt.plot(range(window_size-1, len(history_phase3)), moving_avg_phase3, label='Moving Average', color='red')
plt.xlabel('Episodes')
plt.ylabel('Moving Average Reward')
plt.title('Moving Average Reward Over Time (Phase 3)')
plt.legend()
plt.show()


plt.plot(epsilon_values_phase3,label='Epsilon')
plt.xlabel('Episodes')
plt.ylabel('Epsilon')
plt.title('Epsilon Increase Over Time (Phase 3)')
plt.legend()
plt.show()


plt.plot(goal_counts_phase3,label='Goals')
plt.plot(miss_counts_phase3,label='Misses')
plt.xlabel('Episodes')
plt.ylabel('Count')
plt.title('Goals and Misses Over Time (Phase 3)')
plt.legend()
plt.show()
