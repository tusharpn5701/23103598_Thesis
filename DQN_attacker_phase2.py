#Phase 2 envrronment
class SoccerEnvPhase2:
    def __init__(self,goal_width=7.32,goal_height=2.44):
        self.goal_width=goal_width
        self.goal_height=goal_height
        self.state=self.reset()
    def reset(self):
        self.state=np.array([0,0])
        return self.state
    def step(self,action):
        x,y=action
        reward=self._calculate_reward(x, y)
        return self.state,reward
    #added an extra reward to the previous reward ie distance to goal ie the intermediate reward
    def calculate_reward(self,x,y):
        goal_center_x=self.goal_width/ 2
        distance_to_center=np.abs(x- goal_center_x)
        max_distance=self.goal_width/2
        reward =1-(distance_to_center/ max_distance)
        distance_to_goal = np.sqrt((x - goal_center_x)**2 + y**2)
        max_distance_to_goal = np.sqrt((goal_center_x - self.goal_width)**2+self.goal_height**2)
        intermediate_reward=(1-(distance_to_goal/max_distance_to_goal)) * 5
        return(reward*10)+intermediate_reward
env_phase2 =SoccerEnvPhase2()
agent_phase2 =SoccerAgent(env_phase2)
agent_phase2.model.load_state_dict(torch.load('phase1_model.pth'))#loading of phase 1 model pre trained weightts
history_phase2,epsilon_values_phase2=agent_phase2.train(episodes=100)
plt.plot(history_phase2,label='Reward')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Agent Performance Over Time (Phase 2)')
plt.legend()
plt.show()
plt.plot(epsilon_values_phase2, label='Epsilon')
plt.xlabel('Episodes')
plt.ylabel('Epsilon')
plt.title('Epsilon Increase Over Time (Phase 2)')
plt.legend()
plt.show()
torch.save(agent_phase2.model.state_dict(), 'phase2_model.pth')
