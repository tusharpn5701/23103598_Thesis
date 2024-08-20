import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
#this is the combined env which can deal with both goalkeeeper and attacker agents different input demands as mentioned in the thesis chapter 7
class CombinedSoccerEnv:
    def __init__(self, goal_width=7.32, goal_height=2.44, field_length=11.0):
        self.goal_width =goal_width
        self.goal_height= goal_height
        self.field_length= field_length
        self.state=self.reset()
    def reset(self):
        #Initial positions
        shooter_x_position=np.random.uniform(0, self.field_length)
        shooter_y_position= np.random.uniform(0, self.goal_height)
        goalkeeper_x_position=self.goal_width/2
        goalkeeper_y_position=0
        self.state=np.array([shooter_x_position,shooter_y_position,goalkeeper_x_position,goalkeeper_y_position])
        return self.state
    def step(self, shooter_action, goalkeeper_action):
        #shooter action is determining the shot aim
        target_x,target_y=np.clip(shooter_action,[0, 0],[self.field_length,self.goal_height])
        #Goalkeeper action determines movment
        move_x,move_y=np.clip(goalkeeper_action,[0,0],[self.goal_width, self.goal_height])
        self.state=np.array([target_x,target_y,move_x,move_y])
        shooter_reward=self._calculate_shooter_reward(target_x,target_y,move_x,move_y)
        goalkeeper_reward=self._calculate_goalkeeper_reward(target_x,target_y,move_x,move_y)
        return self.state, shooter_reward,goalkeeper_reward
    def _calculate_shooter_reward(self,target_x,target_y,move_x,move_y):
        if self.is_goal(target_x, target_y) and not self.is_save(target_x, target_y, move_x, move_y):
            return 10
        return -10#when shot saved
    def _calculate_goalkeeper_reward(self, target_x, target_y, move_x, move_y):
        if self.is_save(target_x, target_y, move_x, move_y):
            return 10  #Save successful
        return -10  #Failed to save
    def is_goal(self, x, y):
        #Checking as in if the ball is within the goal
        return 0 <= x <= self.goal_width and 0 <= y <= self.goal_height
    def is_save(self, target_x, target_y, gk_x, gk_y):
        #checking if gk is close to bal
        distance_to_ball = np.sqrt((gk_x - target_x)**2 + (gk_y - target_y)**2)
        return distance_to_ball < 0.5
def draw_pitch(ax):
    pitch = patches.Rectangle([0, 0], width=11, height=7.32, edgecolor="black", facecolor='green', lw=2)
    centre_circle = patches.Circle((5.5, 3.66), 9.15 * 11/120, color="white", fill=False, lw=2)
    centre_spot = plt.Circle((5.5, 3.66), 0.8 * 11/120, color="white")
    left_goal = patches.Rectangle([-0.1, (7.32 - 2.44) / 2], width=0.1, height=2.44, fill=True, color="white")
    right_goal = patches.Rectangle([11, (7.32 - 2.44) / 2], width=0.1, height=2.44, fill=True, color="white")
    ax.add_patch(pitch)
    ax.add_patch(centre_circle)
    ax.add_patch(centre_spot)
    ax.add_patch(left_goal)
    ax.add_patch(right_goal)
    plt.plot([5.5, 5.5], [0, 7.32], color="white", lw=2)  # Centre line
    plt.plot([0, 0], [0, 7.32], color="white", lw=2)
    plt.plot([11, 11], [0, 7.32], color="white", lw=2)
    plt.plot([0, 11], [7.32, 7.32], color="white", lw=2)
    plt.plot([0, 11], [0, 0], color="white", lw=2)
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 8.32)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_facecolor('green')
    return ax
def run_combined_simulation(attacker_agent, goalkeeper_agent, num_shots=100):
    env_combined = CombinedSoccerEnv()
    fig, ax = plt.subplots(figsize=(16, 10))
    ax = draw_pitch(ax)
    for _ in range(num_shots):
        state = env_combined.reset()
        shooter_action = attacker_agent.act(state[:2])
        goalkeeper_action = goalkeeper_agent.act([*shooter_action,state[2],state[3]])
        state, shooter_reward, goalkeeper_reward = env_combined.step(shooter_action, goalkeeper_action)
        shooter_x, shooter_y, gk_x, gk_y = state
        if shooter_reward > 0
            ax.arrow(shooter_x, shooter_y, gk_x - shooter_x, gk_y - shooter_y, head_width=0.2, head_length=0.2, fc='red', ec='red', lw=2)
        else:
            ax.arrow(shooter_x, shooter_y, gk_x - shooter_x, gk_y - shooter_y, head_width=0.2, head_length=0.2, fc='blue', ec='blue', lw=2)
            ax.plot(gk_x, gk_y, 'o', markersize=10, color='orange')
    plt.title('Shooter vs Goalkeeper: 100 Shots on a Single Field', fontsize=20, color='white', pad=20)
    plt.show()
attacker_agent = SoccerAgent(env_phase3)
attacker_agent.model.load_state_dict(torch.load('shooter_phase3_model.pth'))
goalkeeper_agent = SoccerGoalkeeperAgent(env_goalkeeper)
goalkeeper_agent.model.load_state_dict(torch.load('goalkeeper_model.pth'))
run_combined_simulation(attacker_agent, goalkeeper_agent, num_shots=100)
