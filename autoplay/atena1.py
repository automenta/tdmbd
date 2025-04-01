# -*- coding: utf-8 -*-
"""
ATENA (Adversarial Task Evolving Neuro Agent) Proof-of-Concept

Dependencies:
- Python 3.8+
- PyTorch (tested with 2.0+)
- Dear PyGui (tested with 1.9+)

Installation:
pip install torch dearpygui

Instructions to Run:
python atena_poc.py

Description:
This script implements a simplified proof-of-concept of the ATENA system.
It features:
1. A Solver Agent (Actor-Critic RL) learning to navigate a simple grid world.
2. A Task Generator Agent adversarially learning to propose goal locations
   in the grid world to maximize the Solver's learning progress (approximated
   by targeting a 'sweet spot' of task difficulty).
3. An interactive GUI using Dear PyGui to visualize the training process,
   agent/generator behavior, performance metrics, and allow hyperparameter tuning.

Visualization Focus:
- Solver Pane: Grid world view, agent's path, rewards, losses.
- Generator Pane: Task difficulty distribution (goal locations), generator rewards.
- Overall statistics and control panel.

Note: This is a simplified PoC for demonstration and visualization. The models,
task complexity, and training algorithms are minimal for interactive performance
on commodity hardware (<10GB VRAM, often runnable on CPU). It demonstrates the
*concept* of adversarial task generation for efficient learning, not state-of-the-art
performance.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
import numpy as np
import dearpygui.dearpygui as dpg
import time
import math
import collections
import threading
import queue

solver_size = 256

# --- Configuration ---
CONFIG = {
    "grid_size": 4,
    "max_episode_steps": 30,
    "solver_lr": 3e-4,
    "generator_lr": 5e-4,
    "gamma": 0.99,  # Discount factor for solver
    "solver_entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "solver_updates_per_task": 5,  # How many updates solver performs per generated task
    "solver_episodes_per_update": 10,  # Episodes collected before solver update
    "generator_target_return_mean": 0.6,  # Target normalized return (0=min, 1=max)
    "generator_target_return_std": 0.2,  # How wide the "sweet spot" is
    "stats_window": 100,  # Rolling window size for stats
    "update_interval_ms": 50,  # GUI update frequency
    "device": "cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 2e9 else "cpu"
    # Use GPU if available and > 2GB
}

print(f"Using device: {CONFIG['device']}")
DEVICE = torch.device(CONFIG["device"])


# --- Environment ---
class SimpleGridEnv:
    """A simple grid world environment."""

    def __init__(self, grid_size, max_steps, start_pos=None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self._action_to_direction = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]),  # Down
            2: np.array([0, -1]),  # Left
            3: np.array([0, 1]),  # Right
        }
        self.action_space_n = len(self._action_to_direction)
        # State: [agent_x, agent_y, goal_x, goal_y] normalized to [0, 1]
        self.observation_space_shape = (4,)
        self.start_pos = start_pos if start_pos else (grid_size // 2, grid_size // 2)
        self.goal_pos = None
        self.agent_pos = None
        self.current_step = 0

    def _get_obs(self):
        obs = np.array([
            self.agent_pos[0] / (self.grid_size - 1),
            self.agent_pos[1] / (self.grid_size - 1),
            self.goal_pos[0] / (self.grid_size - 1),
            self.goal_pos[1] / (self.grid_size - 1)
        ], dtype=np.float32)
        return obs

    def reset(self, goal_pos):
        """Resets the environment to a starting state with a new goal."""
        self.goal_pos = np.array(goal_pos)
        self.agent_pos = np.array(self.start_pos)
        self.current_step = 0
        if np.array_equal(self.agent_pos, self.goal_pos):  # Ensure start != goal
            # If goal is same as start, shift goal slightly
            self.goal_pos = np.array([(goal_pos[0] + 1) % self.grid_size, goal_pos[1]])
            if np.array_equal(self.agent_pos, self.goal_pos):  # If still same, shift other way
                self.goal_pos = np.array([goal_pos[0], (goal_pos[1] + 1) % self.grid_size])

        return self._get_obs()

    def step(self, action):
        """Takes an action and returns the next state, reward, done flag."""
        if self.goal_pos is None:
            raise ValueError("Goal position not set. Call reset first.")

        direction = self._action_to_direction[action]
        new_pos = self.agent_pos + direction

        # Clip agent position to grid boundaries
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)
        self.agent_pos = new_pos
        self.current_step += 1

        done = False
        reward = -0.01  # Small penalty for each step

        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 1.0
            done = True
        elif self.current_step >= self.max_steps:
            done = True
            # Optional: distance-based reward on timeout
            # dist = np.linalg.norm(self.agent_pos - self.goal_pos)
            # max_dist = np.sqrt(2 * (self.grid_size - 1)**2)
            # reward = -0.1 * (dist / max_dist)

        return self._get_obs(), reward, done, {}  # Add empty info dict

    def get_state_for_render(self):
        """Returns grid state for rendering."""
        grid = np.zeros((self.grid_size, self.grid_size))
        if self.agent_pos is not None:
            grid[self.agent_pos[0], self.agent_pos[1]] = 0.5  # Agent color
        if self.goal_pos is not None:
            grid[self.goal_pos[0], self.goal_pos[1]] = 1.0  # Goal color
        return grid

    def get_max_possible_return(self):
        # Best case: 1 step to goal (if adjacent) -> reward 1.0
        # Worst case: timeout -> reward approx -0.01 * max_steps
        # Let's normalize based on achieving the goal vs timeout penalty
        return 1.0

    def get_min_possible_return(self):
        return -0.01 * self.max_steps


# --- Solver Agent (Actor-Critic) ---
class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_shape[0], solver_size),
            nn.Tanh(),
            nn.Linear(solver_size, solver_size),
            nn.Tanh(),
            nn.Linear(solver_size, action_dim)  # Output logits for policy
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_shape[0], solver_size),
            nn.Tanh(),
            nn.Linear(solver_size, solver_size),
            nn.Tanh(),
            nn.Linear(solver_size, 1)  # Output state value
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


class SolverTrainer:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent = ActorCritic(env.observation_space_shape, env.action_space_n).to(DEVICE)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config["solver_lr"])
        self.memory = collections.deque(maxlen=10000)  # Simple replay buffer for steps
        self.episode_rewards = collections.deque(maxlen=config["stats_window"])
        self.policy_losses = collections.deque(maxlen=config["stats_window"])
        self.value_losses = collections.deque(maxlen=config["stats_window"])
        self.entropy_losses = collections.deque(maxlen=config["stats_window"])
        self.current_goal = None  # Track the goal for the current batch

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        self.memory.append((state, action, reward, next_state, done, log_prob, value))

    def compute_returns_advantages(self, rewards, values, dones, next_value):
        """Computes discounted returns and advantages."""
        returns = []
        advantages = []
        R = next_value  # Bootstrap from last state value if not done
        gae = 0.0  # Generalized Advantage Estimation

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.config["gamma"] * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.config["gamma"] * 0.95 * (1 - dones[step]) * gae  # Using lambda=0.95 for GAE
            advantages.insert(0, gae)
            R = rewards[step] + self.config["gamma"] * R * (1 - dones[step])
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
        return returns, advantages

    def collect_experience(self, num_episodes, goal_pos):
        """Collects experience for a given number of episodes with a fixed goal."""
        batch_states, batch_actions, batch_rewards, batch_dones = [], [], [], []
        batch_log_probs, batch_values = [], []
        total_steps = 0
        batch_episode_rewards = []
        self.current_goal = goal_pos

        for _ in range(num_episodes):
            state = self.env.reset(goal_pos)
            episode_reward = 0
            done = False
            episode_states, episode_actions, episode_rewards, episode_dones = [], [], [], []
            episode_log_probs, episode_values = [], []

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits, value = self.agent(state_tensor)

                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_state, reward, done, _ = self.env.step(action.item())

                episode_states.append(state)
                episode_actions.append(action.item())
                episode_rewards.append(reward)
                episode_dones.append(done)
                episode_log_probs.append(log_prob.item())
                episode_values.append(value.item())

                state = next_state
                episode_reward += reward
                total_steps += 1

            batch_states.extend(episode_states)
            batch_actions.extend(episode_actions)
            batch_rewards.extend(episode_rewards)
            batch_dones.extend(episode_dones)
            batch_log_probs.extend(episode_log_probs)
            batch_values.extend(episode_values)  # Value for each state
            batch_episode_rewards.append(episode_reward)

            # Add value of final state for return calculation
            if done:
                last_value = 0.0  # Terminal state value is 0
            else:  # Should not happen if max_steps is handled correctly, but for safety
                with torch.no_grad():
                    _, last_value_tensor = self.agent(torch.FloatTensor(state).unsqueeze(0).to(DEVICE))
                last_value = last_value_tensor.item()
            batch_values.append(last_value)  # Value needed for GAE calculation of the last step

        # Prepare data for update
        states_tensor = torch.FloatTensor(np.array(batch_states)).to(DEVICE)
        actions_tensor = torch.LongTensor(batch_actions).to(DEVICE)
        rewards_tensor = torch.FloatTensor(batch_rewards).to(DEVICE)
        dones_tensor = torch.BoolTensor(batch_dones).to(DEVICE)
        old_log_probs_tensor = torch.FloatTensor(batch_log_probs).to(DEVICE)
        values_tensor = torch.FloatTensor(batch_values[:-1]).to(DEVICE)  # Exclude the appended final state value

        # Compute returns and advantages using GAE
        next_value = batch_values[-1]  # The value estimated for the state *after* the last action
        returns, advantages = self.compute_returns_advantages(batch_rewards, batch_values[:-1], batch_dones, next_value)

        self.episode_rewards.extend(batch_episode_rewards)  # Track overall performance

        return states_tensor, actions_tensor, returns, advantages, old_log_probs_tensor, np.mean(
            batch_episode_rewards), np.std(batch_episode_rewards)

    def update(self, states, actions, returns, advantages, old_log_probs):
        """Performs one Actor-Critic update."""
        # Re-evaluate actions and values for current policy
        logits, values = self.agent(states)
        values = values.squeeze()
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Policy Loss (Actor loss) - Using simple policy gradient objective here
        # For PPO, you'd compute ratio = (new_log_probs - old_log_probs).exp()
        # and use clipped objective: torch.min(ratio * advantages, torch.clamp(ratio, 1-eps, 1+eps) * advantages)
        policy_loss = -(new_log_probs * advantages).mean()  # Simple PG loss

        # Value Loss (Critic loss)
        value_loss = F.mse_loss(values, returns)

        # Total Loss
        loss = (policy_loss
                + self.config["value_loss_coef"] * value_loss
                - self.config["solver_entropy_coef"] * entropy)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Record losses
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.entropy_losses.append(entropy.item())

        return policy_loss.item(), value_loss.item(), entropy.item()


# --- Task Generator Agent ---
class TaskGenerator(nn.Module):
    """Generates task parameters (goal locations)."""

    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.num_possible_goals = grid_size * grid_size
        # Simple model: learns a probability distribution over all possible goal locations
        # No input state for now, just learns based on rewards
        self.goal_logits = nn.Parameter(torch.zeros(self.num_possible_goals))

    def forward(self):
        return self.goal_logits

    def sample_goal(self):
        """Samples a goal location based on current logits."""
        logits = self.forward()
        dist = Categorical(logits=logits)
        goal_index = dist.sample()
        log_prob = dist.log_prob(goal_index)

        # Convert index back to (row, col) coordinates
        row = goal_index // self.grid_size
        col = goal_index % self.grid_size
        return (row.item(), col.item()), goal_index, log_prob

    def get_distribution(self):
        """Returns the probability distribution over goals."""
        with torch.no_grad():
            probs = F.softmax(self.forward(), dim=0)
        return probs.cpu().numpy()


class GeneratorTrainer:
    def __init__(self, generator, config):
        self.generator = generator.to(DEVICE)
        self.config = config
        self.optimizer = optim.Adam(self.generator.parameters(), lr=config["generator_lr"])
        self.rewards = collections.deque(maxlen=config["stats_window"])
        self.task_history = collections.deque(maxlen=config["stats_window"] * 5)  # Store (goal_index, log_prob, reward)

    def get_generator_reward(self, solver_mean_return, min_return, max_return):
        """Calculates reward for the generator based on solver performance."""
        # Normalize solver return to [0, 1] range
        normalized_return = (solver_mean_return - min_return) / (max_return - min_return + 1e-6)
        normalized_return = np.clip(normalized_return, 0.0, 1.0)

        # Reward is based on how close the performance is to the target "sweet spot"
        # Using a Gaussian-like reward function centered at the target mean
        target_mean = self.config["generator_target_return_mean"]
        target_std = self.config["generator_target_return_std"]
        reward = math.exp(-0.5 * ((normalized_return - target_mean) / target_std) ** 2)

        # Bonus for being exactly in the sweet spot? Maybe not needed.
        # Penalty for being too easy (normalized_return near 1) or too hard (near 0)? Implicit in Gaussian.
        # Add a small constant reward to encourage exploration?
        # reward += 0.01

        self.rewards.append(reward)
        return reward

    def store_task_result(self, goal_index, log_prob, reward):
        """Stores the result of a generated task."""
        self.task_history.append((goal_index, log_prob, reward))

    def update(self):
        """Updates the generator using REINFORCE-like update based on task results."""
        if not self.task_history:
            return 0.0  # No data to update from

        # Simple REINFORCE: reward * log_prob
        # We can average over the recent history
        policy_loss = 0
        valid_samples = 0

        # In a more complex setup, you might use batches or importance sampling
        # For simplicity, let's just use the last task result for the update
        # This is noisy but works for a simple PoC

        if self.task_history:
            _, last_log_prob, last_reward = self.task_history[-1]

            # Baseline: subtract mean reward to reduce variance (optional but good)
            baseline = np.mean(self.rewards) if self.rewards else 0

            policy_loss = -last_log_prob * (last_reward - baseline)
            valid_samples = 1

        if valid_samples > 0:
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            return policy_loss.item()
        else:
            return 0.0


# --- Dear PyGui Application ---
class AtenaDemo:
    def __init__(self, config):
        self.config = config
        self.env = SimpleGridEnv(config["grid_size"], config["max_episode_steps"])
        self.solver = SolverTrainer(self.env, config)
        self.generator = TaskGenerator(config["grid_size"])
        self.generator_trainer = GeneratorTrainer(self.generator, config)

        # --- State ---
        self.is_running = False
        self.training_thread = None
        self.ui_update_queue = queue.Queue(maxsize=10)  # To pass data from training thread to GUI

        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.total_generator_updates = 0
        self.last_solver_perf_mean = 0.0
        self.last_solver_perf_std = 0.0
        self.last_generator_reward = 0.0

        # Visualization data
        self.solver_rewards_hist = []
        self.solver_policy_loss_hist = []
        self.solver_value_loss_hist = []
        self.generator_rewards_hist = []
        self.generator_dist_hist = np.zeros(self.generator.num_possible_goals)  # Average distribution
        self.goal_vis_grid = np.zeros((config["grid_size"], config["grid_size"]))
        self.render_grid = self.env.get_state_for_render()

        # Setup DPG
        dpg.create_context()
        self.setup_dpg_gui()
        dpg.create_viewport(title=f"ATENA PoC (Device: {DEVICE})", width=1350, height=750)
        dpg.setup_dearpygui()

    def _update_config_value(self, sender, app_data, user_data):
        """Updates config value from GUI."""
        key, type_func = user_data
        try:
            self.config[key] = type_func(app_data)
            print(f"Updated config: {key} = {self.config[key]}")
            # Update related components if necessary
            if key == 'solver_lr':
                for param_group in self.solver.optimizer.param_groups:
                    param_group['lr'] = self.config['solver_lr']
            elif key == 'generator_lr':
                for param_group in self.generator_trainer.optimizer.param_groups:
                    param_group['lr'] = self.config['generator_lr']
            # Other hyperparameters might require restarting the training thread for full effect
        except ValueError:
            print(f"Invalid value for {key}: {app_data}")
            # Revert GUI element to current config value
            dpg.set_value(sender, self.config[key])
        except Exception as e:
            print(f"Error updating {key}: {e}")
            dpg.set_value(sender, self.config[key])

    def setup_dpg_gui(self):
        # --- Themes ---
        with dpg.theme() as self.red_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 50, 50), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (230, 80, 80), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (180, 30, 30), category=dpg.mvThemeCat_Core)

        with dpg.theme() as self.green_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 200, 50), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 230, 80), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (30, 180, 30), category=dpg.mvThemeCat_Core)

        # --- Main Window ---
        with dpg.window(label="ATENA Demo", tag="MainWindow"):
            with dpg.group(horizontal=True):
                # --- Control Panel ---
                with dpg.child_window(label="Controls", width=250, height=-1):
                    dpg.add_text("--- Controls ---")
                    with dpg.group(horizontal=True):
                        self.start_button = dpg.add_button(label="Start Training", callback=self.start_training)
                        dpg.bind_item_theme(self.start_button, self.green_theme)
                        self.stop_button = dpg.add_button(label="Stop Training", callback=self.stop_training,
                                                          enabled=False)
                        dpg.bind_item_theme(self.stop_button, self.red_theme)
                    dpg.add_button(label="Reset All", callback=self.reset_all)

                    dpg.add_separator()
                    dpg.add_text("--- Statistics ---")
                    dpg.add_text("Status: Idle", tag="status_text")
                    dpg.add_text(f"Device: {CONFIG['device']}", tag="device_text")
                    dpg.add_text("Total Steps: 0", tag="total_steps_text")
                    dpg.add_text("Total Episodes: 0", tag="total_episodes_text")
                    dpg.add_text("Generator Updates: 0", tag="gen_updates_text")
                    dpg.add_text("--- Solver Stats (Avg) ---")
                    dpg.add_text("Avg Reward: N/A", tag="solver_avg_reward_text")
                    dpg.add_text("Policy Loss: N/A", tag="solver_policy_loss_text")
                    dpg.add_text("Value Loss: N/A", tag="solver_value_loss_text")
                    dpg.add_text("--- Generator Stats (Avg) ---")
                    dpg.add_text("Avg Reward: N/A", tag="gen_avg_reward_text")
                    dpg.add_text("Target Return Mean: N/A", tag="gen_target_ret_text")

                    dpg.add_separator()
                    dpg.add_text("--- Hyperparameters (Runtime) ---")
                    dpg.add_input_float(label="Solver LR", default_value=self.config['solver_lr'], step=0.00001,
                                        format="%.5f", callback=self._update_config_value,
                                        user_data=('solver_lr', float))
                    dpg.add_input_float(label="Generator LR", default_value=self.config['generator_lr'], step=0.00001,
                                        format="%.5f", callback=self._update_config_value,
                                        user_data=('generator_lr', float))
                    dpg.add_input_float(label="Gamma (Solver)", default_value=self.config['gamma'], step=0.001,
                                        format="%.3f", callback=self._update_config_value, user_data=('gamma', float))
                    dpg.add_input_float(label="Entropy Coef", default_value=self.config['solver_entropy_coef'],
                                        step=0.001, format="%.3f", callback=self._update_config_value,
                                        user_data=('solver_entropy_coef', float))
                    dpg.add_input_float(label="Target Return Mean",
                                        default_value=self.config['generator_target_return_mean'], step=0.01,
                                        format="%.2f", callback=self._update_config_value,
                                        user_data=('generator_target_return_mean', float))
                    dpg.add_input_float(label="Target Return Std",
                                        default_value=self.config['generator_target_return_std'], step=0.01,
                                        format="%.2f", callback=self._update_config_value,
                                        user_data=('generator_target_return_std', float))

                    dpg.add_separator()
                    dpg.add_text("--- Hyperparameters (Restart Req.) ---")
                    # dpg.add_input_int(label="Grid Size", default_value=self.config['grid_size'], step=1, callback=self._update_config_value, user_data=('grid_size', int), enabled=False) # Requires restart
                    # dpg.add_input_int(label="Max Episode Steps", default_value=self.config['max_episode_steps'], step=1, callback=self._update_config_value, user_data=('max_episode_steps', int), enabled=False) # Requires restart
                    dpg.add_input_int(label="Solver Updates/Task", default_value=self.config['solver_updates_per_task'],
                                      step=1, callback=self._update_config_value,
                                      user_data=('solver_updates_per_task', int))
                    dpg.add_input_int(label="Solver Episodes/Update",
                                      default_value=self.config['solver_episodes_per_update'], step=1,
                                      callback=self._update_config_value, user_data=('solver_episodes_per_update', int))

                # --- Main Content Area ---
                with dpg.group():
                    with dpg.group(horizontal=True):
                        # --- Solver Pane ---
                        with dpg.child_window(label="Solver Agent", width=520, height=350):
                            dpg.add_text("Solver Agent View")
                            with dpg.group(horizontal=True):
                                dpg.add_text("Grid World:")
                                dpg.add_text("Current Goal: N/A", tag="current_goal_text")
                            # Placeholder for grid visualization
                            grid_draw_size = 300
                            cell_size = grid_draw_size // self.config['grid_size']
                            with dpg.drawlist(width=grid_draw_size, height=grid_draw_size, tag="grid_drawlist"):
                                # Draw initial grid lines
                                for i in range(self.config['grid_size'] + 1):
                                    dpg.draw_line((0, i * cell_size), (grid_draw_size, i * cell_size),
                                                  color=(100, 100, 100, 255), thickness=1)
                                    dpg.draw_line((i * cell_size, 0), (i * cell_size, grid_draw_size),
                                                  color=(100, 100, 100, 255), thickness=1)
                            # Need to add cells inside render loop
                            self.grid_cell_tags = {}  # Store tags for grid cells for updating color
                            for r in range(self.config['grid_size']):
                                for c in range(self.config['grid_size']):
                                    tag = f"cell_{r}_{c}"
                                    self.grid_cell_tags[(r, c)] = dpg.draw_rectangle((c * cell_size, r * cell_size),
                                                                                     ((c + 1) * cell_size,
                                                                                      (r + 1) * cell_size),
                                                                                     parent="grid_drawlist",
                                                                                     color=(0, 0, 0, 0),
                                                                                     fill=(50, 50, 50, 255),
                                                                                     tag=tag)  # Initially dark gray

                        # --- Generator Pane ---
                        with dpg.child_window(label="Task Generator", width=520, height=350):
                            dpg.add_text("Task Generator View")
                            dpg.add_text("Task Distribution (Goal Location Probability):")
                            # Placeholder for generator distribution visualization (heatmap)
                            gen_draw_size = 300
                            gen_cell_size = gen_draw_size // self.config['grid_size']
                            with dpg.drawlist(width=gen_draw_size, height=gen_draw_size, tag="gen_dist_drawlist"):
                                for i in range(self.config['grid_size'] + 1):
                                    dpg.draw_line((0, i * gen_cell_size), (gen_draw_size, i * gen_cell_size),
                                                  color=(100, 100, 100, 255), thickness=1)
                                    dpg.draw_line((i * gen_cell_size, 0), (i * gen_cell_size, gen_draw_size),
                                                  color=(100, 100, 100, 255), thickness=1)
                            self.gen_dist_cell_tags = {}
                            for r in range(self.config['grid_size']):
                                for c in range(self.config['grid_size']):
                                    tag = f"gen_cell_{r}_{c}"
                                    self.gen_dist_cell_tags[(r, c)] = dpg.draw_rectangle(
                                        (c * gen_cell_size, r * gen_cell_size),
                                        ((c + 1) * gen_cell_size, (r + 1) * gen_cell_size), parent="gen_dist_drawlist",
                                        color=(0, 0, 0, 0), fill=(0, 0, 50, 255), tag=tag)  # Initially dark blue

                    # --- Plots Area ---
                    with dpg.child_window(label="Plots", height=320):
                        with dpg.group(horizontal=True):
                            # Solver Plots
                            with dpg.plot(label="Solver Performance", height=280, width=500):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="Training Steps", auto_fit=True)
                                with dpg.plot_axis(dpg.mvYAxis, label="Value", tag="solver_y_axis", auto_fit=True):
                                    dpg.add_line_series([], [], label="Avg Episode Reward", tag="solver_reward_series")
                                    dpg.add_line_series([], [], label="Policy Loss", tag="solver_policy_loss_series")
                                    dpg.add_line_series([], [], label="Value Loss", tag="solver_value_loss_series")
                                # dpg.set_axis_limits("solver_y_axis", -1, 2) # Adjust as needed

                            # Generator Plots
                            with dpg.plot(label="Generator Performance", height=280, width=500):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="Generator Updates", auto_fit=True)
                                with dpg.plot_axis(dpg.mvYAxis, label="Value", tag="gen_y_axis", auto_fit=True):
                                    dpg.add_line_series([], [], label="Avg Generator Reward", tag="gen_reward_series")
                                # dpg.set_axis_limits("gen_y_axis", 0, 1.1)

        # --- Render Loop Setup ---
        dpg.set_primary_window("MainWindow", True)
        # Use set_render_callback for DPG versions >= 1.9
        dpg.set_frame_callback(frame=dpg.get_frame_count() + 2,
                               callback=self.render_loop)
        # For older versions use: dpg.set_render_loop_callback(self.render_loop)

    def render_loop(self):
        """GUI Render Loop Callback."""
        # Process updates from the training thread
        while not self.ui_update_queue.empty():
            try:
                update_data = self.ui_update_queue.get_nowait()
                self.update_ui(update_data)
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing UI queue: {e}")

        # Render the current state of the grid
        self.render_grid_env()
        # Render the generator distribution heatmap
        self.render_generator_distribution()

    def update_ui(self, data):
        """Updates UI elements with new data from the training thread."""
        if not dpg.is_dearpygui_running(): return  # Exit if GUI closed

        # Update statistics text
        dpg.set_value("status_text", f"Status: {'Running' if self.is_running else 'Stopped'}")
        dpg.set_value("total_steps_text", f"Total Steps: {self.total_steps}")
        dpg.set_value("total_episodes_text", f"Total Episodes: {self.total_episodes}")
        dpg.set_value("gen_updates_text", f"Generator Updates: {self.total_generator_updates}")

        if self.solver.episode_rewards:
            avg_rew = np.mean(self.solver.episode_rewards)
            dpg.set_value("solver_avg_reward_text", f"Avg Reward: {avg_rew:.3f}")
        if self.solver.policy_losses:
            avg_p_loss = np.mean(self.solver.policy_losses)
            dpg.set_value("solver_policy_loss_text", f"Policy Loss: {avg_p_loss:.4f}")
        if self.solver.value_losses:
            avg_v_loss = np.mean(self.solver.value_losses)
            dpg.set_value("solver_value_loss_text", f"Value Loss: {avg_v_loss:.4f}")

        if self.generator_trainer.rewards:
            avg_gen_rew = np.mean(self.generator_trainer.rewards)
            dpg.set_value("gen_avg_reward_text", f"Avg Reward: {avg_gen_rew:.3f}")
        dpg.set_value("gen_target_ret_text",
                      f"Target Return: {self.config['generator_target_return_mean']:.2f} +/- {self.config['generator_target_return_std']:.2f}")

        # Update plots - Add new data points
        max_points = 200  # Limit number of points for performance
        # Solver plots
        if "solver_reward" in data: self.solver_rewards_hist.append(data["solver_reward"])
        if "policy_loss" in data: self.solver_policy_loss_hist.append(data["policy_loss"])
        if "value_loss" in data: self.solver_value_loss_hist.append(data["value_loss"])

        if len(self.solver_rewards_hist) > 1:
            steps_axis = list(range(len(self.solver_rewards_hist)))
            dpg.set_value("solver_reward_series", [steps_axis[-max_points:], self.solver_rewards_hist[-max_points:]])
        if len(self.solver_policy_loss_hist) > 1:
            steps_axis_p = list(range(len(self.solver_policy_loss_hist)))
            dpg.set_value("solver_policy_loss_series",
                          [steps_axis_p[-max_points:], self.solver_policy_loss_hist[-max_points:]])
        if len(self.solver_value_loss_hist) > 1:
            steps_axis_v = list(range(len(self.solver_value_loss_hist)))
            dpg.set_value("solver_value_loss_series",
                          [steps_axis_v[-max_points:], self.solver_value_loss_hist[-max_points:]])

        # Generator plot
        if "generator_reward" in data: self.generator_rewards_hist.append(data["generator_reward"])
        if len(self.generator_rewards_hist) > 1:
            gen_updates_axis = list(range(len(self.generator_rewards_hist)))
            dpg.set_value("gen_reward_series",
                          [gen_updates_axis[-max_points:], self.generator_rewards_hist[-max_points:]])

        # Update visual grid state
        if "render_grid" in data:
            self.render_grid = data["render_grid"]
        if "current_goal" in data:
            dpg.set_value("current_goal_text", f"Current Goal: {data['current_goal']}")

        # Update generator distribution heatmap data
        if "generator_distribution" in data:
            # Use moving average for smoother visualization
            self.generator_dist_hist = 0.95 * self.generator_dist_hist + 0.05 * data["generator_distribution"]

        # Auto-adjust y-axes limits (optional, can be slow)
        # dpg.fit_axis_data("solver_y_axis")
        # dpg.fit_axis_data("gen_y_axis")

    def render_grid_env(self):
        """Renders the grid world in the drawlist."""
        if not dpg.is_dearpygui_running(): return

        grid = self.render_grid
        grid_size = self.config['grid_size']
        draw_size = 300  # Must match setup
        cell_size = draw_size // grid_size

        for r in range(grid_size):
            for c in range(grid_size):
                tag = self.grid_cell_tags.get((r, c))
                if tag:
                    cell_value = grid[r, c]
                    color = (50, 50, 50, 255)  # Background
                    if cell_value == 0.5:  # Agent
                        color = (50, 200, 50, 255)  # Green
                    elif cell_value == 1.0:  # Goal
                        color = (200, 50, 50, 255)  # Red

                    # Only configure if the item exists
                    if dpg.does_item_exist(tag):
                        dpg.configure_item(tag, fill=color)

    def render_generator_distribution(self):
        """Renders the generator's goal distribution as a heatmap."""
        if not dpg.is_dearpygui_running(): return

        dist = self.generator_dist_hist
        if dist is None or len(dist) != self.generator.num_possible_goals: return  # Ensure dist is valid

        grid_size = self.config['grid_size']
        draw_size = 300  # Must match setup
        cell_size = draw_size // grid_size

        max_prob = np.max(dist) + 1e-6  # Avoid division by zero

        for r in range(grid_size):
            for c in range(grid_size):
                tag = self.gen_dist_cell_tags.get((r, c))
                if tag:
                    idx = r * grid_size + c
                    prob = dist[idx]
                    intensity = int(255 * (prob / max_prob))
                    color = (0, intensity // 2, intensity, 255)  # Heatmap (Blue intensity)

                    if dpg.does_item_exist(tag):
                        dpg.configure_item(tag, fill=color)

    def training_loop(self):
        """The main training loop running in a separate thread."""
        min_return = self.env.get_min_possible_return()
        max_return = self.env.get_max_possible_return()

        while self.is_running:
            loop_start_time = time.time()

            # 1. Generator generates a task (goal location)
            goal_pos, goal_index, gen_log_prob = self.generator.sample_goal()
            #current_goal_tuple = (goal_pos[0].item(), goal_pos[1].item())  # Make it a plain tuple
            current_goal_tuple = (goal_pos[0], goal_pos[1])  # Make it a plain tuple

            # 2. Solver trains on this task
            task_solver_returns = []
            task_policy_losses = []
            task_value_losses = []

            for _ in range(self.config["solver_updates_per_task"]):
                if not self.is_running: break  # Check if stopped

                # Collect experience
                states, actions, returns, advantages, old_log_probs, mean_ret, std_ret = \
                    self.solver.collect_experience(self.config["solver_episodes_per_update"], current_goal_tuple)

                self.total_episodes += self.config["solver_episodes_per_update"]
                self.total_steps += len(states)
                task_solver_returns.append(mean_ret)  # Store mean return for this batch

                # Update solver
                p_loss, v_loss, _ = self.solver.update(states, actions, returns, advantages, old_log_probs)
                task_policy_losses.append(p_loss)
                task_value_losses.append(v_loss)

            if not self.is_running: break

            # 3. Calculate Generator Reward
            # Use the average performance of the solver *during* training on this specific task
            avg_perf_on_task = np.mean(task_solver_returns) if task_solver_returns else min_return
            self.last_solver_perf_mean = avg_perf_on_task
            # self.last_solver_perf_std = np.mean(batch_std_ret) # Std might also be useful

            gen_reward = self.generator_trainer.get_generator_reward(avg_perf_on_task, min_return, max_return)
            self.last_generator_reward = gen_reward

            # Store result for generator update
            self.generator_trainer.store_task_result(goal_index, gen_log_prob, gen_reward)

            # 4. Update Generator
            gen_loss = self.generator_trainer.update()
            self.total_generator_updates += 1

            # 5. Send data to UI thread periodically
            if self.ui_update_queue.full():
                try:
                    self.ui_update_queue.get_nowait()  # Discard oldest if full
                except queue.Empty:
                    pass

            update_payload = {
                "solver_reward": avg_perf_on_task,
                "policy_loss": np.mean(task_policy_losses) if task_policy_losses else 0,
                "value_loss": np.mean(task_value_losses) if task_value_losses else 0,
                "generator_reward": self.last_generator_reward,
                "render_grid": self.env.get_state_for_render(),  # Get latest grid state
                "current_goal": current_goal_tuple,
                "generator_distribution": self.generator.get_distribution(),
            }
            try:
                self.ui_update_queue.put_nowait(update_payload)
            except queue.Full:
                pass  # Skip update if queue is still full

            self.render_loop()

            # Control loop speed
            loop_time = time.time() - loop_start_time
            sleep_time = (self.config["update_interval_ms"] / 1000.0) - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Training loop stopped.")

    def start_training(self):
        if not self.is_running:
            self.is_running = True
            dpg.configure_item(self.start_button, enabled=False)
            dpg.configure_item(self.stop_button, enabled=True)
            dpg.set_value("status_text", "Status: Running")

            # Clear history for a fresh start if needed? Or keep accumulating? Let's keep accumulating for now.
            # self.reset_stats() # Optional: reset stats on each start

            self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
            self.training_thread.start()
            print("Training started.")

    def stop_training(self):
        if self.is_running:
            self.is_running = False
            dpg.configure_item(self.start_button, enabled=True)
            dpg.configure_item(self.stop_button, enabled=False)
            dpg.set_value("status_text", "Status: Stopping...")
            print("Stop signal sent...")
            # Wait briefly for thread to potentially finish current step
            if self.training_thread and self.training_thread.is_alive():
                # No join needed if daemon=True, but let's signal stop clearly
                pass
            self.training_thread = None
            dpg.set_value("status_text", "Status: Stopped")

    def reset_stats(self):
        """Resets training statistics and history."""
        self.total_steps = 0
        self.total_episodes = 0
        self.total_generator_updates = 0
        self.last_solver_perf_mean = 0.0
        self.last_solver_perf_std = 0.0
        self.last_generator_reward = 0.0

        self.solver.episode_rewards.clear()
        self.solver.policy_losses.clear()
        self.solver.value_losses.clear()
        self.solver.entropy_losses.clear()
        self.solver.memory.clear()

        self.generator_trainer.rewards.clear()
        self.generator_trainer.task_history.clear()

        self.solver_rewards_hist = []
        self.solver_policy_loss_hist = []
        self.solver_value_loss_hist = []
        self.generator_rewards_hist = []
        self.generator_dist_hist = np.zeros(self.generator.num_possible_goals)

        # Clear plots
        dpg.set_value("solver_reward_series", [[], []])
        dpg.set_value("solver_policy_loss_series", [[], []])
        dpg.set_value("solver_value_loss_series", [[], []])
        dpg.set_value("gen_reward_series", [[], []])

        # Update text displays
        update_payload = {  # Send dummy data to reset displays
            "render_grid": self.env.get_state_for_render(),
            "current_goal": "N/A",
            "generator_distribution": self.generator.get_distribution(),
        }
        self.update_ui(update_payload)  # Update UI immediately

    def reset_all(self):
        """Stops training, resets agents, environment, and stats."""
        print("Resetting simulation...")
        self.stop_training()  # Ensure training is stopped

        # Re-initialize agents and optimizer
        self.env = SimpleGridEnv(self.config["grid_size"],
                                 self.config["max_episode_steps"])  # Recreate env in case params changed
        self.solver = SolverTrainer(self.env, self.config)
        self.generator = TaskGenerator(self.config["grid_size"])
        self.generator_trainer = GeneratorTrainer(self.generator, self.config)

        # Reset statistics
        self.reset_stats()

        # Reset visualizations
        self.render_grid = self.env.get_state_for_render()
        self.render_grid_env()
        self.render_generator_distribution()

        print("Reset complete.")

    def run(self):
        """Starts the DPG application."""
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
        # Ensure thread stops if GUI window is closed
        self.is_running = False
        print("Exiting application.")


if __name__ == "__main__":
    demo = AtenaDemo(CONFIG)
    demo.run()