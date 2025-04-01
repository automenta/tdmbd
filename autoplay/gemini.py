import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame
import random
from collections import deque
from abc import ABC, abstractmethod

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')

# --- Hyperparameters ---
class Hyperparameters:
    def __init__(self):
        self.solver_lr = 0.0001  # Lowered for stability
        self.generator_lr = 0.0002  # Lowered for stability
        self.train_steps = 5
        self.grid_size = 5
        self.clip_eps = 0.2
        self.max_steps = 25
        self.state_dim = 50
        self.entropy_coef = 0.01

hp = Hyperparameters()

# --- Abstract Task Metamodel ---
class Task(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

# --- Diverse Task Implementations ---
class GridTransformTask(Task):
    def __init__(self, rule):
        self.size = hp.grid_size
        self.rule = rule
        self.input_grid = np.random.randint(0, 3, (self.size, self.size))
        self.target_grid = self._apply_rule(self.input_grid)
        self.current_grid = np.zeros((self.size, self.size))
        self.step_count = 0

    def reset(self):
        self.current_grid = np.zeros((self.size, self.size))
        self.step_count = 0
        return self.get_state()

    def step(self, action):
        x, y, color = action // (self.size * 3), (action // 3) % self.size, action % 3
        self.current_grid[x, y] = color
        self.step_count += 1
        done = self.step_count >= hp.max_steps or np.array_equal(self.current_grid, self.target_grid)
        reward = 1.0 if np.array_equal(self.current_grid, self.target_grid) else -0.05
        return self.get_state(), reward, done

    def get_state(self):
        state = np.concatenate([self.input_grid.flatten(), self.current_grid.flatten()])
        return np.pad(state, (0, hp.state_dim - len(state)), mode='constant') if len(state) < hp.state_dim else state[:hp.state_dim]

    def get_action_space(self):
        return self.size * self.size * 3

    def _apply_rule(self, grid):
        if self.rule[0] == 0:
            return np.roll(grid, self.rule[1], axis=0)
        return np.rot90(grid, self.rule[1])

class PatternCompletionTask(Task):
    def __init__(self, length):
        self.length = length
        self.pattern = [random.randint(0, 2) for _ in range(self.length - 1)]
        self.target = (sum(self.pattern) % 3)
        self.current = 0
        self.step_count = 0

    def reset(self):
        self.current = 0
        self.step_count = 0
        return self.get_state()

    def step(self, action):
        self.current = action
        self.step_count += 1
        done = self.step_count >= 1
        reward = 1.0 if self.current == self.target else -0.1
        return self.get_state(), reward, done

    def get_state(self):
        state = np.array(self.pattern + [self.current])
        return np.pad(state, (0, hp.state_dim - len(state)), mode='constant') if len(state) < hp.state_dim else state[:hp.state_dim]

    def get_action_space(self):
        return 3

class SymbolicReasoningTask(Task):
    def __init__(self):
        self.sequence = [random.randint(1, 5) for _ in range(3)]
        self.target = self.sequence[0] + self.sequence[1] - self.sequence[2]
        self.current = 0
        self.step_count = 0

    def reset(self):
        self.current = 0
        self.step_count = 0
        return self.get_state()

    def step(self, action):
        self.current = action - 5
        self.step_count += 1
        done = self.step_count >= 1
        reward = 1.0 if self.current == self.target else -0.1
        return self.get_state(), reward, done

    def get_state(self):
        state = np.array(self.sequence + [self.current])
        return np.pad(state, (0, hp.state_dim - len(state)), mode='constant') if len(state) < hp.state_dim else state[:hp.state_dim]

    def get_action_space(self):
        return 11

# --- Neural Network Models ---
class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_output_dim):
        super(PolicyNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value

# --- Solver Agent (PPO) ---
class Solver:
    def __init__(self):
        self.device = DEVICE
        self.policy = PolicyNet(hp.state_dim, 64, 75).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=hp.solver_lr)
        self.memory = deque(maxlen=1000)

    def act(self, state, action_space):
        state_tensor = torch.FloatTensor(state).to(self.device)
        probs, _ = self.policy(state_tensor)
        probs = probs[:action_space]
        probs = torch.clamp(probs, min=1e-10, max=1.0 - 1e-10)  # Stricter clamping
        probs = probs / probs.sum()  # Re-normalize
        try:
            action = torch.multinomial(probs, 1).item()
        except RuntimeError:
            action = random.randint(0, action_space - 1)
        return action

    def train(self, task, steps):
        states, actions, rewards, old_probs, values = [], [], [], [], []
        state = task.reset()

        for _ in range(steps):
            episode_states, episode_actions, episode_rewards = [], [], []
            for _ in range(hp.max_steps):
                action = self.act(state, task.get_action_space())
                next_state, reward, done = task.step(action)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                probs, value = self.policy(torch.FloatTensor(state).to(self.device))
                old_probs.append(probs[action].detach())
                values.append(value.item())
                state = next_state
                if done:
                    break
            states.extend(episode_states)
            actions.extend(episode_actions)
            rewards.extend(episode_rewards)
            if not done:
                _, final_value = self.policy(torch.FloatTensor(state).to(self.device))
                rewards.append(final_value.item())

        if not rewards:
            return 0.0, 0.0, 0.0

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(np.array([sum(rewards[i:]) for i in range(len(rewards))])).to(self.device)
        old_probs = torch.stack(old_probs)

        for _ in range(3):
            probs, values = self.policy(states)
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
            ratio = action_probs / (old_probs + 1e-5)
            advantages = returns - values.squeeze().detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - hp.clip_eps, 1 + hp.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns.unsqueeze(1))
            loss = actor_loss + 0.5 * critic_loss + hp.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # Tighter clipping
            self.optimizer.step()

        return np.mean(rewards[:-1]), actor_loss.item(), critic_loss.item()

# --- Generator Agent ---
class Generator:
    def __init__(self):
        self.device = DEVICE
        self.policy = PolicyNet(3, 32, 6).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=hp.generator_lr)

    def generate_task(self):
        state = torch.FloatTensor([hp.grid_size, 3, hp.max_steps]).to(self.device)
        probs, _ = self.policy(state)
        probs = torch.clamp(probs, min=1e-10, max=1.0 - 1e-10)  # Stricter clamping
        probs = probs / probs.sum()  # Re-normalize
        try:
            task_type = torch.multinomial(probs[:3], 1).item()
            param = torch.multinomial(probs[3:], 1).item() % 2 + 1
        except RuntimeError as e:
            # Safe debug output without triggering CUDA assert
            probs_cpu = probs.cpu().detach().numpy()
            print(f"Generator multinomial error: {e}, probs (CPU): {probs_cpu}")
            task_type = random.randint(0, 2)
            param = random.randint(1, 2)
        if task_type == 0:
            return GridTransformTask([param % 2, param])
        elif task_type == 1:
            return PatternCompletionTask(4)
        return SymbolicReasoningTask()

    def update(self, reward):
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device, requires_grad=True)
        probs, _ = self.policy(torch.FloatTensor([hp.grid_size, 3, hp.max_steps]).to(self.device))
        loss = -torch.log(probs.mean() + 1e-10) * reward_tensor
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

# --- Visualization ---
class Visualizer:
    def __init__(self):
        pygame.init()
        self.width, self.height = 1400, 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Chollet's Formalism Maximizer")
        self.font = pygame.font.SysFont("Arial", 20)
        self.cell_size = 40
        self.solver_rewards = deque(maxlen=50)
        self.gen_rewards = deque(maxlen=50)

    def draw_grid(self, grid, x_offset, y_offset):
        for y in range(hp.grid_size):
            for x in range(hp.grid_size):
                rect = pygame.Rect(x_offset + x * self.cell_size, y_offset + y * self.cell_size, self.cell_size, self.cell_size)
                color = (255, 255, 255) if grid[y, x] == 0 else (0, 0, 255) if grid[y, x] == 1 else (255, 0, 0)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

    def draw_stats(self, solver_data, gen_data, x_offset, title):
        reward, actor_loss, critic_loss = solver_data if title == "Solver" else gen_data
        stats = [
            f"{title}",
            f"Reward: {reward:.2f}",
            f"Actor Loss: {actor_loss:.4f}" if title == "Solver" else f"Loss: {actor_loss:.4f}",
            f"Critic Loss: {critic_loss:.4f}" if title == "Solver" else "",
            f"LR: {hp.solver_lr if title == 'Solver' else hp.generator_lr:.5f}",
            f"Steps: {hp.train_steps}"
        ]
        for i, text in enumerate(stats):
            if text:
                render = self.font.render(text, True, (0, 0, 0))
                self.screen.blit(render, (x_offset, 350 + i * 30))

    def draw_plot(self, rewards, x_offset, y_offset, color):
        for i in range(1, len(rewards)):
            pygame.draw.line(self.screen, color,
                            (x_offset + (i-1) * 5, y_offset - rewards[i-1] * 10),
                            (x_offset + i * 5, y_offset - rewards[i] * 10), 2)

    def draw_dynamics(self):
        mid_x = self.width // 2
        pygame.draw.line(self.screen, (0, 0, 0), (mid_x, 0), (mid_x, self.height), 2)
        self.draw_plot(self.solver_rewards, mid_x - 100, self.height // 2, (0, 0, 255))
        self.draw_plot(self.gen_rewards, mid_x + 20, self.height // 2, (255, 0, 0))
        text = self.font.render("Solver (Blue) vs Generator (Red)", True, (0, 0, 0))
        self.screen.blit(text, (mid_x - 100, self.height // 2 + 50))

    def draw_sliders(self):
        sliders = [
            ("Solver LR", hp.solver_lr, 0.0001, 0.001, 50, 500),
            ("Gen LR", hp.generator_lr, 0.0001, 0.001, 50, 550),
            ("Steps", hp.train_steps, 1, 10, 50, 600)
        ]
        for name, value, min_v, max_v, x, y in sliders:
            pygame.draw.rect(self.screen, (150, 150, 150), (x, y, 200, 20))
            pos = x + (value - min_v) / (max_v - min_v) * 200
            pygame.draw.circle(self.screen, (0, 0, 0), (int(pos), y + 10), 10)
            text = self.font.render(f"{name}: {value}", True, (0, 0, 0))
            self.screen.blit(text, (x, y - 20))

    def update(self, task, solver_data, gen_data):
        self.screen.fill((200, 200, 200))
        if isinstance(task, GridTransformTask):
            self.draw_grid(task.input_grid, 50, 50)
            self.draw_grid(task.current_grid, 300, 50)
            self.draw_grid(task.target_grid, 650, 50)
        self.draw_stats(solver_data, gen_data, 50, "Solver")
        self.draw_stats(solver_data, gen_data, 650, "Generator")
        self.solver_rewards.append(solver_data[0])
        self.gen_rewards.append(gen_data[0])
        self.draw_dynamics()
        self.draw_sliders()
        pygame.display.flip()

# --- Main Loop ---
def main():
    solver = Solver()
    generator = Generator()
    visualizer = Visualizer()

    validation_tasks = [GridTransformTask([0, 1]), PatternCompletionTask(4), SymbolicReasoningTask()]

    running = True
    dragging = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if 50 <= x <= 250 and 500 <= y <= 520:
                    dragging = "solver_lr"
                elif 50 <= x <= 250 and 550 <= y <= 570:
                    dragging = "generator_lr"
                elif 50 <= x <= 250 and 600 <= y <= 620:
                    dragging = "train_steps"
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                x = event.pos[0]
                if dragging == "solver_lr":
                    hp.solver_lr = max(0.0001, min(0.001, 0.0001 + (x - 50) / 200 * 0.0009))
                elif dragging == "generator_lr":
                    hp.generator_lr = max(0.0001, min(0.001, 0.0001 + (x - 50) / 200 * 0.0009))
                elif dragging == "train_steps":
                    hp.train_steps = max(1, min(10, int(1 + (x - 50) / 200 * 9)))

        task = generator.generate_task()
        pre_reward = np.mean([solver.train(t, 1)[0] for t in validation_tasks])
        solver_reward, actor_loss, critic_loss = solver.train(task, hp.train_steps)
        post_reward = np.mean([solver.train(t, 1)[0] for t in validation_tasks])
        gen_reward = post_reward - pre_reward
        gen_loss = generator.update(gen_reward)

        solver_data = (solver_reward, actor_loss, critic_loss)
        gen_data = (gen_reward, gen_loss, 0)
        visualizer.update(task, solver_data, gen_data)

        pygame.time.wait(50)

    pygame.quit()

if __name__ == "__main__":
    print("Note: For better Pygame performance, set PYGAME_DETECT_AVX2=1 during installation if your CPU supports AVX2.")
    print("Running with CUDA_LAUNCH_BLOCKING=1 for precise error tracing is recommended.")
    main()