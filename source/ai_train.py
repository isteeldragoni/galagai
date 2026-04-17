import os
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .play import Play
from .stars import StarField
from . import constants as c

# --- JOSEPH: DQN Model & Replay Memory ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- BRENNAN: State Representation ---
def get_state(play_state):
    # Normalize Player X (0.0 to 1.0)
    p_x = play_state.player.x / c.GAME_SIZE.width if play_state.player else 0.5

    # Track nearest enemy distance
    enemy_dist = 1.0
    if play_state.enemies:
        nearest = min(play_state.enemies, key=lambda e: e.y)
        enemy_dist = nearest.y / c.GAME_SIZE.height

    return torch.FloatTensor([p_x, enemy_dist])

# --- NATHAN: Reward and Logging ---
def calculate_reward(play_state, prev_score, is_alive):
    if not is_alive:
        return -100.0 # Penalty for dying

    reward = 0.1 # Small survival bonus per frame

    # Penalty for staying in the right corner exploit
    if play_state.player and play_state.player.x > c.GAME_SIZE.width - 25:
        reward -= 0.2

    if play_state.score > prev_score:
        reward += 10.0 # Reward for hitting an enemy
    return reward

def log_progress(episode, total_reward, score):
    log_path = "training_log.csv"
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("Episode,TotalReward,FinalScore\n")
    with open(log_path, "a") as f:
        f.write(f"{episode},{total_reward:.2f},{score}\n")

# --- INTERFACE: Connecting AI to Pygame ---
def apply_action(game, action):
    # Action 0: Left, 1: Right, 2: Shoot
    keys = {pygame.K_LEFT: False, pygame.K_RIGHT: False, pygame.K_SPACE: False}
    if action == 0:
        keys[pygame.K_LEFT] = True
    elif action == 1:
        keys[pygame.K_RIGHT] = True
    elif action == 2:
        keys[pygame.K_SPACE] = True
        current_time = pygame.time.get_ticks()

        # Only fire if the cooldown (200ms) has passed
        if current_time >= (game.last_fire_time + 200):
            game.fighter_shoots()
            game.last_fire_time = current_time 

    game.update(16, keys)

# --- TRAINING LOOP ---
def train():
    pygame.init()
    screen = pygame.display.set_mode(c.GAME_SIZE)
    pygame.display.set_caption("Galaga AI Training")

    # 1. Hyperparameters & Setup
    state_size = 2 # [player_x, enemy_dist]
    action_size = 3 # [left, right, shoot]
    batch_size = 64
    gamma = 0.99
    target_update_freq = 10 # Update target net every 10 episodes
    
    # 2. Initialize Models
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # Target net is not trained directly
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = ReplayMemory(10000)

    # 3. Exploration variables
    epsilon = 0.3
    epsilon_decay = 0.995
    epsilon_min = 0.01

    # Persistence data
    dummy_persist = c.Persist(
        stars=StarField(),
        scores=[],
        current_score=0,
        one_up_score=0,
        high_score=10000,
        num_shots=0,
        num_hits=0,
        stage_num=1
    )

    print("Starting AI Training...")

    for episode in range(1000):
        game = Play(dummy_persist)
        game.done_starting()
        total_reward = 0
        done = False
        state = get_state(game)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Decide action with Epsilon-Greedy
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    action = policy_net(state.unsqueeze(0)).argmax().item()
            
            old_score = game.score
            apply_action(game, action)
            
            next_state = get_state(game)
            reward = calculate_reward(game, old_score, game.is_player_alive)
            is_terminal = not game.is_player_alive
            
            # Store transition in Replay Memory
            memory.push(state, action, reward, next_state, is_terminal)
            
            state = next_state
            total_reward += reward

            # 4. THE LEARNING STEP (Batch Training)
            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                # Transpose the batch (see https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.stack(batch_state)
                batch_action = torch.tensor(batch_action).unsqueeze(1)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
                batch_next_state = torch.stack(batch_next_state)
                batch_done = torch.tensor(batch_done, dtype=torch.float32)

                # Get current Q-values for actions taken
                current_q = policy_net(batch_state).gather(1, batch_action)

                # Compute the expected Q-values from target network
                with torch.no_grad():
                    max_next_q = target_net(batch_next_state).max(1)[0]
                    target_q = batch_reward + (gamma * max_next_q * (1 - batch_done))

                loss = F.mse_loss(current_q.squeeze(), target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Visual Updates
            game.display(screen)
            pygame.display.flip()
            
            if not game.is_player_alive or game.score > 50000:
                done = True

        # Sync Target Network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        log_progress(episode, total_reward, game.score)

        if episode % 10 == 0:
            print(f"Episode {episode} | Score: {game.score} | Reward: {total_reward:.2f} | Eps: {epsilon:.2f}")

    # Save the trained brain
    torch.save(policy_net.state_dict(), "galaga_dqn.pth")

if __name__ == "__main__":
    train()
