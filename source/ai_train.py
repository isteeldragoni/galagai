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
def get_state(play_state, prev_x=None):
    if not play_state.player:
        return torch.FloatTensor([0.5, 0.0, 1.0, 0.0])

    p_x = play_state.player.x / c.GAME_SIZE.width
    
    # 1. Movement: Did we actually move since last time? 
    # (normalized roughly -1.0 to 1.0)
    movement = (play_state.player.x - prev_x) / 10.0 if prev_x is not None else 0.0
    
    # 2. Wall Closeness: 0.0 at center, 1.0 at absolute edges
    wall_closeness = abs(p_x - 0.5) * 2.0

    enemy_dist_y = 1.0
    rel_x = 0.0
    
    if play_state.enemies:
        nearest = min(play_state.enemies, key=lambda e: e.y)
        enemy_dist_y = nearest.y / c.GAME_SIZE.height
        rel_x = (nearest.x - play_state.player.x) / c.GAME_SIZE.width

    # state_size = 4
    return torch.FloatTensor([movement, wall_closeness, enemy_dist_y, rel_x])

# --- NATHAN: Reward and Logging ---
def calculate_reward(play_state, prev_score, is_alive, edge_timer, dist_traveled):
    if not is_alive:
        return -5.0

    reward = 0.0
    p_x = play_state.player.x if play_state.player else c.GAME_SIZE.width / 2

    # --- THE "MOVE OR DIE" RULE ---
    # If it has been more than 3 seconds (180 frames) and 
    # it hasn't traveled at least 20% of the screen width...
    if edge_timer > 180 and dist_traveled < (c.GAME_SIZE.width * 0.2):
        reward -= 2.0  # Heavy penalty for staying in one "sector"

    # --- CONDITIONAL SURVIVAL ---
    # Only reward staying alive if NOT hugging the wall
    if 60 < p_x < c.GAME_SIZE.width - 60:
        reward += 0.5
    else:
        # Escalating penalty for staying at the edge
        # After 60 frames (1 second), this starts hurting bad
        penalty_multiplier = max(0, (edge_timer - 60) / 60.0)
        reward -= (0.3 * penalty_multiplier)

    # --- TRACKING REWARD ---
    if play_state.enemies:
        nearest = min(play_state.enemies, key=lambda e: e.y)
        dist_x = abs(p_x - nearest.x) / c.GAME_SIZE.width
        reward += (1.0 - dist_x) * 0.5

    # --- THE BIG PAYDAY ---
    if play_state.score > prev_score:
        reward += 25.0
        
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
        # Use the correct method name from play.py 
    
        current_time = pygame.time.get_ticks() 
        
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
    state_size = 4 # [movement, wall_closeness, enemy_dist_y, rel_x]
    action_size = 3 
    batch_size = 64
    gamma = 0.99
    target_update_freq = 10 
    
    # 2. Initialize Models
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() 
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005) # Slightly lower LR for stability
    memory = ReplayMemory(10000)

    # 3. Exploration variables
    epsilon = 1.0
    epsilon_decay = 0.998 # Slow decay to force curiosity
    epsilon_min = 0.1

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
        edge_timer = 0 # <--- NEW: Track how long we are at the edge
        start_x_pos = game.player.x if game.player else 0
        done = False
        
        # Initial prev_x for the very first state
        
        state = get_state(game, 0)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Decide action
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    action = policy_net(state.unsqueeze(0)).argmax().item()
            
            # --- TRACKING BEFORE ACTION ---
            dist_traveled = 0
            old_score = game.score
            prev_x = game.player.x if game.player else 0
            
            apply_action(game, action)
            
            # --- UPDATE EDGE TIMER ---
            # If player is within 50 pixels of either side
            if game.player and (game.player.x < 100 or game.player.x > c.GAME_SIZE.width - 100):
                edge_timer += 1
                dist_traveled = abs(game.player.x - start_x_pos)
            else:
                edge_timer = 0 # Reset if they move to the middle
                star_x_pos = game.player.x if game.player else 0
                dist_traveld = 0
            
            # --- GET NEXT STATE & REWARD ---
            next_state = get_state(game, prev_x) # <--- Pass prev_x here
            reward = calculate_reward(game, old_score, game.is_player_alive, edge_timer, dist_traveled) 
            
            is_terminal = not game.is_player_alive
            memory.push(state, action, reward, next_state, is_terminal)
            
            state = next_state
            total_reward += reward

            # 4. THE LEARNING STEP
            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.stack(batch_state)
                batch_action = torch.tensor(batch_action).unsqueeze(1)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
                batch_next_state = torch.stack(batch_next_state)
                batch_done = torch.tensor(batch_done, dtype=torch.float32)

                current_q = policy_net(batch_state).gather(1, batch_action)

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

    torch.save(policy_net.state_dict(), "galaga_dqn.pth")

if __name__ == "__main__":
    train()
