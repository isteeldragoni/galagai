import os
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from .play import Play
from .stars import StarField
from . import constants as c

# --- JOSEPH: DQN Model  ---
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
        return -100.0  # Death penalty
    
    reward = 0.1       # Survival bonus
    if play_state.score > prev_score:
        reward += 10.0 # Hit reward
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
    # Map 0: Left, 1: Right, 2: Shoot
    keys = {pygame.K_LEFT: False, pygame.K_RIGHT: False, pygame.K_SPACE: False}
    
    if action == 0: keys[pygame.K_LEFT] = True
    elif action == 1: keys[pygame.K_RIGHT] = True
    elif action == 2: 
        keys[pygame.K_SPACE] = True
        game.fighter_shoots() # Explicitly call the shoot method
        
    # Step the game logic forward by 1 frame (33ms)
    game.update(33, keys)

# --- TRAINING LOOP ---
def train():
    # Setup the persistence data the Play state requires
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

    state_size = 2 # [player_x, enemy_dist]
    action_size = 3 # [left, right, shoot]
    
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting AI Training...")

    for episode in range(1000):
        game = Play(dummy_persist)
        game.done_starting() # Skip the "Ready" timers
        
        total_reward = 0
        done = False

        while not done:
            # 1. Capture current state and score
            old_score = game.score
            state = get_state(game)
            
            # 2. AI decides action
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()
            
            # 3. Apply action to the game
            apply_action(game, action)
            
            # 4. Calculate reward
            reward = calculate_reward(game, old_score, game.is_player_alive)
            total_reward += reward

            # 5. Check for game over
            if not game.is_player_alive or game.score > 50000:
                done = True
        
        log_progress(episode, total_reward, game.score)
        if episode % 10 == 0:
            print(f"Episode {episode} | Score: {game.score} | Reward: {total_reward:.2f}")

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_mode(c.GAME_SIZE) 
    train()