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
        return -100.0 # Penalty for dying
    
    reward = 0.1 # Small survival bonus per frame
    
    # Penalty for staying in the right corner exploit
    if play_state.player and play_state.player.x > c.GAME_SIZE.width - 25:
        reward -= 0.5
    else:
        reward += 0.2

    if play_state.score > prev_score:
        reward += 15.0 # Reward for hitting an enemy
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
        
        # Only fire if the cooldown (200ms) has passed
        if current_time >= (game.last_fire_time + 200):
            game.fighter_shoots() 
            game.last_fire_time = current_time # Update the last fire timestamp
    
    game.update(16, keys)

# --- TRAINING LOOP ---
def train():
    pygame.init()
    screen = pygame.display.set_mode(c.GAME_SIZE)
    pygame.display.set_caption("Galaga AI Training")

    # 1. Setup persistence data
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

    # 2. INITIALIZE MODEL AND OPTIMIZER (Fixes NameError)
    state_size = 2 # [player_x, enemy_dist]
    action_size = 3 # [left, right, shoot]
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Exploration variables
    epsilon = 0.3      
    epsilon_decay = 0.995
    gamma = 0.99       

    print("Starting AI Training...")

    for episode in range(1000):
        game = Play(dummy_persist)
        game.done_starting()
        total_reward = 0
        done = False

        while not done:
            # Keep window from freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    pygame.quit()
                    return

            state = get_state(game)
            old_score = game.score
            
            # Decide action with Epsilon-Greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 3)
            else:
                with torch.no_grad():
                    # 'model' is now defined above
                    action = torch.argmax(model(state)).item()
            
            apply_action(game, action)
            next_state = get_state(game)
            reward = calculate_reward(game, old_score, game.is_player_alive)
            total_reward += reward

            # 4. THE LEARNING STEP
            current_q = model(state)[action]
            with torch.no_grad():
                max_next_q = torch.max(model(next_state)) if game.is_player_alive else 0
                # Wrap in torch.tensor to avoid the 'float' attribute error
                target_q = torch.tensor(reward + (gamma * max_next_q), dtype=torch.float32)

            loss = F.mse_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5. Visual Updates (Fixes black window)
            game.display(screen)
            pygame.display.flip()

            if not game.is_player_alive or game.score > 50000:
                done = True
        
        # Decay epsilon so it gets smarter over time
        epsilon = max(0.01, epsilon * epsilon_decay)
        log_progress(episode, total_reward, game.score)
        
        if episode % 10 == 0:
            print(f"Episode {episode} | Score: {game.score} | Reward: {total_reward:.2f} | Eps: {epsilon:.2f}")

    # Save the trained brain
    torch.save(model.state_dict(), "galaga_dqn.pth")

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_mode(c.GAME_SIZE) 
    train()