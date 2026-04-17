import sys
import pygame
from source import main, ai_train

if __name__ == '__main__':
    TRAIN_AI = False 

    if TRAIN_AI:
        ai_train.train()
    else:
        main.main()
        
    pygame.quit()
    sys.exit()