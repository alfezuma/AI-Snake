from tkinter.tix import TEXT
import pygame
import numpy as np
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('Montserrat-Medium.ttf', 21)
#font = pygame.font.SysFont('arial', 21)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# Colours Used
TEXT_COLOUR = (241, 250, 238)
FOOD_COLOUR = (230, 57, 70)
SNAKE_HEAD_COLOUR = (168, 218, 220)
SNAKE_BODY_COLOUR = (69, 123, 157)
BACKGROUND_COLOUR = (32, 49, 73)
BACKGROUND_COLOUR_GAME_OVER = (97, 34, 34)
BLACK = (16, 16, 16)

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

BLOCK_SIZE = 16
SPEED = 40

clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

class SnakeGameAI:

    
    
    def __init__(self, w=SCREEN_WIDTH, h=SCREEN_HEIGHT):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset(0,0)
        
    def reset(self, game_count, high_score):
        # init game state
        self.game_count = game_count
        self.high_score = high_score
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        reward = 0
        self.frame_iteration += 1 
        # 1. collect user input - no longer required for AI to play
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False 
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            self._update_ui(game_over)
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui(game_over)
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self, game_over):
        if game_over:
            self.display.fill(BACKGROUND_COLOUR_GAME_OVER)
        else:
            self.display.fill(BACKGROUND_COLOUR)
        
        if game_over:
            #Draw snake head
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.snake[0].x+1, self.snake[0].y+1, BLOCK_SIZE - 2, BLOCK_SIZE - 2), border_radius=2)
            #Draw rest of the snake
            for pt in self.snake[1:]:
               pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x+1, pt.y+1, BLOCK_SIZE - 2, BLOCK_SIZE - 2), border_radius=2)
            #Draw food
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE - 2, BLOCK_SIZE - 2), border_radius=2)
        else:
            #Draw snake head
            pygame.draw.rect(self.display, SNAKE_HEAD_COLOUR, pygame.Rect(self.snake[0].x+1, self.snake[0].y+1, BLOCK_SIZE - 2, BLOCK_SIZE - 2), border_radius=2)
            #Draw rest of the snake
            for pt in self.snake[1:]:
               pygame.draw.rect(self.display, SNAKE_BODY_COLOUR, pygame.Rect(pt.x+1, pt.y+1, BLOCK_SIZE - 2, BLOCK_SIZE - 2), border_radius=2)
            #Draw food
            pygame.draw.rect(self.display, FOOD_COLOUR, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE - 2, BLOCK_SIZE - 2), border_radius=2)

        
        text = font.render("Game: " + str(self.game_count + 1) + "    Score: " + str(self.score) + "    High Score: " + str(self.high_score), True, TEXT_COLOUR)
        self.display.blit(text, [0, 0])

        if game_over:
            game_over_text = font.render("Game Over", True, TEXT_COLOUR)
            game_over_text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)) # Used to find centre point of text so that it can be displayed in the middle of the screen
            self.display.blit(game_over_text, game_over_text_rect)

        pygame.display.flip()
        
    def _move(self, action):
        # [Straight, Right, Left]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1,0,0]):
            self.direction = clockwise[idx] # no change
        elif np.array_equal(action, [0,1,0]):
            self.direction = clockwise[(idx+1) % 4] # right turn
        else:
            self.direction = clockwise[(idx-1) % 4] # left turn

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)