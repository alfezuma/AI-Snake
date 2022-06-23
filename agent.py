import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from plot import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.01

class Agent:
    def __init__(self):
        self.n_games = 0 # Keep track of the number of games
        self.epsilon = 0 # Parameter to control the randomness
        self.gamma = 0.7 # Discount rate, must be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY) # If we exceed max memory then it will automatically remove elements from the left
        self.model = Linear_QNet(11, 512, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        # Points immediately next to the head
        point_L = Point(head.x - BLOCK_SIZE, head.y)
        point_R = Point(head.x + BLOCK_SIZE, head.y)
        point_U = Point(head.x, head.y - BLOCK_SIZE)
        point_D = Point(head.x, head.y + BLOCK_SIZE)

        # Boolean values to check if current direction is equal to left, right, up or down
        dir_L = game.direction == Direction.LEFT
        dir_R = game.direction == Direction.RIGHT
        dir_U = game.direction == Direction.UP
        dir_D = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_L and game.is_collision(point_L)) or
            (dir_R and game.is_collision(point_R)) or
            (dir_U and game.is_collision(point_U)) or
            (dir_D and game.is_collision(point_D)),

            # Danger right
            (dir_L and game.is_collision(point_U)) or
            (dir_R and game.is_collision(point_D)) or
            (dir_U and game.is_collision(point_R)) or
            (dir_D and game.is_collision(point_L)),

            # Danger left
            (dir_L and game.is_collision(point_D)) or
            (dir_R and game.is_collision(point_U)) or
            (dir_U and game.is_collision(point_L)) or
            (dir_D and game.is_collision(point_R)),

            # Move direction
            dir_L,
            dir_R,
            dir_U,
            dir_D,

            # Food location
            game.food.x < game.head.x, # Food left
            game.food.x > game.head.x, # Food right
            game.food.y < game.head.y, # Food up
            game.food.y > game.head.y, # Food down
        ]
        
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if max memory is exceeded

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            mini_sample = self.memory
      
        states, actions, rewards, next_states, dones = zip(*mini_sample) #Extract values from tuples into respective variables
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves:
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2) # Get a random value for the index
            final_move[move] = 1 # Set the final_move to be in a random direction
        else:
            state0 = torch.tensor(state, dtype=torch.float) 
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # Get the index for the largest number in the prediction 
            final_move[move] = 1 # Set the final_move to be in the direction based on that index

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    high_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Run training code
        # Get old state
        state_old = agent.get_state(game)

        # Get move based on current state
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory 
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory and plot result
            agent.n_games += 1
            game.reset(agent.n_games, high_score)
            agent.train_long_memory()

            if score > high_score:
                high_score = score
                agent.model.save()

            print('Game', agent.n_games, 'Score:', score, 'Record:', high_score) 
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)

if __name__ == '__main__':
    train()

