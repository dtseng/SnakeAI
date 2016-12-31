from graphics import *
from enum import Enum
from collections import deque
from copy import deepcopy
import random

UP, DOWN, LEFT, RIGHT = range(4)


class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Game:
    def __init__(self, window_size=200, game_size=15):
        # win = GraphWin(width=window_size, height=window_size)
        # win.setCoords(-1, -1, game_size, game_size)
        # win.setBackground("black")
        # point_width = float(window_size)/(game_size + 1)
        # walls = Rectangle(Point(0, 0), Point(game_size - 1, game_size - 1))
        # walls.setWidth(point_width)
        # walls.setOutline("white")
        # walls.draw(win)
        #
        # self.win = win
        self.boundary = game_size - 2
        # self.pointWidth = point_width
        # self.walls = walls
        self.snake = Snake((int(game_size / 2), int(game_size / 2)))
        self.score = 0
        self.continue_game = True
        self.food = (5, int(game_size/2))

    def step(self):
        """Updates the next food position, snake position, and the graphics for a single time step."""
        inputs = 1  # TODO: Change this
        direction = self.snake.action(inputs)
        snake_head_copy = deepcopy(self.snake.head)
        self.snake.body.appendleft(snake_head_copy)

        # Move snake accordingly
        if direction == Direction.UP:
            self.snake.head = (snake_head_copy[0], snake_head_copy[1] + 1)
        elif direction == Direction.DOWN:
            self.snake.head = (snake_head_copy[0], snake_head_copy[1] - 1)
        elif direction == Direction.LEFT:
            self.snake.head = (snake_head_copy[0] - 1, snake_head_copy[1])
        elif direction == Direction.RIGHT:
            self.snake.head = (snake_head_copy[0] + 1, snake_head_copy[1] + 1)

        if self.snake.head == self.food:  # Update snake if it ate the food
            self.snake.fitness += 1
            self.snake.head = deepcopy(self.food)
            self.regenerate_food()
        else:
            self.snake.body.pop()
        if self.snake.head in self.snake.body or self._coordinate_is_wall(self.snake.head):  # If snake died
            self.continue_game = False

    def _coordinate_is_wall(self, coordinate):
        return coordinate[0] == self.boundary + 1 or coordinate[0] == 0 or \
               coordinate[1] == self.boundary + 1 or coordinate[1] == 0

    def regenerate_food(self): # TODO: Might want to speed this up
        if self.boundary**2 == len(self.snake.body) + 1:
            self.continue_game = False
            return
        else:
            while True:
                food_x = random.randint(1, self.boundary)
                food_y = random.randint(1, self.boundary)
                if (food_x, food_y) != self.snake.head and (food_x, food_y) not in self.snake.body:
                    self.food = (food_x, food_y)
                    break

    def get_score(self):
        return self.score

    def print_status(self):
        print("status:", self.continue_game, "food:", self.food, ".. head:", self.snake.head, "| body:", end='')
        for x in self.snake.body:
            print(x, end='')
        print()

    def get_continue_status(self):
        """Returns false if the snake has hit a wall or itself."""
        return self.continue_game


class Snake:
    def __init__(self, head_coordinate):
        self.fitness = 0
        self.head = head_coordinate
        self.body = deque()
        self.body.append((head_coordinate[0] + 1, head_coordinate[1]))
        self.body.append((head_coordinate[0] + 2, head_coordinate[1]))

    def action(self, inputs):  # Returns LEFT, RIGHT, UP, or DOWN.
        """Returns LEFT, RIGHT, UP, or DOWN. Outputs from neural network."""
        return Direction.LEFT  # TODO: Neural network output here

game = Game()
game.print_status()

game.step()
game.print_status()

game.step()
game.print_status()