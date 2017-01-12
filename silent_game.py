from graphics import *
from enum import Enum
from collections import deque
from copy import deepcopy
import random

UP, DOWN, LEFT, RIGHT = range(4)


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Game:
    def __init__(self, neural_net, window_size=200, game_size=15):
        # win = GraphWin(width=window_size, height=window_size)
        # self.win = win
        self.win = None
        # win.setCoords(-1, -1, game_size, game_size)
        # win.setBackground("black")
        # self.point_width = float(window_size)/(game_size + 1)
        walls = Rectangle(Point(0, 0), Point(game_size - 1, game_size - 1))
        p1 = (0, 0)
        p2 = (game_size - 1, game_size - 1)
        # self.draw_line(p1, p2, "grey")

        self.wall_boundary = game_size - 1
        # self.pointWidth = self.point_width
        self.walls = walls
        self.snake = Snake((int(game_size / 2), int(game_size / 2)), neural_net)
        self.score = 0
        self.continue_game = True
        self.food = (5, int(game_size/2))
        random.seed(2)
        # self.snake_graphic = deque()

        # self.food_graphic = self.draw_line(self.food, self.food, "red")
        # unit = self.draw_line(self.snake.head, self.snake.body[0], "white")
        # self.snake_graphic.append(unit)
        # unit2 = self.draw_line(self.snake.body[0], self.snake.body[1], "white")
        # self.snake_graphic.append(unit2)

    def draw_line(self, p1, p2, color):
        """Draws line, and returns the Rectangle object. point1 and point2 are tuples."""
        point1 = Point(p1[0], p1[1])
        point2 = Point(p2[0], p2[1])
        rect = Rectangle(point1, point2)
        rect.setWidth(self.point_width)
        rect.setOutline(color)
        rect.draw(self.win)
        return rect

    def step(self):
        """Updates the next food position, snake position, and the graphics for a single time step."""
        inputs = self.retrieve_nn_inputs()
        direction = self.snake.action(inputs)
        snake_head_copy = deepcopy(self.snake.head)
        self.snake.body.appendleft(snake_head_copy)

        # Move snake accordingly.
        if direction == Direction.UP:
            self.snake.head = (snake_head_copy[0], snake_head_copy[1] + 1)
        elif direction == Direction.DOWN:
            self.snake.head = (snake_head_copy[0], snake_head_copy[1] - 1)
        elif direction == Direction.LEFT:
            self.snake.head = (snake_head_copy[0] - 1, snake_head_copy[1])
        elif direction == Direction.RIGHT:
            self.snake.head = (snake_head_copy[0] + 1, snake_head_copy[1])

        # unit = self.draw_line(snake_head_copy, deepcopy(self.snake.head), "white")
        # self.snake_graphic.appendleft(unit)

        if self.snake.head == self.food:  # Update snake if it ate the food
            self.snake.fitness += 1
            self.snake.head = deepcopy(self.food)
            self.regenerate_food()
        else:
            self.snake.body.pop()
            # end = self.snake_graphic.pop()
            # end.undraw()
            # self.win.setBackground("black")
        if self.snake.head in self.snake.body or self._coordinate_is_wall(self.snake.head):  # If snake died
            self.continue_game = False

    def _coordinate_is_wall(self, coordinate):
        return coordinate[0] == self.wall_boundary or coordinate[0] == 0 or \
               coordinate[1] == self.wall_boundary or coordinate[1] == 0

    def regenerate_food(self):
        # self.food_graphic.undraw()
        # self.win.setBackground("black")
        if (self.wall_boundary-1)**2 == len(self.snake.body) + 1:
            self.continue_game = False
            return
        else:
            while True:
                food_x = random.randint(1, self.wall_boundary-1)
                food_y = random.randint(1, self.wall_boundary-1)
                if (food_x, food_y) != self.snake.head and (food_x, food_y) not in self.snake.body:
                    self.food = (food_x, food_y)
                    # self.food_graphic = self.draw_line(self.food, self.food, "red")
                    break

    def retrieve_nn_inputs(self):
        """NN Input:
        0) x distance to apple: food.x - head.x
        1) y distance to apple: food.y - head.y
        2) up: vertical dist to wall or self
        3) right: horizontal dist to wall or self
        4) left: horizontal dist to wall or self
        5) down: vertical dist to wall or self
        6) length of the snake """
        nn_input = []
        head = self.snake.head
        nn_input.append(self.food[0] - head[0])
        nn_input.append(self.food[1] - head[1])
        for i in range(head[1] + 1, self.wall_boundary + 1):
            test_pt = (head[0], i)
            if self._coordinate_is_wall(test_pt) or test_pt in self.snake.body:
                nn_input.append(i - head[1])
                break
        for i in range(head[0] + 1, self.wall_boundary + 1):
            test_pt = (i, head[1])
            if self._coordinate_is_wall(test_pt) or test_pt in self.snake.body:
                nn_input.append(i - head[0])
                break
        for i in range(head[0] - 1, -1, -1):
            test_pt = (i, head[1])
            if self._coordinate_is_wall(test_pt) or test_pt in self.snake.body:
                nn_input.append(head[0] - i)
                break
        for i in range(head[1] - 1, -1, -1):
            test_pt = (head[0], i)
            if self._coordinate_is_wall(test_pt) or test_pt in self.snake.body:
                nn_input.append(head[1] - i)
                break
        nn_input.append(self.snake.length())
        return nn_input

    def get_score(self):
        return self.score

    # def print_status(self):
    #     print("status:", self.continue_game, "food:", self.food, ".. head:", self.snake.head, "| body:", end='')
    #     for x in self.snake.body:
    #         print(x, end='')
    #     print()
    #     print(self.retrieve_nn_inputs())

    def get_continue_status(self):
        """Returns false if the snake has hit a wall or itself."""
        return self.continue_game

    def kill(self):
        self.win.close()

    def manhattan_distance_to_food(self):
        return abs(self.food[0] - self.snake.head[0]) + abs(self.food[1] - self.snake.head[1])


class Snake:
    def __init__(self, head_coordinate, neural_net):
        self.fitness = 0
        self.fn = neural_net
        self.head = head_coordinate
        self.body = deque()
        self.body.append((head_coordinate[0] + 1, head_coordinate[1]))
        self.body.append((head_coordinate[0] + 2, head_coordinate[1]))

    def action(self, inputs):  # Returns LEFT, RIGHT, UP, or DOWN.
        """Returns LEFT, RIGHT, UP, or DOWN. Outputs from neural network."""
        nn_output = self.fn(inputs)
        best_direction = nn_output.index(max(nn_output))
        return Direction(best_direction)

    def length(self):
        return 1 + len(self.body)