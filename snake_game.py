from graphics import *

UP, DOWN, LEFT, RIGHT = range(4)


class Game:
    def __init__(self, window_size=200, game_size=50):
        win = GraphWin(width=window_size, height=window_size)
        game_size = 50
        win.setCoords(-1, -1, game_size, game_size)
        win.setBackground("black")
        point_width = float(window_size)/(game_size + 1)
        walls = Rectangle(Point(0, 0), Point(game_size - 1, game_size - 1))
        walls.setWidth(point_width)
        walls.setOutline("white")
        walls.draw(win)
        self.win = win
        self.pointWidth = point_width
        self.walls = walls
        self.snake = Snake((game_size / 2, game_size / 2))


class Snake:
    def __init__(self, head_coordinate):
        self.body = [(head_coordinate[0] + 1, head_coordinate[1]), (head_coordinate[0] + 2, head_coordinate[1])]

    def action(self, inputs):  # Returns LEFT, RIGHT, UP, or DOWN.
        return RIGHT