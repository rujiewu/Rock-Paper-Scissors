import uuid
from excpetion import RPSException
from game import Game


class Master(object):
    def __init__(self):
        self.id = uuid.uuid1()
        self.game = 0

    def create_game(self):
        self.game = Game()

    def add_player(self, is_bot):
        self.game.add_player(is_bot)

    def destroy_game(self):
        self.game = 0
