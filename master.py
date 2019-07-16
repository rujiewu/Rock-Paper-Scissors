import uuid
from excpetion import RPSException
from game import Game


class Master(object):
    def __init__(self):
        self.id = uuid.uuid1()
        self.game = None

    def create_game(self):
        self.game = Game()
        self.game.music_player.play()
        self.game.add_player(is_bot=True)
        self.game.add_player(True)

    def add_player(self, is_bot=False):
        self.game.add_player(is_bot)

    def destroy_game(self):
        self.game = None


def crate_game():
    game_master = Master()
    game_master.create_game()
    game_master.game.judge()


crate_game()
