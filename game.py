import uuid
from enum import Enum
from excpetion import *
from music import music


class PlayerType(Enum):
    human = 1
    bot = 0


class Bot(object):
    def __init__(self):
        self.classifier = 0


class Player(object):
    def __init__(self, game, is_bot=False):
        self.id = uuid.uuid1()
        self.game = game
        if is_bot:
            self.type = PlayerType.bot
            self.controller = Bot()
        else:
            self.type = PlayerType.human


class Game(object):
    def __init__(self):
        self.id = uuid.uuid1()
        self.score_one = 0
        self.score_two = 0
        self.player_one = 0
        self.player_two = 0
        self.music_player = music.Music()

    def add_player(self, is_bot=False):
        if self.player_one is 0:
            self.player_one = Player(self, is_bot)
        elif self.player_two is 0:
            self.player_two = Player(self, is_bot)
        else:
            raise RPSException(300, 'Unable to create new player')

    def score(self, player):
        if player is self.player_one:
            self.score_one += 1
        elif player is self.player_two:
            self.score_two += 1
        else:
            raise RPSException(400, 'Player wining is illegal')
