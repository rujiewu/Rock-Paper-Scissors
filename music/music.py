import pygame


class Music(object):
    def __init__(self):
        pygame.init()
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.mixer.init()

    @staticmethod
    def play(music=0):
        if music == 0:
            pygame.mixer.music.load("begin.mp3")
        elif music == 1:
            pygame.mixer.music.load("win.mp3")
        elif music == 2:
            pygame.mixer.music.load("lose.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
