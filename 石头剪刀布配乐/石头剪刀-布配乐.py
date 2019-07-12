import pygame


def music(which_music=0):
    """
    foo是其他模块返回的值
    返回值为0是表示游戏准备阶段
    返回值为1表示玩家猜拳胜利
    返回值为2表示玩家猜拳失败
    """
    pygame.mixer.init()
    if which_music == 0:
        track = pygame.mixer.music.load("begin.mp3")
        pygame.mixer.music.play()

    if which_music == 1:
        track1 = pygame.mixer.music.load("win.mp3")
        pygame.mixer.music.play()

    if which_music == 2:
        track2 = pygame.mixer.Sound("lose.mp3")
        track2.play()


music()