from .tools import time_millis
import pygame
from . import constants as c, tools
from .constants import Rectangle
from .tools import grab_sheet


class GalagaSprite(pygame.sprite.Sprite):
    """
    Base class for a general sprite in Galaga.
    Updated to support sub-pixel movement using float storage.
    """

    def __init__(self, x, y, width, height, *groups: pygame.sprite.Group):
        super(GalagaSprite, self).__init__(groups)

        # Internal float storage for smooth movement
        self._x = float(x)
        self._y = float(y)
        
        # Rectangle used for collision and integer blitting
        self.rect = pygame.Rect(0, 0, width, height)
        self.rect.center = (int(self._x), int(self._y))

        # Display and image variables
        self.image = None
        self.image_offset_x: int = 0
        self.image_offset_y: int = 0
        self.is_visible: bool = True
        self.flip_horizontal: bool = False
        self.flip_vertical: bool = False

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = float(value)
        self.rect.centerx = int(self._x)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = float(value)
        self.rect.centery = int(self._y)

    def update(self, delta_time: int, flash_flag: bool):
        pass

    def display(self, surface: pygame.Surface):
        if self.image is not None and self.is_visible:
            image = pygame.transform.flip(self.image, self.flip_horizontal, self.flip_vertical)
            img_width, img_height = image.get_size()
            # Calculate blit position based on the float center
            draw_x = int(self._x) - img_width // 2 + self.image_offset_x
            draw_y = int(self._y) - img_height // 2 + self.image_offset_y
            surface.blit(image, (draw_x, draw_y))


class Player(GalagaSprite):

    def __init__(self, x, y):
        super(Player, self).__init__(x, y, 14, 12)
        self.image = grab_sheet(6 * 16, 0 * 16, 16, 16)
        self.image_offset_x = 1

    def update(self, delta_time, keys):
        # Scale movement by delta_time for consistency
        s = c.PLAYER_SPEED * delta_time
        if keys[pygame.K_RIGHT]:
            self.x += s
        elif keys[pygame.K_LEFT]:
            self.x -= s


class Enemy(GalagaSprite):

    FRAMES = {
            'bee': [(96, 32, 16, 16)],
            'butterfly': [(112, 32, 16, 16)],
            'purple': [(128, 32, 16, 16)]
            }

    def __init__(self, x, y, enemy_type):
        super(Enemy, self).__init__(x, y, 16, 16)
        self.enemy_type = enemy_type
        self.origin_x = x
        
    def get_frame(self):
        return 0

    def update(self, delta_time: int, flash_flag: bool):
        import math
        # X MOVEMENT: Swaying relative to spawning origin
        self.x = self.origin_x + math.sin(pygame.time.get_ticks() * 0.002) * 40
        
        # Y MOVEMENT: Downward drift (Now functional thanks to float storage)
        self.y += 0.0115 * delta_time

        if self.y > c.GAME_SIZE.height:
            self.y = c.STAGE_TOP_Y - 16

    def display(self, surface: pygame.Surface):
        frame_num = self.get_frame()
        x, y, w, h = self.FRAMES[self.enemy_type][frame_num]
        self.image = grab_sheet(x, y, w, h)
        super(Enemy, self).display(surface)
        

class Missile(GalagaSprite):
    ENEMY_MISSILE = 246, 51, 3, 8
    PLAYER_MISSILE = 246, 67, 3, 8

    def __init__(self, x, y, vel, is_enemy):
        super(Missile, self).__init__(x, y, 2, 10)
        self.vel = vel
        self.is_enemy = is_enemy

        if self.is_enemy:
            img_slice = self.ENEMY_MISSILE
        else:
            img_slice = self.PLAYER_MISSILE
        ix, iy, w, h = img_slice
        self.image = grab_sheet(ix, iy, w, h)

    def update(self, delta_time: int, flash_flag: bool):
        self.x += self.vel.x * delta_time
        self.y += self.vel.y * delta_time


class Explosion(GalagaSprite):
    PLAYER_FRAME_DURATION = 140
    OTHER_FRAME_DURATION = 120

    PLAYER_FRAMES = [Rectangle(64, 112, 32, 32), Rectangle(96, 112, 32, 32), Rectangle(128, 112, 32, 32),
                     Rectangle(160, 112, 32, 32)]

    OTHER_FRAMES = [Rectangle(224, 80, 16, 16), Rectangle(240, 80, 16, 16), Rectangle(224, 96, 16, 16),
                    Rectangle(0, 112, 32, 32), Rectangle(32, 112, 32, 32)]

    def __init__(self, x: int, y: int, is_player_type=False):
        super(Explosion, self).__init__(x, y, 16, 16)
        self.is_player_type = is_player_type # Typo Fixed

        self.frame_timer = 0

        if self.is_player_type:
            self.image = grab_sheet(64, 112, 32, 32)
            self.frames = iter(self.PLAYER_FRAMES)
            self.frame_duration = self.PLAYER_FRAME_DURATION
        else:
            self.image = grab_sheet(224, 80, 16, 16)
            self.frames = iter(self.OTHER_FRAMES)
            self.frame_duration = self.OTHER_FRAME_DURATION

        self.frame = None
        self.next_frame()

    def next_frame(self):
        self.frame = next(self.frames)
        x, y, w, h = self.frame
        self.image = grab_sheet(x, y, w, h)
        self.frame_timer = 0

    def update(self, delta_time: int, flash_flag: bool):
        self.frame_timer += delta_time
        if self.frame_timer >= self.frame_duration:
            try:
                self.next_frame()
            except StopIteration:
                self.kill()
                return

    def display(self, surface: pygame.Surface):
        super(Explosion, self).display(surface)


def create_score_surface(number):
    sheet_y = 240
    char_width = 5
    char_height = 8
    sheet_char_width = 4

    number = int(number)
    str_num = str(number)
    length = len(str_num)

    surface = pygame.Surface((char_width * length, char_height)).convert_alpha()

    color = None
    if number in (800, 1000):
        color = c.BLUE
    else:
        color = c.YELLOW

    for i, character in enumerate(str_num):
        value = int(character)
        sheet_x = value * sheet_char_width
        number_sprite = grab_sheet(sheet_x, sheet_y, 4, 8)
        surface.blit(number_sprite, (i * char_width, 0))

    pixels = pygame.PixelArray(surface)
    pixels.replace((255, 255, 255), color)
    pixels.close()

    return surface


class ScoreText(GalagaSprite):
    text_sprites = pygame.sprite.Group()

    def __init__(self, x, y, number, lifetime=950):
        super(ScoreText, self).__init__(x, y, 1, 1, self.text_sprites)
        self.number = number
        self.image = create_score_surface(self.number)
        self.lifetime = lifetime

    def update(self, delta_time: int, flash_flag: bool):
        if self.lifetime < 0:
            self.kill()
            return
        self.lifetime -= delta_time