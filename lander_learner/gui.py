import pygame
import sys
from lander_learner.utils.config import Config
import importlib.resources as pkg_resources
from lander_learner import assets

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_GREY = (50, 50, 50)
BLACK = (0, 0, 0)


class LunarLanderGUI:
    """
    Manages a pygame window to visualize the lander and environment.
    """

    def __init__(self, env):
        self.env = env
        self._key_callback = None
        pygame.init()
        pygame.display.set_caption("2D Lunar Lander")
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        # Set a font for rendering text
        self.font = pygame.font.SysFont(None, 24)

        self.create_lander_surface()

    def render(self):
        """
        Draw the lander, the terrain, and any relevant info to the screen.
        """
        # Handle Pygame events (like closing window, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif (event.type == pygame.KEYDOWN or event.type == pygame.KEYUP) and self._key_callback:
                self._key_callback(event)

        self.screen.fill(BLACK)  # Clear to black

        # --- Draw ground as a simple line ---
        # TODO: Replace with automatic rendering for ease with more complex terrain
        ground_y = self.world_to_screen_y(0)
        pygame.draw.line(self.screen, WHITE, (0, ground_y), (Config.SCREEN_WIDTH, ground_y), 2)

        # Retrieve state from environment
        self.lander_x, self.lander_y = self.env.lander_position
        self.lander_angle = self.env.lander_angle
        self.lander_fuel = self.env.fuel_remaining

        if self.env.target_zone:
            self.draw_target_zone()

        self.draw_background()
        self.draw_lander()
        self.draw_debug_text()

        pygame.display.flip()
        self.clock.tick(Config.FPS * Config.REPLAY_SPEED)

    def create_lander_surface(self):
        """
        Create a surface for the lander and load the lander image.
        """
        self.lander_x = 0
        self.lander_y = 0
        self.lander_angle = 0
        self.lander_fuel = 0

        # Create a surface for the lander
        lander_width_px = Config.LANDER_WIDTH * Config.RENDER_SCALE
        lander_height_px = Config.LANDER_HEIGHT * Config.RENDER_SCALE
        self.lander_surface = pygame.Surface((lander_width_px, lander_height_px))
        self.lander_surface.set_colorkey(WHITE)
        self.lander_surface.fill(GREEN)
        with pkg_resources.open_binary(assets, "lander.png") as img_file:
            image = pygame.image.load(img_file).convert_alpha()
        image = pygame.transform.smoothscale(image, (int(lander_width_px), int(lander_height_px)))
        self.lander_surface.blit(image, (0, 0))
        # Draw the lander as a rotated rectangle

    def draw_lander(self):
        """
        Draw the lander on the screen.
        """
        lander_screen_x = self.world_to_screen_x(self.lander_x)
        lander_screen_y = self.world_to_screen_y(self.lander_y)

        rotated_surface = pygame.transform.rotate(self.lander_surface, self.lander_angle * 180 / 3.14159)
        lander_rect = rotated_surface.get_rect(center=(lander_screen_x, lander_screen_y))

        self.screen.blit(rotated_surface, lander_rect)

    def draw_target_zone(self):
        """
        Draw the target zone on the screen as a semi-transparent rectanglular outline.
        """
        targ_x, targ_y = self.env.target_position
        targ_w, targ_h = self.env.target_zone_width, self.env.target_zone_height

        # Draw the target zone
        target_x_px = self.world_to_screen_x(targ_x - targ_w / 2)
        target_y_px = self.world_to_screen_y(targ_y + targ_h / 2)
        target_rect = pygame.Rect(
            target_x_px, target_y_px, int(targ_w * Config.RENDER_SCALE), int(targ_h * Config.RENDER_SCALE)
        )
        outline_surface = pygame.Surface((target_rect.width, target_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(
            outline_surface,
            (BLUE[0], BLUE[1], BLUE[2], 128),  # blue with 50% transparency
            outline_surface.get_rect(),
            2,  # outline thickness
        )
        self.screen.blit(outline_surface, target_rect.topleft)

    def draw_background(self):
        """
        Draw a crosshatch pattern as a reference for motion.
        """
        # Add moving crosshatch
        hatch_size = 50
        offset = hatch_size - ((self.lander_x * Config.RENDER_SCALE) % hatch_size)
        for i in range(0, Config.SCREEN_WIDTH, hatch_size):
            dx = int(i + offset)
            pygame.draw.line(self.screen, DARK_GREY, (dx, 0), (dx, Config.SCREEN_HEIGHT))
        for j in range(0, Config.SCREEN_HEIGHT, hatch_size):
            dy = j
            pygame.draw.line(self.screen, DARK_GREY, (0, dy), (Config.SCREEN_WIDTH, dy))

    def draw_debug_text(self):
        """
        Draw some debugging text to the screen.
        """
        fps = self.clock.get_fps()  # Get the actual FPS
        debug_text = [f"Pos=({self.lander_x:.2f}, {self.lander_y:.2f}), Angle={self.lander_angle:.2f}, "
                      f"Fuel={self.lander_fuel:.2f}, FPS={fps:.1f}, time={self.env.elapsed_time:.1f}, "
                      f"reward={self.env._calculate_reward(False):.2f}"]
        text_surface = self.font.render(debug_text[0], True, WHITE)
        self.screen.blit(text_surface, (10, 10))

    def set_key_callback(self, callback):
        self._key_callback = callback

    def world_to_screen(self, x, y):
        """
        Convert world coordinates to screen coordinates.
        View centre is (lander_x, (SCREEN HEIGHT / 2 - 50) / Config.RENDER_SCALE)
        """
        return self.world_to_screen_x(x), self.world_to_screen_y(y)

    def world_to_screen_x(self, x):
        """
        Convert world x-coordinate to screen x-coordinate.
        Lander is always in the centre of the screen.
        """
        return int(Config.SCREEN_WIDTH // 2 + (x - self.lander_x) * Config.RENDER_SCALE)

    def world_to_screen_y(self, y):
        """
        Convert world y-coordinate to screen y-coordinate.
        y = -(50 / Config.RENDER_SCALE) is the bottom of the screen.
        """
        return int(Config.SCREEN_HEIGHT - (y * Config.RENDER_SCALE + 50))
