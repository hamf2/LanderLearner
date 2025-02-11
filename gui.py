import pygame
import sys
from utils.config import Config

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
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
        self.screen = pygame.display.set_mode(
            (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT)
        )
        self.clock = pygame.time.Clock()

        # Draw the lander as a rotated rectangle
        lander_width_px = Config.LANDER_WIDTH * Config.RENDER_SCALE
        lander_height_px = Config.LANDER_HEIGHT * Config.RENDER_SCALE
        
        # Create a surface for the lander
        self.lander_surface = pygame.Surface((lander_width_px, lander_height_px))
        self.lander_surface.set_colorkey(WHITE)
        self.lander_surface.fill(GREEN)
        image = pygame.image.load("assets/lander.png").convert_alpha()
        image = pygame.transform.smoothscale(image, (int(lander_width_px), int(lander_height_px)))
        self.lander_surface.blit(image, (0, 0))

        # Create a font for rendering text
        self.font = pygame.font.SysFont(None, 24)

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
        ground_y = int(Config.SCREEN_HEIGHT - 50)
        pygame.draw.line(
            self.screen,
            WHITE,
            (0, ground_y),
            (Config.SCREEN_WIDTH, ground_y),
            2
        )

        # Retrieve state from environment
        x, y = self.env.lander_position
        angle = self.env.lander_angle
        fuel = self.env.fuel_remaining

        # Add moving crosshatch
        hatch_size = 50
        offset = -((x * Config.RENDER_SCALE) % hatch_size)
        for i in range(0, Config.SCREEN_WIDTH, hatch_size):
            dx = i + offset
            pygame.draw.line(self.screen, DARK_GREY, (dx, 0), (dx, Config.SCREEN_HEIGHT))
        for j in range(0, Config.SCREEN_HEIGHT, hatch_size):
            dy = j
            pygame.draw.line(self.screen, DARK_GREY, (0, dy), (Config.SCREEN_WIDTH, dy))

        # Convert physics coordinates to screen coordinates
        # view_centre = (x, 50 / Config.RENDER_SCALE)
        lander_screen_x = int(Config.SCREEN_WIDTH // 2)
        lander_screen_y = int(Config.SCREEN_HEIGHT - (y * Config.RENDER_SCALE + 50))
        
        rotated_surface = pygame.transform.rotate(self.lander_surface, angle * 180 / 3.14159)
        lander_rect = rotated_surface.get_rect(center=(lander_screen_x, lander_screen_y))

        self.screen.blit(rotated_surface, lander_rect)

        # Draw some text for debugging
        fps = self.clock.get_fps()  # Get the actual FPS
        debug_text = f"Pos=({x:.2f}, {y:.2f}), Angle={angle:.2f}, Fuel={fuel:.2f}, FPS={fps:.1f}, time={self.env.elapsed_time:.1f}, reward={self.env._calculate_reward(False):.2f}"
        text_surface = self.font.render(debug_text, True, WHITE)
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(Config.FPS*Config.REPLAY_SPEED)

    def set_key_callback(self, callback):
        self._key_callback = callback
