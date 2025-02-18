import pygame
import sys
import importlib.resources as pkg_resources
from lander_learner.utils.config import Config
from lander_learner import assets

# Colour constants
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_GREY = (50, 50, 50)
BLACK = (0, 0, 0)


class LunarLanderGUI:
    """
    Manages a pygame window to visualize the lander and environment.
    Capable of rendering a single environment or multiple environments in parallel.

    If multi_mode is False (default), env is a single environment.
    If multi_mode is True, env should be a list of environment objects, and
    styles should be a list of dictionaries (one per environment) specifying
    drawing options (e.g. tint colour and alpha).
    """

    def __init__(self, env, multi_mode=False, styles=None):
        self.multi_mode = multi_mode
        if self.multi_mode:
            # Expect a list of environments.
            self.envs = env
            # If no styles provided, default to white (opaque) for all.
            if styles is None:
                self.styles = [{"color": GREEN, "alpha": 255} for _ in self.envs]
            else:
                self.styles = styles
        else:
            # Single environment mode.
            self.env = env
            if styles is None:
                self.styles = {"color": GREEN, "alpha": 255}

        self._key_callback = None
        pygame.init()
        pygame.display.set_caption("2D Lunar Lander")
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.view_ref_x = 0
        self._load_lander_image()

    def render(self):
        """
        Draw the lander(s), the terrain, and any relevant info to the screen.
        """
        # Handle Pygame events (like closing window, key-press, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif (event.type == pygame.KEYDOWN or event.type == pygame.KEYUP) and self._key_callback:
                self._key_callback(event)

        self.screen.fill(BLACK)  # Clear to black

        self._draw_ground()

        view_ref_env = self.envs[-1] if self.multi_mode else self.env
        self.view_ref_x = view_ref_env.lander_position[0]

        if view_ref_env.target_zone:
            self._draw_target_zone(view_ref_env)
        self._draw_background()
        if self.multi_mode:
            pass
            for env, style in zip(self.envs, self.styles):
                self._draw_lander(env, style)
        else:
            self._draw_lander(self.env, self.styles)
        self._draw_debug_text()
        pygame.display.flip()
        self.clock.tick(Config.FPS * Config.REPLAY_SPEED)

    def _load_lander_image(self):
        """
        Load the lander image.
        """
        lander_width_px = int(Config.LANDER_WIDTH * Config.RENDER_SCALE)
        lander_height_px = int(Config.LANDER_HEIGHT * Config.RENDER_SCALE)
        self.lander_surface = pygame.Surface((lander_width_px, lander_height_px), pygame.SRCALPHA)
        with pkg_resources.open_binary(assets, "lander.png") as img_file:
            image = pygame.image.load(img_file).convert_alpha()
        self.lander_surface = pygame.transform.smoothscale(image, (lander_width_px, lander_height_px))

    def _draw_lander(self, env, style: dict = {}):
        """
        Draw the lander for a given environment.
        If a style dict is provided, apply a tint and transparency.
        """
        # Determine the lander surface.
        if style is not None and "color" in style and "alpha" in style:
            # Make a copy to apply tint.
            lander_surf = self.lander_surface.copy()
            tint_color = style["color"]
            lander_surf.fill(tint_color + (255,), None, pygame.BLEND_RGBA_MULT)
            lander_surf.set_alpha(style["alpha"])
        else:
            lander_surf = self.lander_surface

        # Get state info.
        lander_x, lander_y = env.lander_position
        lander_angle = env.lander_angle

        # Rotate the image (note the minus sign to match expected orientation).
        rotated_surface = pygame.transform.rotate(lander_surf, lander_angle * 180 / 3.14159)
        lander_rect = rotated_surface.get_rect(center=(
            self.world_to_screen_x(lander_x),
            self.world_to_screen_y(lander_y))
        )
        self.screen.blit(rotated_surface, lander_rect)

    def _draw_target_zone(self, env, style: dict = {}):
        """
        Draw the target zone on the screen as a semi-transparent rectanglular outline.
        """
        targ_x, targ_y = env.target_position
        targ_w, targ_h = env.target_zone_width, env.target_zone_height
        color = style.get("color", BLUE)
        alpha = style.get("alpha", 128)

        # Draw the target zone
        target_x_px = self.world_to_screen_x(targ_x - targ_w / 2)
        target_y_px = self.world_to_screen_y(targ_y + targ_h / 2)
        target_rect = pygame.Rect(
            target_x_px, target_y_px, int(targ_w * Config.RENDER_SCALE), int(targ_h * Config.RENDER_SCALE)
        )
        outline_surface = pygame.Surface((target_rect.width, target_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(
            outline_surface,
            (color[0], color[1], color[2], alpha),  # blue with 50% transparency
            outline_surface.get_rect(),
            2,  # outline thickness
        )
        self.screen.blit(outline_surface, target_rect.topleft)

    def _draw_background(self):
        """
        Draw a crosshatch pattern as a background grid.
        Uses the last environment's position as a reference.
        """
        hatch_size = 50
        offset = hatch_size - ((self.view_ref_x * Config.RENDER_SCALE) % hatch_size)
        for i in range(0, Config.SCREEN_WIDTH, hatch_size):
            dx = int(i + offset)
            pygame.draw.line(self.screen, DARK_GREY, (dx, 0), (dx, Config.SCREEN_HEIGHT))
        for dy in range(0, Config.SCREEN_HEIGHT, hatch_size):
            pygame.draw.line(self.screen, DARK_GREY, (0, dy), (Config.SCREEN_WIDTH, dy))

    def _draw_ground(self):
        """
        Draw a simple ground line (y = 0 in world coordinates).
        # TODO: Replace with automatic rendering for ease with more complex terrain
        """
        ground_y = self.world_to_screen_y(0)
        pygame.draw.line(self.screen, WHITE, (0, ground_y), (Config.SCREEN_WIDTH, ground_y), 2)

    def _draw_debug_text(self):
        """
        Render debugging information.
        Uses the last environment's state in multi_mode.
        """
        env = self.envs[-1] if self.multi_mode else self.env
        fps = self.clock.get_fps()
        debug_text = (
            f"Pos=({env.lander_position[0]:.2f}, {env.lander_position[1]:.2f}), "
            f"Angle={env.lander_angle:.2f}, Fuel={env.fuel_remaining:.2f}, "
            f"FPS={fps:.1f}, time={env.elapsed_time:.1f}, "
            f"reward={env._calculate_reward(False):.2f}"
        )
        text_surface = self.font.render(debug_text, True, WHITE)
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
        (Last) lander is always in the centre of the screen.
        """
        return int(Config.SCREEN_WIDTH // 2 + (x - self.view_ref_x) * Config.RENDER_SCALE)

    def world_to_screen_y(self, y):
        """
        Convert world y-coordinate to screen y-coordinate.
        y = -(50 / Config.RENDER_SCALE) is the bottom of the screen.
        """
        return int(Config.SCREEN_HEIGHT - (y * Config.RENDER_SCALE + 50))
