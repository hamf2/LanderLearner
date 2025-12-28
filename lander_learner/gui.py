"""
LunarLanderGUI Module

This module provides the LunarLanderGUI class, which manages a pygame window to visualize the
lander and environment. It supports rendering a single environment or multiple environments in
parallel. When in multi-mode, it accepts a list of environments and a corresponding list of style
dictionaries (for tinting, transparency, etc.).
"""

import pygame
import sys
import os
from datetime import datetime
import importlib.resources as pkg_resources
from lander_learner.utils.config import Config
from lander_learner import assets

from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:  # pragma: no cover - assist typing only
    from lander_learner.environment import LunarLanderEnv

# Colour constants
WHITE = (255, 255, 255)
LIGHT_BLUE = (128, 200, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_GREY = (50, 50, 50)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)


class LunarLanderGUI:
    """Manages a pygame window to visualize the lander and environment.

    This class is capable of rendering either a single environment or multiple environments in parallel.

    In single-mode, `env` is expected to be a single environment instance and a single style is applied.
    In multi-mode, `env` should be a list of environment objects and `styles` should be a list of
    dictionaries (one per environment) specifying drawing options such as tint colour and alpha.

    Attributes:
        view_ref_x (float): The reference x-coordinate for converting world coordinates to screen coordinates.
        screen (pygame.Surface): The main display surface.
        clock (pygame.time.Clock): Clock used for regulating frame rate.
        font (pygame.font.Font): Font used for rendering debug text.
        lander_surface (pygame.Surface): Surface containing the lander image.
        _key_callback (callable): Optional callback for key events.
    """

    def __init__(
        self,
        env: Union["LunarLanderEnv", list["LunarLanderEnv"]],
        multi_mode: bool = False,
        styles: Union[dict, list[dict]] = None,
        record: bool = False
    ):
        """Initializes the LunarLanderGUI.

        Args:
            env (Union["LunarLanderEnv", list["LunarLanderEnv"]]): A single environment instance
                (if multi_mode is False) or a list of environment instances.
            multi_mode (bool, optional): Flag indicating whether to render multiple environments.
                Defaults to False.
            styles (dict or list of dict, optional): For single-mode a dictionary specifying the style.
                For multi-mode, a list of dictionaries (one per environment) specifying the style.
                Each style dict may contain keys "color" (an RGB tuple) and "alpha" (an integer 0-255).
                Defaults to None, in which case a default style is applied (GREEN, opaque).
        """
        self.multi_mode = multi_mode
        if self.multi_mode:
            # Expect a list of environments.
            self.envs = env
            # If no styles provided, default to green (opaque) for all.
            if styles is None:
                self.styles = [{"color": GREEN, "alpha": 255} for _ in self.envs]
            else:
                self.styles = styles
        else:
            # Single environment mode.
            self.env = env
            if styles is None:
                self.styles = {"color": GREEN, "alpha": 255}
            else:
                self.styles = styles

        self.record = record
        if record:
            self.record_dir = Config.DEFAULT_RECORDING_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
            os.makedirs(str(self.record_dir), exist_ok=True)
            self.frame_count = 0

        self._key_callback = None
        pygame.init()
        pygame.display.set_caption("2D Lunar Lander")
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.view_ref_x = 0.0
        self.view_ref_y = 25.0
        self._load_lander_image()

    def render(self):
        """Draws the lander(s), terrain, and debug information to the screen.

        Handles pygame events, clears the screen, draws the ground, background grid,
        target zone (if enabled), lander(s), and debug text. Finally, it updates the display
        and regulates the frame rate.
        """
        # Handle Pygame events (e.g., closing window, key events).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif (event.type == pygame.KEYDOWN or event.type == pygame.KEYUP) and self._key_callback:
                self._key_callback(event)

        self.screen.fill(BLACK)  # Clear to black

        # Use the last environment in the list (or the single env) as reference for view.
        view_ref_env = self.envs[-1] if self.multi_mode else self.env
        self._set_view_reference(view_ref_env)

        self._draw_background()
        self._draw_terrain(view_ref_env)
        if view_ref_env.target_zone:
            self._draw_target_zone(view_ref_env)
        if view_ref_env.finish_line:
            self._draw_finish_line(view_ref_env)
        if self.multi_mode:
            for env, style in zip(self.envs, self.styles):
                self._draw_lander(env, style)
        else:
            self._draw_lander(self.env, self.styles)
        self._draw_debug_text(view_ref_env)
        pygame.display.flip()

        if self.record:
            frame_path = os.path.join(self.record_dir, f"frame_{self.frame_count:06d}.png")
            pygame.image.save(self.screen, frame_path)
            self.frame_count += 1

        self.clock.tick(Config.FPS * Config.REPLAY_SPEED)

    def _set_view_reference(self, env: "LunarLanderEnv"):
        """Sets the view reference coordinates based on the lander's position.

        Args:
            env (LunarLanderEnv): The environment instance whose lander position is used for the view reference.
        """
        self.view_ref_x = env.lander_position[0]
        if env.get_level_metadata().get("type", "") == "half_plane":
            # Fixed vertical offset for better visibility.
            self.view_ref_y = 25.0
        else:
            self.view_ref_y = env.lander_position[1]

    def _load_lander_image(self):
        """Loads and scales the lander image from the assets.

        The image is loaded from the assets package and scaled based on the configuration
        settings for lander width, height, and render scale.
        """
        lander_width_px = int(Config.LANDER_WIDTH * Config.RENDER_SCALE)
        lander_height_px = int(Config.LANDER_HEIGHT * Config.RENDER_SCALE)
        self.lander_surface = pygame.Surface((lander_width_px, lander_height_px), pygame.SRCALPHA)
        with pkg_resources.open_binary(assets, "lander.png") as img_file:
            image = pygame.image.load(img_file).convert_alpha()
        self.lander_surface = pygame.transform.smoothscale(image, (lander_width_px, lander_height_px))

    def _draw_lander(self, env: "LunarLanderEnv", style: dict = {}):
        """Draws the lander for a given environment.

        If a style dictionary is provided with "color" and "alpha", a tinted copy of the
        lander image is created.

        Args:
            env (LunarLanderEnv): The environment instance whose lander state is to be drawn.
            style (dict, optional): Dictionary with style parameters ("color" and "alpha").
                Defaults to an empty dict.
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

        # Rotate the image (note the multiplication factor converts radians to degrees).
        rotated_surface = pygame.transform.rotate(lander_surf, lander_angle * 180 / 3.14159)
        lander_rect = rotated_surface.get_rect(center=(
            self.world_to_screen_x(lander_x),
            self.world_to_screen_y(lander_y)
        ))
        self.screen.blit(rotated_surface, lander_rect)

    def _draw_target_zone(self, env: "LunarLanderEnv", style: dict = {}):
        """Draws the target zone as a semi-transparent rectangular outline.

        The target zone is drawn using the specified style (default is blue with 50% transparency).

        Args:
            env (LunarLanderEnv): The environment instance containing the target zone parameters.
            style (dict, optional): Dictionary with style parameters ("color" and "alpha").
                Defaults to an empty dict.
        """
        targ_x, targ_y = env.target_position
        targ_w, targ_h = env.target_zone_width, env.target_zone_height
        color = style.get("color", BLUE)
        alpha = style.get("alpha", 128)

        # Calculate the screen coordinates for the target zone.
        target_x_px = self.world_to_screen_x(targ_x - targ_w / 2)
        target_y_px = self.world_to_screen_y(targ_y + targ_h / 2)
        target_rect = pygame.Rect(
            target_x_px, target_y_px, int(targ_w * Config.RENDER_SCALE), int(targ_h * Config.RENDER_SCALE)
        )
        outline_surface = pygame.Surface((target_rect.width, target_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(
            outline_surface,
            (color[0], color[1], color[2], alpha),
            outline_surface.get_rect(),
            2  # Outline thickness.
        )
        self.screen.blit(outline_surface, target_rect.topleft)

    def _draw_finish_line(self, env: "LunarLanderEnv"):
        """Draws the lap finish line as a bold segment across the corridor."""

        segment = getattr(env, "finish_line", None)
        if not segment:
            return
        start, end = segment
        start_px = self.world_to_screen(*start)
        end_px = self.world_to_screen(*end)
        pygame.draw.line(self.screen, WHITE, start_px, end_px, 3)

    def _draw_background(self):
        """Draws a crosshatch pattern as a background grid.

        The grid uses the view reference x-coordinate (from the last environment) to determine the offset.
        """
        hatch_size = 50
        offset_x = hatch_size - ((self.view_ref_x * Config.RENDER_SCALE) % hatch_size)
        offset_y = hatch_size + ((self.view_ref_y * Config.RENDER_SCALE) % hatch_size)
        for i in range(0, Config.SCREEN_WIDTH, hatch_size):
            dx = int(i + offset_x)
            pygame.draw.line(self.screen, DARK_GREY, (dx, 0), (dx, Config.SCREEN_HEIGHT))
        for j in range(0, Config.SCREEN_HEIGHT, hatch_size):
            dy = int(j + offset_y)
            pygame.draw.line(self.screen, DARK_GREY, (0, dy), (Config.SCREEN_WIDTH, dy))

    def _draw_ground(self):
        """Draws a simple ground line corresponding to y = 0 in world coordinates.

        Note:
            This is a basic rendering of the ground. It may be replaced in the future with
            automatic terrain rendering for more complex scenarios.
        """
        ground_y = self.world_to_screen_y(0)
        pygame.draw.line(self.screen, WHITE, (0, ground_y), (Config.SCREEN_WIDTH, ground_y), 2)

    def _draw_terrain(self, env: "LunarLanderEnv"):
        """Draws level geometry using static shapes from the physics engine.

        Args:
            env (LunarLanderEnv): The environment instance whose terrain is to be drawn.
        """

        def colour_for(body_type: str):
            if body_type == "static":
                return WHITE
            if body_type == "kinematic":
                return LIGHT_BLUE
            if body_type == "dynamic":
                return ORANGE
            return ORANGE

        geometry = env.get_body_vertices(static=True, kinematic=True, dynamic=True, lander=False)
        for segment in geometry.segments:
            start = self.world_to_screen(*segment.start)
            end = self.world_to_screen(*segment.end)
            width_pixels = max(1, int(self._segment_width_pixels(segment.radius)))
            pygame.draw.line(self.screen, colour_for(segment.body_type), start, end, width_pixels)

        for polygon in geometry.polys:
            if len(polygon.vertices) >= 2:
                vertices = [self.world_to_screen(*vertex) for vertex in polygon.vertices]
                pygame.draw.polygon(self.screen, colour_for(polygon.body_type), vertices, 0)

    def _segment_width_pixels(self, radius: float) -> float:
        """Converts a segment radius to pixel width for rendering.

        Args:
            radius (float): The radius of the segment in world units.

        Returns:
            float: The width of the segment in pixels."""
        return radius * Config.RENDER_SCALE * 2

    def _draw_debug_text(self, env: "LunarLanderEnv"):
        """Renders debugging information on the screen.

        Uses the state of the last environment (in multi-mode) or the single environment.
        Displays position, angle, fuel, FPS, elapsed time, and the current reward.

        Args:
            env (LunarLanderEnv): The environment instance whose debug information is to be displayed.
        """
        fps = self.clock.get_fps()
        debug_text = (
            f"Pos=({env.lander_position[0]:.2f}, {env.lander_position[1]:.2f}), "
            f"Angle={env.lander_angle:.2f}, Fuel={env.fuel_remaining:.2f}, "
            f"FPS={fps:.1f}, time={env.elapsed_time:.1f}, "
            f"reward={env._calculate_reward(False):.2f}"
        )
        if env.physics_engine.get_level_metadata().get("type", "") == "lap":
            debug_text += f", laps={getattr(env, 'lap_counter', 0)}"
        text_surface = self.font.render(debug_text, True, WHITE)
        self.screen.blit(text_surface, (10, 10))

    def set_key_callback(self, callback: callable):
        """Sets the key event callback.

        Args:
            callback (callable): A function to be called when a key event occurs.
        """
        self._key_callback = callback

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Converts world coordinates to screen coordinates.

        The view center is determined by the lander's x-position and a fixed offset in y.

        Args:
            x (float): World x-coordinate.
            y (float): World y-coordinate.

        Returns:
            tuple: A tuple (screen_x, screen_y) representing the screen coordinates.
        """
        return self.world_to_screen_x(x), self.world_to_screen_y(y)

    def world_to_screen_x(self, x: float) -> int:
        """Converts a world x-coordinate to a screen x-coordinate.

        The (last) lander's x-position is always centered on the screen.

        Args:
            x (float): World x-coordinate.

        Returns:
            int: Screen x-coordinate.
        """
        return int(Config.SCREEN_WIDTH // 2 + (x - self.view_ref_x) * Config.RENDER_SCALE)

    def world_to_screen_y(self, y: float) -> int:
        """Converts a world y-coordinate to a screen y-coordinate.

        The conversion uses the render scale and a fixed vertical offset.

        Args:
            y (float): World y-coordinate.

        Returns:
            int: Screen y-coordinate.
        """
        return int(Config.SCREEN_HEIGHT // 2 - (y - self.view_ref_y) * Config.RENDER_SCALE)
