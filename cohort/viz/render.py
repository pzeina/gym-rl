"""Render CohortEnv frames: map + units + radio transcript sidebar.

Unit symbology follows NATO APP-6 / MIL-STD-2525 conventions:

* friendly — blue rectangle frame, infantry saltire (crossed diagonals),
  echelon indicator above the frame (∅ team, ● squad, ●●● platoon, | company);
  riflemen are individuals and get a smaller frame with no echelon mark
* hostile — red diamond frame with saltire
* affiliation colors use the 2525 "light" fills: Crystal Blue / Salmon
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle

from cohort.core.ranks import ECHELON_MARKS
from cohort.core.world import FOREST, WALL

if TYPE_CHECKING:
    from cohort.env.cohort_env import CohortEnv

#: MIL-STD-2525 affiliation fills (light) + frame line colors.
FRIEND_FILL = "#80e0ff"
FRIEND_LINE = "#005a8c"
HOSTILE_FILL = "#ff8080"
HOSTILE_LINE = "#8c0000"
KIA_COLOR = "#8a8a8a"
HUMAN_RING = "#c9971c"  # gold ring marking a human-commander unit

_TERRAIN_RGB = {
    0: (0.93, 0.92, 0.86),  # open
    FOREST: (0.68, 0.82, 0.62),
    WALL: (0.25, 0.25, 0.28),
}


def _friendly_symbol(ax: plt.Axes, x: float, y: float, echelon: str, *, small: bool) -> None:
    """APP-6 friendly infantry: blue rectangle + saltire (+ echelon mark)."""
    w, h = (0.9, 0.62) if not small else (0.62, 0.44)
    ax.add_patch(
        Rectangle((x - w / 2, y - h / 2), w, h, facecolor=FRIEND_FILL, edgecolor=FRIEND_LINE,
                  linewidth=1.1, zorder=4)
    )
    ax.add_line(Line2D([x - w / 2, x + w / 2], [y - h / 2, y + h / 2], color=FRIEND_LINE, lw=0.9, zorder=5))
    ax.add_line(Line2D([x - w / 2, x + w / 2], [y + h / 2, y - h / 2], color=FRIEND_LINE, lw=0.9, zorder=5))
    if echelon:
        ax.annotate(echelon, (x, y - h / 2 - 0.15), ha="center", va="bottom",
                    fontsize=6.5, color=FRIEND_LINE, zorder=6, annotation_clip=False)


def _hostile_symbol(ax: plt.Axes, x: float, y: float, *, alive: bool) -> None:
    """APP-6 hostile: red diamond + saltire (gray when KIA)."""
    r = 0.5
    fill = HOSTILE_FILL if alive else "#d0d0d0"
    line = HOSTILE_LINE if alive else KIA_COLOR
    ax.add_patch(
        Polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], closed=True,
                facecolor=fill, edgecolor=line, linewidth=1.1, zorder=3)
    )
    s = r * 0.45
    ax.add_line(Line2D([x - s, x + s], [y - s, y + s], color=line, lw=0.9, zorder=4))
    ax.add_line(Line2D([x - s, x + s], [y + s, y - s], color=line, lw=0.9, zorder=4))


def render_frame(env: CohortEnv, transcript_lines: int = 10) -> np.ndarray:
    """Return an RGB frame (H, W, 3) uint8 of the current state."""
    world = env.world
    img = np.zeros((world.height, world.width, 3))
    for value, color in _TERRAIN_RGB.items():
        img[world.grid == value] = color

    fig, (ax, ax_log) = plt.subplots(
        1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1.05, 1.0]}, dpi=110
    )
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    outcome = f" — {env._episode_outcome.upper()}" if env._episode_outcome else ""
    ax.set_title(f"{env.spec_cfg.name}  t={env._step_count}{outcome}", fontsize=11)

    for obj in world.objectives:
        ax.add_patch(Circle(obj.pos, obj.radius, fill=False, ls="--", lw=1.2, color="#666666"))
        ax.annotate(
            f"OBJ {obj.name}", obj.pos, textcoords="offset points", xytext=(0, 11),
            ha="center", fontsize=7, color="#444444",
        )

    # control measures (A5): waypoints as small diamonds, phase lines as thin
    # dashed lines — both labeled, drawn under the unit symbology
    for wp in world.waypoints:
        x, y, r = wp.pos[0], wp.pos[1], 0.4
        ax.add_patch(
            Polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], closed=True,
                    fill=False, edgecolor="#7a6f9a", linewidth=1.0, ls="--", zorder=2)
        )
        ax.annotate(
            f"WP {wp.name}", wp.pos, textcoords="offset points", xytext=(0, -11),
            ha="center", fontsize=6.5, color="#7a6f9a",
        )
    for pl in world.phase_lines:
        ax.plot(
            [pl.a[0], pl.b[0]], [pl.a[1], pl.b[1]],
            color="#7a6f9a", lw=0.9, ls="--", alpha=0.85, zorder=2,
        )
        mid = ((pl.a[0] + pl.b[0]) / 2, (pl.a[1] + pl.b[1]) / 2)
        ax.annotate(
            f"PL {pl.name}", mid, textcoords="offset points", xytext=(4, 4),
            fontsize=6.5, color="#7a6f9a",
        )

    # chain-of-command links
    for s in env.roster.soldiers:
        if s.alive and s.leader_id is not None:
            leader = env.roster.by_id[s.leader_id]
            if leader.alive:
                ax.plot(
                    [s.pos[0], leader.pos[0]], [s.pos[1], leader.pos[1]],
                    color="#999999", lw=0.6, alpha=0.5, zorder=1,
                )

    # revealed traps (BRIQUE devices): hostile-colored warning triangle.
    # Unrevealed traps stay hidden — the frame shows what the fight revealed.
    for trap in getattr(env, "traps", []):
        if trap.revealed:
            x, y, r = trap.pos[0], trap.pos[1], 0.45
            ax.add_patch(
                Polygon([(x, y - r), (x + r, y + r * 0.8), (x - r, y + r * 0.8)], closed=True,
                        facecolor="#ffd24d", edgecolor=HOSTILE_LINE, linewidth=1.1, zorder=3)
            )

    for e in env.enemies:
        _hostile_symbol(ax, e.pos[0], e.pos[1], alive=e.alive)

    for s in env.roster.soldiers:
        if not s.alive:
            ax.scatter(*s.pos, marker="x", s=28, color=KIA_COLOR, zorder=2)
            continue
        _friendly_symbol(
            ax, s.pos[0], s.pos[1], ECHELON_MARKS[s.effective_rank],
            small=s.effective_authority == 0,
        )
        if s.human:  # gold ring: a human commander embodied in the sim
            ax.add_patch(
                Circle(s.pos, 0.85, fill=False, lw=1.3, color=HUMAN_RING, zorder=6)
            )
        ax.annotate(
            s.callsign, s.pos, textcoords="offset points", xytext=(0, -12),
            ha="center", fontsize=6.5, color=FRIEND_LINE, fontweight="bold",
        )
        if s.mission is not None:  # dashed line to the mission anchor
            ax.plot(
                [s.pos[0], s.mission.anchor[0]], [s.pos[1], s.mission.anchor[1]],
                color=FRIEND_LINE, lw=0.7, ls=":", alpha=0.5, zorder=1,
            )

    # radio log sidebar
    ax_log.axis("off")
    ax_log.set_title("radio net", fontsize=10, loc="left")
    recent = env.transcript.messages[-transcript_lines:]
    lines = [f"[t={m.step:>3}] {m.text}" for m in recent] or ["(net silent)"]
    ax_log.text(
        0.0, 0.98, "\n".join(lines), transform=ax_log.transAxes, fontsize=6.8,
        family="monospace", va="top", wrap=True,
    )

    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def save_gif(frames: list[np.ndarray], path: str, fps: int = 8) -> None:
    """Write frames to an animated GIF."""
    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0
    )
