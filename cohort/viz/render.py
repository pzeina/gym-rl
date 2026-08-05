"""Render CohortEnv frames: map + units + radio transcript sidebar."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from cohort.core.ranks import Rank
from cohort.core.world import FOREST, WALL

if TYPE_CHECKING:
    from cohort.env.cohort_env import CohortEnv

#: display colors per rank (leaders warm, riflemen blue)
RANK_COLORS: dict[Rank, str] = {
    Rank.SLD: "#3b6fd4",
    Rank.CAP: "#e0a521",
    Rank.CDG: "#e0641f",
    Rank.SOA: "#b03ac2",
    Rank.CDS: "#d42f2f",
    Rank.ADU: "#8036d9",
    Rank.CDU: "#171717",
}

_TERRAIN_RGB = {
    0: (0.93, 0.92, 0.86),  # open
    FOREST: (0.68, 0.82, 0.62),
    WALL: (0.25, 0.25, 0.28),
}


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

    # chain-of-command links
    for s in env.roster.soldiers:
        if s.alive and s.leader_id is not None:
            leader = env.roster.by_id[s.leader_id]
            if leader.alive:
                ax.plot(
                    [s.pos[0], leader.pos[0]], [s.pos[1], leader.pos[1]],
                    color="#999999", lw=0.6, alpha=0.5, zorder=1,
                )

    for e in env.enemies:
        if e.alive:
            ax.scatter(*e.pos, marker="X", s=70, color="#c01717", zorder=3, edgecolors="white", linewidths=0.5)

    for s in env.roster.soldiers:
        if not s.alive:
            ax.scatter(*s.pos, marker="x", s=28, color="#aaaaaa", zorder=2)
            continue
        color = RANK_COLORS[s.effective_rank]
        is_leader = s.effective_authority > 0
        ax.scatter(
            *s.pos, marker="^" if is_leader else "o", s=80 if is_leader else 55,
            color=color, zorder=4, edgecolors="white", linewidths=0.8,
        )
        ax.annotate(
            s.callsign, s.pos, textcoords="offset points", xytext=(0, -11),
            ha="center", fontsize=6.5, color=color, fontweight="bold",
        )
        if s.mission is not None:  # dashed line to the mission anchor
            ax.plot(
                [s.pos[0], s.mission.anchor[0]], [s.pos[1], s.mission.anchor[1]],
                color=color, lw=0.7, ls=":", alpha=0.55, zorder=1,
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
