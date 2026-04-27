"""
Final teaching-oriented Vicsek demo

This version is designed for a teaching paper:
1. The snapshot figure emphasizes noise eta directly.
2. The animation is tuned to show the emergence process more clearly.
3. The animation title uses eta and step number instead of Phi.

Outputs
-------
- flocking_snapshots.png
- flocking_emergence.gif

Suggested use
-------------
- flocking_snapshots.png -> main text figure
- flocking_emergence.gif -> supplementary material
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# Parameters
# ============================================================

@dataclass
class VicsekParams:
    n: int = 300
    L: float = 10.0
    v0: float = 0.05
    r: float = 1.0
    dt: float = 1.0
    steps: int = 500
    seed: int = 42


# ============================================================
# Core utilities
# ============================================================

def wrap_angle(theta: np.ndarray) -> np.ndarray:
    return np.mod(theta, 2.0 * np.pi)


def order_parameter(theta: np.ndarray) -> float:
    vx = np.cos(theta).mean()
    vy = np.sin(theta).mean()
    return float(np.hypot(vx, vy))


def pairwise_periodic_displacement(x: np.ndarray, L: float) -> np.ndarray:
    dx = x[None, :] - x[:, None]
    dx -= L * np.round(dx / L)
    return dx


def vicsek_step(
    pos: np.ndarray,
    theta: np.ndarray,
    eta: float,
    params: VicsekParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = pos[:, 0]
    y = pos[:, 1]

    dx = pairwise_periodic_displacement(x, params.L)
    dy = pairwise_periodic_displacement(y, params.L)
    dist2 = dx**2 + dy**2

    neighbors = dist2 <= params.r**2

    sin_sum = neighbors @ np.sin(theta)
    cos_sum = neighbors @ np.cos(theta)
    mean_theta = np.arctan2(sin_sum, cos_sum)

    noise = rng.uniform(-eta / 2.0, eta / 2.0, size=params.n)
    new_theta = wrap_angle(mean_theta + noise)

    vel = np.column_stack((np.cos(new_theta), np.sin(new_theta)))
    new_pos = pos + params.v0 * params.dt * vel
    new_pos %= params.L

    return new_pos, new_theta


def run_simulation(
    eta: float,
    params: VicsekParams,
    burn_in: int = 0,
    seed_offset: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(params.seed + seed_offset)

    pos = rng.uniform(0.0, params.L, size=(params.n, 2))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=params.n)

    frames_pos = []
    frames_theta = []
    phi_values = []

    total_steps = burn_in + params.steps
    for step in range(total_steps):
        pos, theta = vicsek_step(pos, theta, eta, params, rng)
        if step >= burn_in:
            frames_pos.append(pos.copy())
            frames_theta.append(theta.copy())
            phi_values.append(order_parameter(theta))

    return {
        "pos": np.array(frames_pos),
        "theta": np.array(frames_theta),
        "phi": np.array(phi_values),
        "eta": np.array([eta]),
    }


# ============================================================
# Plotting
# ============================================================

def plot_snapshot(
    ax,
    pos: np.ndarray,
    theta: np.ndarray,
    L: float,
    title: str,
) -> None:
    u = np.cos(theta)
    v = np.sin(theta)

    ax.quiver(
        pos[:, 0],
        pos[:, 1],
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=3.0,
        width=0.004,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
    )
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)


def save_snapshot_figure(
    high_data: dict[str, np.ndarray],
    low_data: dict[str, np.ndarray],
    params: VicsekParams,
    eta_high_snapshot: float,
    eta_low_snapshot: float,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    plot_snapshot(
        axes[0],
        high_data["pos"][-1],
        high_data["theta"][-1],
        params.L,
        f"(a) High noise, η = {eta_high_snapshot:.2f}",
    )
    plot_snapshot(
        axes[1],
        low_data["pos"][-1],
        low_data["theta"][-1],
        params.L,
        f"(b) Low noise, η = {eta_low_snapshot:.2f}",
    )

    fig.suptitle("Flocking model at different noise levels", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_emergence_animation(
    data: dict[str, np.ndarray],
    params: VicsekParams,
    eta_animation: float,
    outpath: Path,
    frame_stride: int = 2,
    fps: int = 12,
) -> None:
    pos_all = data["pos"][::frame_stride]
    theta_all = data["theta"][::frame_stride]

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.set_xlim(0, params.L)
    ax.set_ylim(0, params.L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    pos0 = pos_all[0]
    th0 = theta_all[0]

    quiv = ax.quiver(
        pos0[:, 0],
        pos0[:, 1],
        np.cos(th0),
        np.sin(th0),
        angles="xy",
        scale_units="xy",
        scale=3.0,
        width=0.004,
    )
    title = ax.set_title("", fontsize=12)

    def update(frame_idx: int):
        pos = pos_all[frame_idx]
        theta = theta_all[frame_idx]

        quiv.set_offsets(pos)
        quiv.set_UVC(np.cos(theta), np.sin(theta))
        step_number = frame_idx * frame_stride + 1
        title.set_text(
            f"Emergence of collective motion (η = {eta_animation:.2f}), "
            f"step = {step_number}"
        )
        return quiv, title

    anim = FuncAnimation(
        fig,
        update,
        frames=len(pos_all),
        interval=1000 / fps,
        blit=False,
    )

    anim.save(outpath, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    outdir = Path(".")
    params = VicsekParams()

    # --------------------------------------------------------
    # Parameters for the paper figure
    # --------------------------------------------------------
    eta_high_snapshot = 2.8   # clearly disordered
    eta_low_snapshot = 0.25   # clearly aligned

    # --------------------------------------------------------
    # Parameters for the animation
    # Chosen to keep the disordered stage visible for longer
    # and to show a clearer emergence process.
    # --------------------------------------------------------
    eta_animation = 0.85

    print("Running snapshot simulations...")
    high_data = run_simulation(
        eta=eta_high_snapshot,
        params=params,
        burn_in=120,
        seed_offset=0,
    )
    low_data = run_simulation(
        eta=eta_low_snapshot,
        params=params,
        burn_in=120,
        seed_offset=1,
    )

    print("Running emergence animation simulation...")
    anim_data = run_simulation(
        eta=eta_animation,
        params=params,
        burn_in=0,
        seed_offset=2,
    )

    snapshot_path = outdir / "flocking_snapshots.png"
    gif_path = outdir / "flocking_emergence.gif"

    print("Saving snapshot figure...")
    save_snapshot_figure(
        high_data,
        low_data,
        params,
        eta_high_snapshot,
        eta_low_snapshot,
        snapshot_path,
    )

    print("Saving emergence animation...")
    save_emergence_animation(
        anim_data,
        params,
        eta_animation,
        gif_path,
        frame_stride=2,
        fps=12,
    )

    print("\nDone.")
    print(f"Saved snapshot figure: {snapshot_path.resolve()}")
    print(f"Saved animation GIF:   {gif_path.resolve()}")
    print(f"Mean Φ (high-noise snapshot): {high_data['phi'].mean():.3f}")
    print(f"Mean Φ (low-noise snapshot):  {low_data['phi'].mean():.3f}")
    print(f"Mean Φ (animation run):       {anim_data['phi'].mean():.3f}")


if __name__ == "__main__":
    main()