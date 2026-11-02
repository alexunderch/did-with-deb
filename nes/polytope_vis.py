"""Polytope visualisation for 2x2 games.

A joint strategy of a 2x2 game is a point of the 3-simplex (a tetrahedron with
the four joint actions 00, 01, 10, 11 as vertices). The (epsilon-)CE and CCE
sets are polytopes inside it:

  Nash (product distributions, a curved surface)  subset of  CE  subset of  CCE.
"""

import itertools

import numpy as np
import scipy.optimize
import scipy.spatial

# Regular tetrahedron: sigma (barycentric, [4]) -> 3-D point = sigma @ _TETRA.
_TETRA = np.array(
  [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
) / np.sqrt(3.0)


def to_xyz(sigma: np.ndarray) -> np.ndarray:
  return np.asarray(sigma, dtype=float).reshape(-1, 4) @ _TETRA


def constraint_rows(payoffs: np.ndarray, mode: str) -> np.ndarray:
  """Rows R [m, 4] such that the eps-(C)CE set is {sigma in simplex: R @ sigma <= eps}.

  CCE: R[(p, d), a] = G_p(d, a_-p) - G_p(a).
  CE:  R[(p, d, r), a] = 1{a_p = r} * (G_p(d, a_-p) - G_p(r, a_-p)),  d != r.
  (CE gains are joint-weighted, as in equilibria_utils.)
  """
  g = np.asarray(payoffs, dtype=float)
  joint = list(itertools.product(range(2), range(2)))  # flat index = 2 * i + j

  def deviation(p, d, a):  # G_p(d, a_-p)
    return g[p][(d, a[1])] if p == 0 else g[p][(a[0], d)]

  rows = []
  for p in range(2):
    if mode == "CCE":
      for d in range(2):
        rows.append([deviation(p, d, a) - g[p][a] for a in joint])
    elif mode == "CE":
      for d in range(2):
        for r in range(2):
          if d != r:
            rows.append(
              [(deviation(p, d, a) - deviation(p, r, a)) * (a[p] == r) for a in joint]
            )
    else:
      raise ValueError(f"mode must be 'CE' or 'CCE', got {mode!r}")
  return np.array(rows)


def polytope_vertices(
  payoffs: np.ndarray, mode: str, eps: float = 0.0, tol: float = 1e-9
) -> np.ndarray:
  """Vertices ([k, 4], barycentric) of the eps-(C)CE polytope, by brute-force
  enumeration: every vertex is the intersection of 3 active constraints (in
  the 3 free coordinates sigma_0..2, sigma_3 = 1 - sum). Handles degenerate
  polytopes (a single point, a segment, ...) that a half-space-intersection
  routine cannot."""
  rows = constraint_rows(payoffs, mode)
  # R @ sigma <= eps with sigma_3 = 1 - x0 - x1 - x2  ->  a . x <= b
  a_eq = rows[:, :3] - rows[:, 3:4]
  b_eq = eps - rows[:, 3]
  a_all = np.vstack([a_eq, -np.eye(3), np.ones((1, 3))])  # sigma_i >= 0, sigma_3 >= 0
  b_all = np.concatenate([b_eq, np.zeros(3), [1.0]])

  verts = []
  for idx in itertools.combinations(range(len(a_all)), 3):
    sub = a_all[list(idx)]
    if abs(np.linalg.det(sub)) < 1e-12:
      continue
    x = np.linalg.solve(sub, b_all[list(idx)])
    if np.all(a_all @ x <= b_all + tol):
      verts.append(x)
  if not verts:
    return np.zeros((0, 4))
  verts = np.unique(np.round(np.array(verts), 9), axis=0)
  return np.hstack([verts, 1.0 - verts.sum(axis=1, keepdims=True)])


def nash_equilibria(payoffs: np.ndarray) -> list[np.ndarray]:
  """Nash equilibria of a 2x2 game as [2, 2] product distributions: pure ones,
  plus the fully mixed one if it exists (support enumeration; degenerate
  games with continua of equilibria are only represented by their extreme
  points)."""
  g0, g1 = np.asarray(payoffs[0], float), np.asarray(payoffs[1], float)
  found = []
  for i, j in itertools.product(range(2), range(2)):
    if g0[i, j] >= g0[:, j].max() - 1e-12 and g1[i, j] >= g1[i, :].max() - 1e-12:
      s = np.zeros((2, 2))
      s[i, j] = 1.0
      found.append(s)
  d0 = g1[0, 0] - g1[0, 1] - g1[1, 0] + g1[1, 1]  # player 1 indifferent -> p
  d1 = g0[0, 0] - g0[1, 0] - g0[0, 1] + g0[1, 1]  # player 0 indifferent -> q
  if abs(d0) > 1e-12 and abs(d1) > 1e-12:
    p = (g1[1, 1] - g1[1, 0]) / d0
    q = (g0[1, 1] - g0[0, 1]) / d1
    if 1e-9 < p < 1 - 1e-9 and 1e-9 < q < 1 - 1e-9:
      found.append(np.outer([p, 1 - p], [q, 1 - q]))
  return found


def max_entropy_point(payoffs: np.ndarray, mode: str, eps: float = 0.0) -> np.ndarray:
  """Max-entropy point of the eps-(C)CE polytope ([2, 2]); scipy SLSQP, for
  visualisation only (use equilibria_utils.solve_me_eql for the real thing)."""
  rows = constraint_rows(payoffs, mode)
  cons = [
    {"type": "eq", "fun": lambda s: s.sum() - 1.0},
    {"type": "ineq", "fun": lambda s: eps - rows @ s},
  ]
  res = scipy.optimize.minimize(
    lambda s: float(np.sum(s * np.log(s))),
    np.full(4, 0.25),
    bounds=[(1e-12, 1.0)] * 4,
    constraints=cons,
    method="SLSQP",
    options={"maxiter": 500, "ftol": 1e-14},
  )
  return res.x.reshape(2, 2)


def max_welfare_point(payoffs: np.ndarray, mode: str, eps: float = 0.0) -> np.ndarray:
  """Utilitarian-welfare-maximising point of the eps-(C)CE polytope ([2, 2]), by LP."""
  rows = constraint_rows(payoffs, mode)
  welfare = (np.asarray(payoffs[0], float) + np.asarray(payoffs[1], float)).ravel()
  res = scipy.optimize.linprog(
    -welfare,
    A_ub=rows,
    b_ub=np.full(len(rows), eps),
    A_eq=np.ones((1, 4)),
    b_eq=[1.0],
    bounds=[(0, 1)] * 4,
  )
  return res.x.reshape(2, 2)


def _draw_polytope(ax, verts4, color, alpha, label, zorder):
  """Draws a convex polytope given by barycentric vertices (any dimension 0-3)."""
  if len(verts4) == 0:
    return
  pts = to_xyz(verts4)
  centred = pts - pts.mean(axis=0)
  rank = np.linalg.matrix_rank(centred, tol=1e-9)
  kw = dict(color=color, label=label, zorder=zorder)
  if rank == 0:
    ax.scatter(*pts[:1].T, s=70, **kw)
  elif rank == 1:
    d = centred @ np.linalg.svd(centred)[2][0]
    ends = pts[[d.argmin(), d.argmax()]]
    ax.plot(*ends.T, lw=4, **kw)
  else:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if rank == 2:  # planar polygon: order the vertices in the plane
      basis = np.linalg.svd(centred)[2][:2]
      order = scipy.spatial.ConvexHull(centred @ basis.T).vertices
      faces = [pts[order]]
    else:
      hull = scipy.spatial.ConvexHull(pts)
      faces = [pts[s] for s in hull.simplices]
    ax.add_collection3d(
      Poly3DCollection(
        faces,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidths=0.8,
        zorder=zorder,
        label=label,
      )
    )


def plot_polytopes(
  payoffs: np.ndarray,
  *,
  ax=None,
  eps: float = 0.0,
  points: dict[str, np.ndarray] | None = None,
  action_names: tuple[tuple[str, str], tuple[str, str]] = (("0", "1"), ("0", "1")),
  title: str | None = None,
  show_max_entropy: bool = True,
  show_max_welfare: bool = False,
  view: tuple[float, float] = (14.0, 62.0),
):
  """Draws the (epsilon-)CE = CCE polytope, the Nash set and `points`
  ({label: [2, 2] joint strategy}) in the tetrahedron of joint strategies.
  With eps > 0 the exact (eps = 0) polytope is drawn in front of the relaxed one."""
  import matplotlib.pyplot as plt

  if ax is None:
    ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")

  # tetrahedron edges and vertex labels (row action, column action)
  for i, j in itertools.combinations(range(4), 2):
    ax.plot(*_TETRA[[i, j]].T, color="0.35", lw=1.0, zorder=1)
  names = [
    f"({action_names[0][i]}, {action_names[1][j]})"
    for i, j in itertools.product(range(2), range(2))
  ]
  for v, name in zip(_TETRA * 1.2, names):
    ax.text(*v, name, ha="center", va="center", fontsize=9)

  # Nash set: product distributions sigma = p (x) q, a curved surface
  p, q = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
  prod = np.stack([p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)], axis=-1)
  surf = to_xyz(prod.reshape(-1, 4)).reshape(25, 25, 3)
  ax.plot_wireframe(
    surf[..., 0],
    surf[..., 1],
    surf[..., 2],
    color="0.55",
    alpha=0.30,
    linewidth=0.5,
    rstride=4,
    cstride=4,
    zorder=2,
  )

  # CE and CCE coincide for 2x2 games: draw one polytope (and say so); only
  # fall back to two if they ever differ (they can't here; kept as a check).
  ce_eps, cce_eps = (
    polytope_vertices(payoffs, "CE", eps),
    polytope_vertices(payoffs, "CCE", eps),
  )
  same = ce_eps.shape == cce_eps.shape and np.allclose(ce_eps, cce_eps)
  if eps > 0:  # the relaxed set behind, the exact one in front
    _draw_polytope(ax, ce_eps, "tab:cyan", 0.18, f"eps-CE = eps-CCE (eps={eps:g})", 3)
  exact_ce, exact_cce = (
    polytope_vertices(payoffs, "CE", 0.0),
    polytope_vertices(payoffs, "CCE", 0.0),
  )
  if same:
    _draw_polytope(ax, exact_ce, "tab:blue", 0.45, "CE = CCE", 4)
  else:
    _draw_polytope(ax, exact_cce, "tab:orange", 0.22, "CCE", 3)
    _draw_polytope(ax, exact_ce, "tab:blue", 0.45, "CE", 4)

  ne = nash_equilibria(payoffs)
  if ne:
    # hollow rings: mplot3d depth-sorts scatter artists (zorder is ignored), so
    # coincident points (e.g. a unique equilibrium) stay visible only if the
    # larger markers are not filled.
    ax.scatter(
      *to_xyz(np.array([s.ravel() for s in ne])).T,
      marker="D",
      s=120,
      facecolors="none",
      edgecolors="black",
      linewidths=1.8,
      label="Nash",
      depthshade=False,
    )

  markers = []
  if show_max_entropy:
    markers.append(
      (
        "max-entropy (eps-ME-Eql)",
        max_entropy_point(payoffs, "CE", eps),
        "gold",
        "*",
        150,
        True,
      )
    )
  if show_max_welfare:
    markers.append(
      ("max welfare", max_welfare_point(payoffs, "CE", eps), "tab:pink", "P", 330, False)
    )
  palette = ["tab:green", "tab:red", "tab:purple", "tab:brown"]
  for k, (label, s) in enumerate((points or {}).items()):
    markers.append((label, np.asarray(s), palette[k % len(palette)], "X", 120, True))
  for label, s, color, marker, size, filled in markers:
    xyz = to_xyz(np.asarray(s).ravel()).T
    if filled:
      ax.scatter(
        *xyz,
        marker=marker,
        s=size,
        color=color,
        edgecolor="black",
        linewidths=0.8,
        label=label,
        depthshade=False,
      )
    else:
      ax.scatter(
        *xyz,
        marker=marker,
        s=size,
        facecolors="none",
        edgecolors=color,
        linewidths=2.4,
        label=label,
        depthshade=False,
      )

  ax.set_box_aspect((1, 1, 1))
  ax.set_axis_off()
  ax.view_init(elev=view[0], azim=view[1])
  lim = 0.62
  ax.set_xlim(-lim, lim)
  ax.set_ylim(-lim, lim)
  ax.set_zlim(-lim, lim)
  if title:
    ax.set_title(title + (f"  (eps = {eps:g})" if eps else ""), fontsize=11)
  return ax


# Canonical 2x2 games, payoffs[p][row][col].
CANONICAL_GAMES = {
  "Prisoner's dilemma": (
    np.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]], float),
    (("C", "D"), ("C", "D")),
  ),
  "Matching pennies": (
    np.array([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], float),
    (("H", "T"), ("H", "T")),
  ),
  "Battle of the sexes": (
    np.array([[[2, 0], [0, 1]], [[1, 0], [0, 2]]], float),
    (("A", "B"), ("A", "B")),
  ),
  "Chicken": (
    np.array([[[0, -1], [1, -10]], [[0, 1], [-1, -10]]], float),
    (("S", "D"), ("S", "D")),
  ),
}


def plot_canonical_games(
  path: str, eps: float = 0.0, points_per_game=None, **plot_kwargs
) -> None:
  """2x2 figure of the canonical games. `points_per_game`: optional
  {game name: {label: [2, 2] sigma}} to overlay solver outputs."""
  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=(13, 12))
  for k, (name, (g, names)) in enumerate(CANONICAL_GAMES.items()):
    ax = fig.add_subplot(2, 2, k + 1, projection="3d")
    plot_polytopes(
      g,
      ax=ax,
      eps=eps,
      action_names=names,
      title=name,
      points=(points_per_game or {}).get(name),
      **plot_kwargs,
    )
  labels = {}
  for ax in fig.axes:
    for h, ln_ in zip(*ax.get_legend_handles_labels()):
      labels.setdefault(ln_, h)
  fig.legend(
    list(labels.values()),
    list(labels.keys()),
    loc="lower center",
    ncol=min(6, len(labels)),
    frameon=False,
  )
  fig.tight_layout(rect=(0, 0.04, 1, 1))
  fig.savefig(path, dpi=130)
  plt.close(fig)
