"""Plain matplotlib style for the paper figures, legend and label placement
that avoids the data, and an overlap check that runs before every figure is
saved.

The style is matplotlib's own default (full axes frame, no grid, framed
legend) with a smaller font so that the figures fit a page. Placement and
checking work in display coordinates after the figure has been drawn:

  place_legend(ax)      put the legend where it covers no data point, line,
                        bar or error bar; add headroom if no such place exists
  place_labels(ax, ..)  put point labels where they cover no data and no
                        other label
  check(fig, name)      raise if any legend or text overlaps data, another
                        text, or a neighbouring axes
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon
from matplotlib.text import Annotation

STYLE = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.8,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 200, "savefig.bbox": "tight",
         "savefig.pad_inches": 0.03, "legend.framealpha": 1.0, "errorbar.capsize": 2.5}


def apply():
    matplotlib.rcdefaults()
    plt.rcParams.update(STYLE)


# ----------------------------------------------------------------- geometry
def _render(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _is_data(artist, ax):
    """True when the artist is positioned in data coordinates on both axes
    (excludes axhline, axvline, axhspan and axvspan, which span the axes)."""
    try:
        return artist.get_transform().contains_branch(ax.transData) and \
            not any(artist.get_transform() == t for t in (ax.get_xaxis_transform(), ax.get_yaxis_transform()))
    except Exception:
        return False


def _spans_axes(bb, ax_bb):
    return bb.width >= 0.98 * ax_bb.width or bb.height >= 0.98 * ax_bb.height


def _sample(xy, step=1.5):
    """Points every `step` pixels along a polyline given in display coordinates."""
    out = []
    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
        if not np.all(np.isfinite([x0, y0, x1, y1])):
            continue
        n = max(1, int(np.hypot(x1 - x0, y1 - y0) / step))
        t = np.linspace(0, 1, n + 1)
        out.append(np.c_[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])
    return np.vstack(out) if out else np.zeros((0, 2))


def data_boxes(ax, renderer, skip=()):
    """Every visible data mark of `ax` as boxes (x0, y0, x1, y1) in pixels."""
    ax_bb = ax.get_window_extent(renderer)
    boxes = []
    dpi = ax.figure.dpi

    def pts_box(P, r):
        for x, y in P:
            if ax_bb.x0 - r <= x <= ax_bb.x1 + r and ax_bb.y0 - r <= y <= ax_bb.y1 + r:
                boxes.append((x - r, y - r, x + r, y + r))

    for ln in ax.lines:
        if ln in skip or not ln.get_visible():
            continue           # reference lines (axhline, axvline) count as data too
        P = ln.get_transform().transform(np.c_[ln.get_xdata(orig=False), ln.get_ydata(orig=False)].astype(float))
        if ln.get_marker() not in (None, "None", "", " ") and ln.get_markersize() > 0:
            pts_box(P, ln.get_markersize() * dpi / 72 / 2)
        if ln.get_linestyle() not in ("None", "", " ") and len(P) > 1:
            pts_box(_sample(P), max(ln.get_linewidth() * dpi / 72 / 2, 1))
    for c in ax.collections:
        if c in skip or not c.get_visible():
            continue
        if isinstance(c, LineCollection):
            tr = c.get_transform()
            for seg in c.get_segments():
                P = tr.transform(np.asarray(seg, float))
                bb_w = np.ptp(P[:, 0]) if len(P) else 0
                if bb_w >= 0.98 * ax_bb.width:
                    continue
                pts_box(_sample(P), max(np.max(c.get_linewidths()) * dpi / 72 / 2, 1))
        elif isinstance(c, PathCollection):
            P = c.get_offset_transform().transform(np.asarray(c.get_offsets(), float))
            s = np.asarray(c.get_sizes(), float)
            r = (np.sqrt(s.max()) if len(s) else 6) * dpi / 72 / 2
            pts_box(P, r)
    for p in ax.patches:
        if p in skip or not p.get_visible() or not isinstance(p, (Rectangle, Polygon)):
            continue
        bb = p.get_window_extent(renderer)
        if _spans_axes(bb, ax_bb) or not _is_data(p, ax):
            continue
        boxes.append((bb.x0, bb.y0, bb.x1, bb.y1))
    for t in ax.texts:
        if isinstance(t, Annotation) and t.arrow_patch is not None and t not in skip:
            path = t.arrow_patch.get_path()
            pts_box(_sample(path.vertices), 1)
    return boxes


def _overlap(a, b, tol=0.5):
    return a[0] + tol < b[2] and b[0] + tol < a[2] and a[1] + tol < b[3] and b[1] + tol < a[3]


def _bb(artist, renderer, pad=0.0):
    bb = artist.get_window_extent(renderer)
    return (bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


def _hits(box, boxes):
    return sum(_overlap(box, b) for b in boxes)


def _poly(text, renderer):
    """Corners of a text's own (possibly rotated) rectangle, in pixels. The
    axis-aligned extent of a rotated label overstates its footprint, so the
    rectangle is measured unrotated and turned about the centre of that extent."""
    bb = text.get_window_extent(renderer)
    ang = text.get_rotation() % 180
    if ang == 0:
        return np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]])
    text.set_rotation(0); b0 = text.get_window_extent(renderer); text.set_rotation(ang)
    w, h = b0.width, b0.height
    c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
    th = np.deg2rad(ang); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return corners @ R.T + c


def _box_poly(b):
    return np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])


def _poly_overlap(P, Q, tol=0.8):
    """Separating-axis test for two convex polygons, with a tolerance in pixels."""
    for S in (P, Q):
        for i in range(len(S)):
            e = S[(i + 1) % len(S)] - S[i]
            n = np.array([-e[1], e[0]]); nn = np.hypot(*n)
            if nn == 0:
                continue
            n = n / nn
            a, b = P @ n, Q @ n
            if a.max() - tol <= b.min() or b.max() - tol <= a.min():
                return False
    return True


# --------------------------------------------------------------- placement
LOCS = ["upper left", "upper right", "lower left", "lower right", "upper center", "lower center",
        "center left", "center right"]


def place_legend(ax, handles=None, labels=None, max_headroom=4, **kw):
    """Inside the axes, where it covers no data. If no location is free,
    raise the upper limit step by step; as a last resort, outside on the right."""
    fig = ax.figure
    args = (handles, labels) if handles is not None else ()
    for step in range(max_headroom + 1):
        r = _render(fig)
        data = data_boxes(ax, r) + [_bb(t, r, 1) for t in ax.texts if t.get_text()]
        ax_bb = _bb(ax, r)
        for loc in LOCS:
            leg = ax.legend(*args, loc=loc, **kw)
            r = _render(fig)
            lb = _bb(leg, r, 1.5)
            inside = lb[0] >= ax_bb[0] and lb[2] <= ax_bb[2] and lb[1] >= ax_bb[1] and lb[3] <= ax_bb[3]
            if inside and not _hits(lb, data):
                return leg
        lo, hi = ax.get_ylim()
        if ax.get_yscale() == "log":
            ax.set_ylim(lo, hi * (hi / lo) ** 0.18)
        else:
            ax.set_ylim(lo, hi + 0.18 * (hi - lo))
    return ax.legend(*args, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, **kw)


OFFSETS = [(4, 3, "left", "bottom"), (4, -3, "left", "top"), (-4, 3, "right", "bottom"), (-4, -3, "right", "top"),
           (0, 5, "center", "bottom"), (0, -5, "center", "top"), (6, 0, "left", "center"), (-6, 0, "right", "center"),
           (8, 8, "left", "bottom"), (8, -8, "left", "top"), (-8, 8, "right", "bottom"), (-8, -8, "right", "top"),
           (0, 10, "center", "bottom"), (0, -10, "center", "top"), (12, 0, "left", "center"), (-12, 0, "right", "center"),
           (14, 14, "left", "bottom"), (14, -14, "left", "top"), (-14, 14, "right", "bottom"), (-14, -14, "right", "top"),
           (0, 18, "center", "bottom"), (0, -18, "center", "top"), (20, 0, "left", "center"), (-20, 0, "right", "center")]


def place_labels(ax, items, fontsize=6, tries=4, **kw):
    """items: (x, y, text) in data coordinates. Each label goes to the first
    offset where it covers no data mark, no other label and no legend, and
    stays inside the axes. If some label finds no free place, the y range is
    widened a little and the labels are placed again."""
    for attempt in range(tries + 1):
        placed, worst = _place_once(ax, items, fontsize, **kw)
        if worst == 0 or attempt == tries:
            return placed
        for p_ in placed:
            p_.remove()
        lo, hi = ax.get_ylim()
        if ax.get_yscale() == "log":
            f_ = (hi / lo) ** 0.06; ax.set_ylim(lo / f_, hi * f_)
        else:
            d = 0.06 * (hi - lo); ax.set_ylim(lo - d, hi + d)
        lo, hi = ax.get_xlim()
        if ax.get_xscale() == "log":
            f_ = (hi / lo) ** 0.04; ax.set_xlim(lo / f_, hi * f_)
        else:
            d = 0.04 * (hi - lo); ax.set_xlim(lo - d, hi + d)


def _place_once(ax, items, fontsize, **kw):
    fig = ax.figure
    placed, worst = [], 0
    for x, y, s in items:
        best, best_score = None, None
        for dx, dy, ha, va in OFFSETS:
            t = ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize, **kw)
            r = _render(fig)
            tb = _bb(t, r, 1)
            others = data_boxes(ax, r, skip=(t,)) + [_bb(p, r, 1) for p in placed]
            if ax.get_legend() is not None:
                others.append(_bb(ax.get_legend(), r, 1))
            ab = _bb(ax, r)
            inside = tb[0] >= ab[0] and tb[2] <= ab[2] and tb[1] >= ab[1] and tb[3] <= ab[3]
            score = _hits(tb, others) + (0 if inside else 100)
            if best_score is None or score < best_score:
                if best is not None:
                    best.remove()
                best, best_score = t, score
            else:
                t.remove()
            if score == 0:
                break
        placed.append(best); worst = max(worst, best_score)
    return placed, worst


# ------------------------------------------------------------------- check
def check(fig, name="figure"):
    """Raise AssertionError when any legend or text overlaps data, another
    text, or another axes."""
    r = _render(fig)
    problems = []
    texts = []          # (owner axes or None, box, label)
    axes = [a for a in fig.axes if a.get_visible()]
    for ax in axes:
        data = data_boxes(ax, r)
        own = []
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            lb = _bb(leg, r)
            n = _hits(lb, data)
            if n:
                problems.append(f"{name}: legend of '{ax.get_title()}' covers {n} data marks")
            own.append((lb, "legend"))
        for t in ax.texts:
            if not t.get_text() or not t.get_visible():
                continue
            tb = _bb(t, r)
            skip = [t] if isinstance(t, Annotation) else []
            d = data_boxes(ax, r, skip=skip)
            n = _hits(tb, d)
            if n:
                problems.append(f"{name}: text '{t.get_text()}' in '{ax.get_title()}' covers {n} data marks")
            ab = _bb(ax, r)
            if tb[0] < ab[0] - 1 or tb[2] > ab[2] + 1 or tb[1] < ab[1] - 1 or tb[3] > ab[3] + 1:
                problems.append(f"{name}: text '{t.get_text()}' in '{ax.get_title()}' leaves the axes")
            own.append((tb, f"text '{t.get_text()}'"))
        abx = _bb(ax, r)
        drawn = [tl for tl in (ax.get_xticklabels() if ax.xaxis.get_visible() else [])
                 if abx[0] - 1 <= sum(_bb(tl, r)[0::2]) / 2 <= abx[2] + 1] + \
                [tl for tl in (ax.get_yticklabels() if ax.yaxis.get_visible() else [])
                 if abx[1] - 1 <= sum(_bb(tl, r)[1::2]) / 2 <= abx[3] + 1]
        for tl in drawn:     # labels of ticks outside the view limits are not drawn
            if tl.get_visible() and tl.get_text():
                texts.append((ax, _poly(tl, r), f"tick '{tl.get_text()}'"))
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if lab.get_visible() and lab.get_text():
                texts.append((ax, _poly(lab, r), f"'{lab.get_text()}'"))
        for b, s in own:
            texts.append((ax, _box_poly(b), s))
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (a1, b1, s1), (a2, b2, s2) = texts[i], texts[j]
            if _poly_overlap(b1, b2):
                problems.append(f"{name}: {s1} overlaps {s2}")
    # text of one axes inside the frame of another (twin axes share a frame)
    for a1, b1, s1 in texts:
        for ax in axes:
            if ax is a1 or ax.bbox.bounds == a1.bbox.bounds:
                continue
            if _poly_overlap(b1, _box_poly(_bb(ax, r))):
                problems.append(f"{name}: {s1} reaches into another panel")
    if problems:
        raise AssertionError("figure overlap check failed:\n  " + "\n  ".join(dict.fromkeys(problems)))
    return True


def save(path, fig=None, **kw):
    fig = fig or plt.gcf()
    fig.tight_layout()
    check(fig, path)
    fig.savefig(path, **kw)
