"""
Marching Cubes Convergence Study
=================================
Tests whether geometric quantities (volume, surface area, integral mean curvature)
computed from a scikit-image marching_cubes mesh converge to known analytical values
as voxel resolution increases, and whether Gaussian or constrained smoothing
improves the convergence rate. A unit sphere is used as the test shape.

The marching cubes step uses skimage.measure.marching_cubes throughout.
Three smoothing variants are compared:
  binary     – raw binary mask, isovalue 0.5
  gaussian   – mcubes.smooth(method='gaussian'), isovalue 0
  constrained – mcubes.smooth(method='constrained'), isovalue 0
                (constrained is forced explicitly, bypassing the size gate)
"""

import matplotlib
matplotlib.use('Agg')  # headless

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import mcubes
from skimage.measure import marching_cubes as ski_marching_cubes
import pykarambola as pk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAIN_SIZE   = 4.0          # [-2, 2] per axis
SPHERE_RADIUS = 1.0
RESOLUTIONS   = [20, 30, 40, 50, 65, 80, 100]
ANALYTICAL    = {
    'volume':    4.0 * np.pi / 3.0,   # (4/3)π R³, R=1
    'area':      4.0 * np.pi,          # 4π R²
    'curvature': 4.0 * np.pi,          # (1/R) · 4πR² = 4π for R=1
}
METHODS = ['binary', 'gaussian', 'constrained']


# ---------------------------------------------------------------------------
# 1. Build binary sphere mask
# ---------------------------------------------------------------------------
def make_binary_sphere(N):
    """Return (bool_mask [N,N,N], h) for a unit sphere in [-2,2]^3."""
    coords = np.linspace(-2.0, 2.0, N)
    h = coords[1] - coords[0]  # = DOMAIN_SIZE / (N-1)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    mask = (X**2 + Y**2 + Z**2) < SPHERE_RADIUS**2
    return mask, h


# ---------------------------------------------------------------------------
# 2. Scale voxel-space vertices to physical coordinates
# ---------------------------------------------------------------------------
def scale_to_physical(vertices, faces, h):
    """Return (v_phys float64, faces int64), or (None, None) if empty."""
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        return None, None
    v_phys = (vertices * h - 2.0).astype(np.float64)
    return v_phys, faces.astype(np.int64)


# ---------------------------------------------------------------------------
# 3. Extract geometric properties via pykarambola Minkowski functionals
# ---------------------------------------------------------------------------
def get_properties(v_phys, faces, method, N):
    """Return (volume, area, integral_mean_curvature) using pykarambola.

    w000 = volume, w100 = surface area, w200 = integral mean curvature.
    Negative volume signals inward normals; both volume and curvature are
    negated in that case so the returned values are always positive.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pk.minkowski_tensors(
            v_phys, faces,
            compute=['w000', 'w100', 'w200'],
            compute_eigensystems=False,
        )
    for w in caught:
        warnings.warn(f"[N={N}, method={method}] {w.message}")

    volume    = float(result['w000'])
    area      = float(result['w100']) * 3.0   # w100 = A/3
    curvature = float(result['w200']) * 3.0   # w200 = M/3

    # inward normals → flip sign of both volume-dependent quantities
    if np.isfinite(volume) and volume < 0:
        volume    = -volume
        curvature = -curvature

    return (volume    if np.isfinite(volume)    else np.nan,
            area,
            curvature if np.isfinite(curvature) else np.nan)


# ---------------------------------------------------------------------------
# 4. Run convergence test
# ---------------------------------------------------------------------------
def run_convergence_test():
    """Main loop over resolutions × methods.

    Returns a nested dict:
        results[method][metric] = list of (N, h, raw_value, rel_error)
    """
    results = {m: {k: [] for k in ('volume', 'area', 'curvature')} for m in METHODS}

    header = f"{'N':>5} {'h':>8} {'method':>12}  {'vol':>10} {'area':>10} {'curv':>10}  {'t (s)':>7}"
    print(header)
    print('-' * len(header))

    for N in RESOLUTIONS:
        mask, h = make_binary_sphere(N)

        for method in METHODS:
            t0 = time.perf_counter()
            try:
                if method == 'binary':
                    vertices, faces, _, _ = ski_marching_cubes(
                        mask.astype(np.float64), level=0.5)
                elif method == 'gaussian':
                    smoothed = mcubes.smooth(mask, method='gaussian', sigma=3)
                    vertices, faces, _, _ = ski_marching_cubes(smoothed, level=0)
                else:  # constrained — explicitly bypass the size gate
                    smoothed = mcubes.smooth(mask, method='constrained')
                    vertices, faces, _, _ = ski_marching_cubes(smoothed, level=0)

                v_phys, faces_i64 = scale_to_physical(vertices, faces, h)

                if v_phys is None:
                    raise ValueError("Empty mesh returned by marching_cubes")

                vol, area, curv = get_properties(v_phys, faces_i64, method, N)

            except Exception as exc:
                warnings.warn(f"[N={N}, method={method}] Failed: {exc}")
                vol = area = curv = np.nan

            elapsed = time.perf_counter() - t0

            # relative errors
            rel_vol  = abs(vol  - ANALYTICAL['volume'])    / ANALYTICAL['volume']
            rel_area = abs(area - ANALYTICAL['area'])       / ANALYTICAL['area']
            rel_curv = abs(curv - ANALYTICAL['curvature'])  / ANALYTICAL['curvature']

            results[method]['volume'].append((N, h, vol, rel_vol))
            results[method]['area'].append((N, h, area, rel_area))
            results[method]['curvature'].append((N, h, curv, rel_curv))

            print(
                f"{N:>5} {h:>8.4f} {method:>12}  "
                f"{rel_vol:>10.4%} {rel_area:>10.4%} {rel_curv:>10.4%}  "
                f"{elapsed:>7.2f}"
            )

    return results


# ---------------------------------------------------------------------------
# 5. Estimate convergence rate via log-log regression
# ---------------------------------------------------------------------------
def estimate_convergence_rate(resolutions, errors):
    """Log-log linear regression: log(err) = a + b·log(N), order = -b.

    Returns (order, r_squared). Returns (nan, nan) if < 2 valid points.
    """
    Ns  = np.asarray(resolutions, dtype=float)
    err = np.asarray(errors,      dtype=float)

    valid = np.isfinite(err) & (err > 0)
    if valid.sum() < 2:
        return np.nan, np.nan

    log_N   = np.log(Ns[valid])
    log_err = np.log(err[valid])

    b, a = np.polyfit(log_N, log_err, 1)
    # R²
    fitted  = a + b * log_N
    ss_res  = np.sum((log_err - fitted)**2)
    ss_tot  = np.sum((log_err - log_err.mean())**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return -b, r2   # order (positive means convergence)


# ---------------------------------------------------------------------------
# 6. Print summary table
# ---------------------------------------------------------------------------
def print_results_table(results):
    """Print per-property tables of relative errors and convergence orders."""
    Ns = RESOLUTIONS

    for metric in ('volume', 'area', 'curvature'):
        print(f"\n=== {metric.upper()} relative error ===")
        col_w = 9
        header = f"{'method':>12} " + " ".join(f"N={n:>3}" for n in Ns) + f"  {'order':>6}  {'R²':>5}"
        print(header)
        print('-' * len(header))

        for method in METHODS:
            data   = results[method][metric]
            errs   = [row[3] for row in data]
            order, r2 = estimate_convergence_rate(Ns, errs)

            err_str = " ".join(f"{e:>7.3%}" for e in errs)
            order_s = f"{order:>6.2f}" if np.isfinite(order) else f"{'N/A':>6}"
            r2_s    = f"{r2:>5.3f}"  if np.isfinite(r2)    else f"{'N/A':>5}"
            print(f"{method:>12} {err_str}  {order_s}  {r2_s}")


# ---------------------------------------------------------------------------
# 7. Plot convergence
# ---------------------------------------------------------------------------
def _add_convergence_axes(ax, results, metric, title):
    """Populate a single log-log convergence axes for one metric."""
    colors  = {'binary': '#1f77b4', 'gaussian': '#ff7f0e', 'constrained': '#2ca02c'}
    markers = {'binary': 'o',       'gaussian': 's',        'constrained': '^'}

    Ns = np.asarray(RESOLUTIONS, dtype=float)

    for method in METHODS:
        data  = results[method][metric]
        errs  = np.array([row[3] for row in data], dtype=float)
        valid = np.isfinite(errs) & (errs > 0)
        if not valid.any():
            continue
        order, r2 = estimate_convergence_rate(RESOLUTIONS, errs)
        order_lbl = f"{order:.2f}" if np.isfinite(order) else "N/A"
        label = f"{method}  (order {order_lbl}, R²={r2:.3f})" if np.isfinite(r2) \
                else method
        ax.loglog(Ns[valid], errs[valid] * 100,
                  color=colors[method], marker=markers[method],
                  label=label, linewidth=2, markersize=7)

    # reference lines anchored at the median error of the first resolution
    anchor_errs = []
    for method in METHODS:
        data = results[method][metric]
        e = data[0][3]
        if np.isfinite(e) and e > 0:
            anchor_errs.append(e)
    if anchor_errs:
        e0   = np.median(anchor_errs) * 100   # in %
        N0   = Ns[0]
        ref  = np.array([Ns[0], Ns[-1]])
        ax.loglog(ref, e0 * (ref / N0)**(-1), color='gray', linestyle='--',
                  linewidth=1.2, label=r'$O(N^{-1})$')
        ax.loglog(ref, e0 * (ref / N0)**(-2), color='gray', linestyle=':',
                  linewidth=1.2, label=r'$O(N^{-2})$')

    ax.set_xlabel("N  (grid points per axis)", fontsize=11)
    ax.set_ylabel("Relative error  (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, which='both', alpha=0.25)
    ax.tick_params(labelsize=10)


def plot_convergence(results):
    """Save one PDF per quantity + one PDF with the convergence-order bar chart."""
    from matplotlib.backends.backend_pdf import PdfPages

    metric_specs = [
        ('volume',    'Volume',
         rf'Unit sphere: $V = \frac{{4}}{{3}}\pi \approx {ANALYTICAL["volume"]:.5f}$'),
        ('area',      'Surface Area',
         rf'Unit sphere: $A = 4\pi \approx {ANALYTICAL["area"]:.5f}$'),
        ('curvature', 'Integral Mean Curvature',
         rf'Unit sphere: $M = 4\pi \approx {ANALYTICAL["curvature"]:.5f}$'),
    ]

    saved = []

    for metric, title, subtitle in metric_specs:
        fname = f"convergence_{metric}.pdf"
        with PdfPages(fname) as pdf:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            fig.subplots_adjust(top=0.84)
            _add_convergence_axes(ax, results, metric, title)
            fig.text(0.5, 0.93, subtitle, ha='center', fontsize=10, color='#444444')
            fig.suptitle("Marching Cubes Convergence  —  Unit Sphere",
                         fontsize=12, y=1.00)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        saved.append(fname)

    # bar chart of convergence orders
    fname = "convergence_orders.pdf"
    with PdfPages(fname) as pdf:
        colors  = {'binary': '#1f77b4', 'gaussian': '#ff7f0e', 'constrained': '#2ca02c'}
        metric_labels = [('volume', 'Volume'), ('area', 'Surface Area'),
                         ('curvature', 'Mean Curv.')]
        x     = np.arange(len(metric_labels))
        width = 0.24

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, method in enumerate(METHODS):
            orders = []
            for metric, _ in metric_labels:
                errs  = [row[3] for row in results[method][metric]]
                order, _ = estimate_convergence_rate(RESOLUTIONS, errs)
                orders.append(order if np.isfinite(order) else 0.0)
            bars = ax.bar(x + i * width, orders, width, label=method,
                          color=colors[method], alpha=0.85, edgecolor='white')
            for bar, val in zip(bars, orders):
                if val > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.04, f"{val:.2f}",
                            ha='center', va='bottom', fontsize=8)

        ax.axhline(1, color='gray', linestyle='--', linewidth=1.2, label=r'$O(N^{-1})$')
        ax.axhline(2, color='gray', linestyle=':',  linewidth=1.2, label=r'$O(N^{-2})$')
        ax.set_xticks(x + (len(METHODS) - 1) / 2 * width)
        ax.set_xticklabels([lbl for _, lbl in metric_labels], fontsize=11)
        ax.set_ylabel("Estimated convergence order", fontsize=11)
        ax.set_title("Convergence Orders by Method and Metric", fontsize=13,
                     fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.25)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]) - 0.1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    saved.append(fname)

    # 2×2 overview PNG
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Marching Cubes Convergence Study (Unit Sphere)", fontsize=14)
    for (metric, title, _), ax in zip(metric_specs, axes.flat):
        _add_convergence_axes(ax, results, metric, title)

    # bar chart in last panel
    ax_bar = axes[1, 1]
    colors  = {'binary': '#1f77b4', 'gaussian': '#ff7f0e', 'constrained': '#2ca02c'}
    metric_labels = [('volume', 'Volume'), ('area', 'Surface Area'),
                     ('curvature', 'Mean Curv.')]
    x     = np.arange(len(metric_labels))
    width = 0.27
    for i, method in enumerate(METHODS):
        orders = []
        for metric, _ in metric_labels:
            errs  = [row[3] for row in results[method][metric]]
            order, _ = estimate_convergence_rate(RESOLUTIONS, errs)
            orders.append(order if np.isfinite(order) else 0.0)
        bars = ax_bar.bar(x + i * width, orders, width, label=method,
                          color=colors[method], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, orders):
            if val > 0.05:
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.04, f"{val:.2f}",
                            ha='center', va='bottom', fontsize=8)
    ax_bar.axhline(1, color='gray', linestyle='--', linewidth=1.2, label=r'$O(N^{-1})$')
    ax_bar.axhline(2, color='gray', linestyle=':',  linewidth=1.2, label=r'$O(N^{-2})$')
    ax_bar.set_xticks(x + (len(METHODS) - 1) / 2 * width)
    ax_bar.set_xticklabels([lbl for _, lbl in metric_labels], fontsize=10)
    ax_bar.set_ylabel("Estimated convergence order", fontsize=10)
    ax_bar.set_title("Convergence Orders", fontsize=12, fontweight='bold')
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    png_file = "convergence_plot.png"
    fig.savefig(png_file, dpi=150)
    plt.close(fig)

    print("\nPDFs saved:")
    for f in saved:
        print(f"  {f}")
    print(f"  {png_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("Marching Cubes Convergence Study")
    print(f"Analytical targets: {ANALYTICAL}")
    print("=" * 60)

    results = run_convergence_test()
    print_results_table(results)
    plot_convergence(results)
