#!/opt/homebrew/bin/python3
# Created: Jun, 17, 2026 14:52:40 by Wataru Fukuda

import KratosMultiphysics as KM

def make_knots(vals):
    v = KM.Vector(len(vals))
    for i, x in enumerate(vals):
        v[i] = x
    return v

def greville(knots, p):
    n = len(knots) - p + 1
    return [sum(knots[i + s] for s in range(p)) / p for i in range(n)]

def unique_spans(knots):
    spans = []
    for i in range(len(knots) - 1):
        if knots[i + 1] - knots[i] > 1e-10:
            spans.append((float(knots[i]), float(knots[i + 1])))
    return spans

def in_ref_domain(mid_u, mid_v, level, ref_domains):
    for d in ref_domains:
        if d.Level == level and d.MinU <= mid_u <= d.MaxU and d.MinV <= mid_v <= d.MaxV:
            return True
    return False

def draw_active_element_borders_2d(ax, geom, level_breaks, ref_domains, min_level=0):
    """Draw active-level cell borders for levels >= min_level."""
    n_lvls = geom.NumberOfLevels()
    for l, (ku, kv) in enumerate(level_breaks):
        if l < min_level:
            continue
        for su in unique_spans(ku):
            for sv in unique_spans(kv):
                mid_u = 0.5 * (su[0] + su[1])
                mid_v = 0.5 * (sv[0] + sv[1])
                in_this = (l == 0) or in_ref_domain(mid_u, mid_v, l, ref_domains)
                in_next = (l + 1 < n_lvls) and in_ref_domain(mid_u, mid_v, l + 1, ref_domains)
                if in_this and not in_next:
                    rect_u = [su[0], su[1], su[1], su[0], su[0]]
                    rect_v = [sv[0], sv[0], sv[1], sv[1], sv[0]]
                    ax.plot(rect_u, rect_v, color='black', lw=0.7, zorder=2)
    if min_level == 0:
        # Outer domain boundary (only when showing the full hierarchy)
        ku0, kv0 = level_breaks[0]
        outer_u = [ku0[0], ku0[-1], ku0[-1], ku0[0], ku0[0]]
        outer_v = [kv0[0], kv0[0], kv0[-1], kv0[-1], kv0[0]]
        ax.plot(outer_u, outer_v, color='black', lw=0.7, zorder=3)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="""\
"""
    )
    parser.add_argument("--outdir", default="fig",  help="output directory")
    parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
    options = parser.parse_args()
    import os
    import numpy

    ### Model Setup ###
    model = KM.Model()
    mp = model.CreateModelPart("main")
    p = q = 2

    knots_u = make_knots([0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 8.0])
    knots_v = make_knots([0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 8.0])

    grev_u = greville(list(knots_u), p)
    grev_v = greville(list(knots_v), q)

    node_id = 1
    points = KM.NodesVector()
    for iv, v in enumerate(grev_v):
        for iu, u in enumerate(grev_u):
            points.append(mp.CreateNewNode(node_id, u, v, 0.0))  # Flat patch: x=u, y=v, z=0
            node_id += 1

    geom = KM.THBSurfaceGeometry3D(points, p, q, knots_u, knots_v)

    ### Refinement ###
    geom.AddLevel(3, mp)
    geom.AddRefinementDomain(1, 1.0, 6.0, 2.0, 7.0)
    geom.AddRefinementDomain(2, 2.0, 5.0, 3.0, 5.0)
    geom.AddRefinementDomain(2, 3.0, 5.0, 2.0, 3.0)
    geom.AddRefinementDomain(3, 3.0, 4.0, 3.0, 4.0)
    geom.AddRefinementDomain(3, 3.5, 5.0, 2.0, 3.5)
    geom.EliminateInactiveFunctions(mp)
    num_active_cps = geom.PointsNumber()

    ### Dense grid evaluation ###
    u_lo, u_hi = float(knots_u[0]), float(knots_u[len(knots_u) - 1])
    v_lo, v_hi = float(knots_v[0]), float(knots_v[len(knots_v) - 1])

    n_eval = 100
    u_vals = numpy.linspace(u_lo, u_hi - 1e-4, n_eval)
    v_vals = numpy.linspace(v_lo, v_hi - 1e-4, n_eval)
    U, V = numpy.meshgrid(u_vals, v_vals)   # both (n_eval, n_eval)

    uv = KM.Matrix(n_eval * n_eval, 2)
    for j in range(n_eval):
        for i in range(n_eval):
            idx = j * n_eval + i
            uv[idx, 0] = u_vals[i]
            uv[idx, 1] = v_vals[j]

    N_mat = geom.EvaluateShapeFunctions(uv)
    N_np = numpy.array([[N_mat[r, c] for c in range(num_active_cps)] for r in range(n_eval * n_eval)])

    ### Level assignment and support rectangles for packed CPs ###
    cp_levels = []
    cp_supports = []   # (supp_min_u, supp_max_u, supp_min_v, supp_max_v) per packed CP
    level_breaks = []   # Unique knot breaks per level -> element boundary lines
    for l in range(geom.NumberOfLevels()):
        lvl_data = geom.Levels()[l]
        ku = list(lvl_data.KnotsU)
        kv = list(lvl_data.KnotsV)
        nu = len(ku) - p + 1
        nv = len(kv) - q + 1
        flags = geom.GetActiveFunctions(l)
        for i_v in range(nv):
            for i_u in range(nu):
                if not flags[i_v * nu + i_u]:
                    continue
                cp_levels.append(l)
                supp_min_u = ku[0] if i_u == 0 else ku[i_u - 1]
                supp_max_u = ku[min(i_u + p, len(ku) - 1)]
                supp_min_v = kv[0] if i_v == 0 else kv[i_v - 1]
                supp_max_v = kv[min(i_v + q, len(kv) - 1)]
                cp_supports.append((supp_min_u, supp_max_u, supp_min_v, supp_max_v))
        level_breaks.append((sorted(set(lvl_data.KnotsU)), sorted(set(lvl_data.KnotsV))))

    if options.debug:
        print("=== Construction ===")
        print(f"Levels           : {geom.NumberOfLevels()}")
        print(f"Polynomial degree: p={geom.PolynomialDegree(0)}, q={geom.PolynomialDegree(1)}")
        for i, lvl in enumerate(geom.Levels()):
            nu = len(lvl.KnotsU) - p + 1
            nv = len(lvl.KnotsV) - q + 1
            print(f"  Level {i}: {nu}×{nv} = {nu*nv} CPs")
        print("\nActive functions per level (before elimination):")
        for l in range(geom.NumberOfLevels()):
            print(f"  Level {l}: {geom.NumberOfActiveFunctions(l)} active")
        print(f"\nAfter elimination: {num_active_cps} active CPs")
        pou_err = numpy.max(numpy.abs(N_np.sum(axis=1) - 1.0))
        print(f"Max POU error: {pou_err:.2e}  ({'PASS' if pou_err < 1e-10 else 'FAIL'})")
        print(f"\nShape function matrix: {N_mat.Size1()} × {N_mat.Size2()}\n")
        for l, (ku, kv) in enumerate(level_breaks):
            print(f"  level_breaks[{l}]: u={ku}, v={kv}")

    ### 2D top-view: contour lines per active CP ###
    import matplotlib.pyplot as plt
    import matplotlib.cm as mpl_cm
    import matplotlib.colors as mpl_colors

    cmap_lines = mpl_cm.get_cmap('rainbow')
    global_norm = mpl_colors.Normalize(vmin=0.0, vmax=1.0)

    ref_domains = list(geom.RefinementDomains())

    # Bounding box of all refinement domains — used as viewport for level-1+ figures
    ref_u_lo = min(d.MinU for d in ref_domains)
    ref_u_hi = max(d.MaxU for d in ref_domains)
    ref_v_lo = min(d.MinV for d in ref_domains)
    ref_v_hi = max(d.MaxV for d in ref_domains)
    ref_margin = max(ref_u_hi - ref_u_lo, ref_v_hi - ref_v_lo) * 0.05

    # Only visualize shape functions from level 1 and finer
    fine_cp_indices = [j for j in range(num_active_cps) if cp_levels[j] >= 1]
    num_fine_cps = len(fine_cp_indices)

    def draw_single_cp(ax, Z_masked, supp_min_u, supp_max_u, supp_min_v, supp_max_v,
                       lvl, local_idx, title_fontsize=7):
        draw_active_element_borders_2d(ax, geom, level_breaks, ref_domains, min_level=1)
        z_min = float(Z_masked.min())
        z_max = float(Z_masked.max())
        levels_local = numpy.linspace(z_min, z_max, 10)
        ax.contour(U, V, Z_masked, levels=levels_local,
                   cmap=cmap_lines, norm=global_norm, linewidths=1.5, zorder=2)
        rect_u = [supp_min_u, supp_max_u, supp_max_u, supp_min_u, supp_min_u]
        rect_v = [supp_min_v, supp_min_v, supp_max_v, supp_max_v, supp_min_v]
        ax.plot(rect_u, rect_v, color='blue', lw=2.0, zorder=4)
        ax.set_title(f'L{lvl}-N{local_idx}', fontsize=title_fontsize, pad=2)
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)
        ax.set_xlim(ref_u_lo - ref_margin, ref_u_hi + ref_margin)
        ax.set_ylim(ref_v_lo - ref_margin, ref_v_hi + ref_margin)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    single_dir = os.path.join(options.outdir, 'shape_function')
    os.makedirs(single_dir, exist_ok=True)

    ncols = 6
    nrows = (num_fine_cps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols + 0.6, 1.8 * nrows), squeeze=False)
    fig.subplots_adjust(right=0.88)
    level_counts = [0] * geom.NumberOfLevels()

    for plot_idx, j in enumerate(fine_cp_indices):
        r, c = divmod(plot_idx, ncols)
        lvl = cp_levels[j]
        local_idx = level_counts[lvl]
        level_counts[lvl] += 1

        supp_min_u, supp_max_u, supp_min_v, supp_max_v = cp_supports[j]
        mask = (U < supp_min_u - 1e-10) | (U > supp_max_u + 1e-10) \
             | (V < supp_min_v - 1e-10) | (V > supp_max_v + 1e-10)
        Z = N_np[:, j].reshape(n_eval, n_eval)
        Z_masked = numpy.ma.masked_where(mask, Z)

        # Draw into the grid subplot
        draw_single_cp(axes[r][c], Z_masked,
                       supp_min_u, supp_max_u, supp_min_v, supp_max_v, lvl, local_idx)

        # Save individual figure
        fig_s, ax_s = plt.subplots(figsize=(3.5, 3.5))
        draw_single_cp(ax_s, Z_masked,
                       supp_min_u, supp_max_u, supp_min_v, supp_max_v, lvl, local_idx,
                       title_fontsize=10)
        fn_single = os.path.join(single_dir, f'L{lvl}-N{local_idx}.png')
        fig_s.savefig(fn_single, dpi=150, bbox_inches='tight')
        plt.close(fig_s)

    # Hide unused subplots
    for j in range(num_fine_cps, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    plt.suptitle("THB shape functions — top view", fontsize=11)
    filename = os.path.join(options.outdir, 'thb_surface_shapes_2d.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")

    ### Active knot spans + control points per level (level 1 and finer only) ###
    level_cp_colors = ['red', 'orange', 'green', 'purple', 'cyan', 'brown']
    fig2, ax2 = plt.subplots(figsize=(5.0, 5.0))
    draw_active_element_borders_2d(ax2, geom, level_breaks, ref_domains, min_level=1)

    # Collect CP (u, v) per level from packed nodes, skip level 0
    cp_coords_by_level = [[] for _ in range(geom.NumberOfLevels())]
    for j in range(num_active_cps):
        if cp_levels[j] == 0:
            continue
        node = geom[j]
        cp_coords_by_level[cp_levels[j]].append((node.X, node.Y))

    for l, coords in enumerate(cp_coords_by_level):
        if not coords:
            continue
        xs, ys = zip(*coords)
        color = level_cp_colors[(l - 1) % len(level_cp_colors)]
        ax2.scatter(xs, ys, color=color, s=30, zorder=5, label=f'Level {l}')

    ax2.set_xlim(ref_u_lo - ref_margin, ref_u_hi + ref_margin)
    ax2.set_ylim(ref_v_lo - ref_margin, ref_v_hi + ref_margin)
    ax2.set_aspect('equal')
    ax2.set_xlabel('u')
    ax2.set_ylabel('v')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.set_title('Active knot spans and control points', fontsize=11)

    filename2 = os.path.join(options.outdir, 'thb_mesh_and_cps.png')
    plt.savefig(filename2, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename2}")

    if options.debug:
        plt.show()


if __name__ == "__main__":
    main()

