#!/usr/bin/env python3
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

def draw_active_element_borders_2d(ax, geom, level_breaks, ref_domains):
    """Draw only active-level cell borders (cells not covered by a finer level)."""
    n_lvls = geom.NumberOfLevels()
    for l, (ku, kv) in enumerate(level_breaks):
        for su in unique_spans(ku):
            for sv in unique_spans(kv):
                mid_u = 0.5 * (su[0] + su[1])
                mid_v = 0.5 * (sv[0] + sv[1])
                in_this = (l == 0) or in_ref_domain(mid_u, mid_v, l, ref_domains)
                in_next = (l + 1 < n_lvls) and in_ref_domain(mid_u, mid_v, l + 1, ref_domains)
                if in_this and not in_next:
                    rect_u = [su[0], su[1], su[1], su[0], su[0]]
                    rect_v = [sv[0], sv[0], sv[1], sv[1], sv[0]]
                    ax.plot(rect_u, rect_v, color='gray', lw=0.7, zorder=2)
    # Outer domain boundary
    ku0, kv0 = level_breaks[0]
    outer_u = [ku0[0], ku0[-1], ku0[-1], ku0[0], ku0[0]]
    outer_v = [kv0[0], kv0[0], kv0[-1], kv0[-1], kv0[0]]
    ax.plot(outer_u, outer_v, color='gray', lw=0.7, zorder=3)


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

    n_eval = 60
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

    ncols = 6
    nrows = (num_active_cps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols + 0.6, 1.8 * nrows), squeeze=False)
    fig.subplots_adjust(right=0.88)
    level_counts = [0] * geom.NumberOfLevels()

    ref_domains = list(geom.RefinementDomains())

    for j in range(num_active_cps):
        r, c = divmod(j, ncols)
        ax = axes[r][c]
        lvl = cp_levels[j]
        local_idx = level_counts[lvl]
        level_counts[lvl] += 1

        supp_min_u, supp_max_u, supp_min_v, supp_max_v = cp_supports[j]
        mask = (U < supp_min_u - 1e-10) | (U > supp_max_u + 1e-10) \
             | (V < supp_min_v - 1e-10) | (V > supp_max_v + 1e-10)
        Z = N_np[:, j].reshape(n_eval, n_eval)
        Z_masked = numpy.ma.masked_where(mask, Z)

        draw_active_element_borders_2d(ax, geom, level_breaks, ref_domains)

        # 10 levels spanning this function's actual value range
        z_min = float(Z_masked.min())
        z_max = float(Z_masked.max())
        levels_local = numpy.linspace(z_min, z_max, 10)

        # Contour lines colored by global [0,1] norm so all subplots share the same scale
        ax.contour(U, V, Z_masked, levels=levels_local, cmap=cmap_lines, norm=global_norm, linewidths=1.0, zorder=2)

        # Support rectangle -- bold blue (drawn on top of grid)
        rect_u = [supp_min_u, supp_max_u, supp_max_u, supp_min_u, supp_min_u]
        rect_v = [supp_min_v, supp_min_v, supp_max_v, supp_max_v, supp_min_v]
        ax.plot(rect_u, rect_v, color='blue', lw=1.0, zorder=4)

        ax.set_title(f'L{lvl}-N{local_idx}', fontsize=7, pad=2)
        margin = (u_hi - u_lo) * 0.05
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)
        ax.set_xlim(u_lo - margin, u_hi + margin)
        ax.set_ylim(v_lo - margin, v_hi + margin)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for j in range(num_active_cps, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    plt.suptitle("THB shape functions — top view", fontsize=11)
    filename = os.path.join(options.outdir, 'thb_surface_shapes_2d.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")

    ### Active knot spans + control points per level ###
    level_cp_colors = ['red', 'orange', 'green', 'purple', 'cyan', 'brown']
    fig2, ax2 = plt.subplots(figsize=(5.0, 5.0))
    draw_active_element_borders_2d(ax2, geom, level_breaks, ref_domains)

    # Collect CP (u, v) per level from packed nodes
    cp_coords_by_level = [[] for _ in range(geom.NumberOfLevels())]
    for j in range(num_active_cps):
        node = geom[j]
        cp_coords_by_level[cp_levels[j]].append((node.X, node.Y))

    for l, coords in enumerate(cp_coords_by_level):
        if not coords:
            continue
        xs, ys = zip(*coords)
        color = level_cp_colors[l % len(level_cp_colors)]
        ax2.scatter(xs, ys, color=color, s=30, zorder=5, label=f'Level {l}')

    margin = (u_hi - u_lo) * 0.05
    ax2.set_xlim(u_lo - margin, u_hi + margin)
    ax2.set_ylim(v_lo - margin, v_hi + margin)
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

