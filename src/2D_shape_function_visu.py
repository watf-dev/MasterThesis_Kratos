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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="""\
THBSurfaceGeometry3D — shape function value distribution on a 3D plot.

Geometry: degree-(2,2) patch on [0,1]×[0,1].
Refinement: level-1 region on [0.5,1]×[0.5,1].
Each active THB basis function is plotted as a surface z = N_j(u,v).
"""
    )
    parser.add_argument("--outdir", default="fig",  help="output directory")
    parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
    options = parser.parse_args()
    import os
    import numpy
    import matplotlib.pyplot as plt

    ### Model Setup ###
    model = KM.Model()
    mp = model.CreateModelPart("main")
    p = q = 2
    knots_u = make_knots([0.0, 0.0, 0.5, 1.0, 1.0])  # 4 CPs in u
    knots_v = make_knots([0.0, 0.0, 0.5, 1.0, 1.0])  # 4 CPs in v
    grev_u = greville(list(knots_u), p)
    grev_v = greville(list(knots_v), q)
    n_u0, n_v0 = len(grev_u), len(grev_v)   # 4, 4  → 16 CPs

    node_id = 1
    points = KM.NodesVector()
    for iv, v in enumerate(grev_v):
        for iu, u in enumerate(grev_u):
            points.append(mp.CreateNewNode(node_id, u, v, 0.0))  # Flat patch: x=u, y=v, z=0
            node_id += 1

    geom = KM.THBSurfaceGeometry3D(points, p, q, knots_u, knots_v)

    ### Refinement ###
    geom.AddLevel(1, mp)
    geom.AddRefinementDomain(1, 0.5, 1.0, 0.5, 1.0)

    geom.EliminateInactiveFunctions(mp)
    n_active = geom.PointsNumber()

    ### Dense grid evaluation ###
    n_eval = 60
    u_vals = numpy.linspace(0.0, 0.9999, n_eval)
    v_vals = numpy.linspace(0.0, 0.9999, n_eval)
    U, V = numpy.meshgrid(u_vals, v_vals)   # both (n_eval, n_eval)

    uv = KM.Matrix(n_eval * n_eval, 2)
    for j in range(n_eval):
        for i in range(n_eval):
            idx = j * n_eval + i
            uv[idx, 0] = u_vals[i]
            uv[idx, 1] = v_vals[j]

    N_mat = geom.EvaluateShapeFunctions(uv)
    N_np = numpy.array([[N_mat[r, c] for c in range(n_active)] for r in range(n_eval * n_eval)])

    pou_err = numpy.max(numpy.abs(N_np.sum(axis=1) - 1.0))
    if options.debug:
        print(f"Max POU error: {pou_err:.2e}  ({'PASS' if pou_err < 1e-10 else 'FAIL'})")

    ### Level assignment and support rectangles for packed CPs ###
    cp_levels = []
    cp_supports = []   # (smin_u, smax_u, smin_v, smax_v) per packed CP
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
                smin_u = ku[0]          if i_u == 0 else ku[i_u - 1]
                smax_u = ku[min(i_u + p, len(ku) - 1)]
                smin_v = kv[0]          if i_v == 0 else kv[i_v - 1]
                smax_v = kv[min(i_v + q, len(kv) - 1)]
                cp_supports.append((smin_u, smax_u, smin_v, smax_v))

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
        print(f"\nAfter elimination: {n_active} active CPs")
        print(f"\nShape function matrix: {N_mat.Size1()} × {N_mat.Size2()}\n")

    from matplotlib.patches import Patch

    def plot_shape_on_ax(ax, j, fontsize_title=7, fontsize_label=6, fontsize_tick=5):
        lvl = cp_levels[j]
        smin_u, smax_u, smin_v, smax_v = cp_supports[j]

        # Clip grid to support domain
        mask = (U < smin_u - 1e-10) | (U > smax_u + 1e-10) \
             | (V < smin_v - 1e-10) | (V > smax_v + 1e-10)
        Z = N_np[:, j].reshape(n_eval, n_eval)
        Z_masked = numpy.ma.masked_where(mask, Z)

        color = level_colors[lvl % len(level_colors)]
        ax.plot_surface(U, V, Z_masked, color=color, alpha=0.75,
                        linewidth=0, antialiased=False)

        # Support rectangle outline on the base plane (z=0)
        rect_u = [smin_u, smax_u, smax_u, smin_u, smin_u]
        rect_v = [smin_v, smin_v, smax_v, smax_v, smin_v]
        ax.plot(rect_u, rect_v, [0]*5, color='k', linewidth=0.8, zorder=5)

        return color

    ### 3D surface plot --- one subplot per active CP ###
    level_colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']
    ncols = 4
    nrows = (n_active + ncols - 1) // ncols

    fig = plt.figure(figsize=(2.0 * ncols, 1.7 * nrows))
    axes_grid = []
    level_counts = [0] * geom.NumberOfLevels()

    for j in range(n_active):
        ax = fig.add_subplot(nrows, ncols, j + 1, projection='3d')
        axes_grid.append(ax)
        lvl = cp_levels[j]
        local_idx = level_counts[lvl]
        level_counts[lvl] += 1

        plot_shape_on_ax(ax, j)

        ax.set_title(f'L{lvl}-N{local_idx}', fontsize=7, pad=1)
        ax.set_xlabel('u', fontsize=6, labelpad=0)
        ax.set_ylabel('v', fontsize=6, labelpad=0)
        ax.set_zlim(0.0, 1.0)
        ax.set_zticks([0, 0.5, 1])
        ax.tick_params(labelsize=5, pad=0)
        ax.view_init(elev=30, azim=-60)

    # Link camera angles: rotating one subplot updates all others
    def on_move(event):
        for src in axes_grid:
            if event.inaxes == src:
                for dst in axes_grid:
                    if dst is not src:
                        dst.view_init(elev=src.elev, azim=src.azim)
                fig.canvas.draw_idle()
                break
    fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.suptitle("THB surface shape functions", fontsize=11)
    plt.tight_layout()
    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)
    filename = os.path.join(options.outdir, 'thb_surface_shapes.png')
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")

    ### Rotatable overlay: all shape functions in one 3D axes ###
    fig2 = plt.figure(figsize=(5.0, 4.0))
    ax2 = fig2.add_subplot(111, projection='3d')

    level_counts = [0] * geom.NumberOfLevels()
    handles = []
    for j in range(n_active):
        lvl = cp_levels[j]
        local_idx = level_counts[lvl]
        level_counts[lvl] += 1

        smin_u, smax_u, smin_v, smax_v = cp_supports[j]
        mask = (U < smin_u - 1e-10) | (U > smax_u + 1e-10) \
             | (V < smin_v - 1e-10) | (V > smax_v + 1e-10)
        Z = N_np[:, j].reshape(n_eval, n_eval)
        Z_masked = numpy.ma.masked_where(mask, Z)

        color = level_colors[lvl % len(level_colors)]
        ax2.plot_surface(U, V, Z_masked, color=color, alpha=0.45,
                         linewidth=0, antialiased=True)
        rect_u = [smin_u, smax_u, smax_u, smin_u, smin_u]
        rect_v = [smin_v, smin_v, smax_v, smax_v, smin_v]
        ax2.plot(rect_u, rect_v, [0]*5, color='k', linewidth=0.6, zorder=5)
        handles.append((color, f'L{lvl}-N{local_idx}'))

    ax2.set_xlabel('u', fontsize=10)
    ax2.set_ylabel('v', fontsize=10)
    ax2.set_zlabel('N(u,v)', fontsize=10)
    ax2.set_zlim(0.0, 1.0)
    ax2.set_title("All THB shape functions", fontsize=10)

    legend_elements = [Patch(facecolor=c, alpha=0.7, label=lbl) for c, lbl in handles]
    ax2.legend(handles=legend_elements, fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(0.0, 1.0))

    plt.tight_layout()
    filename = os.path.join(options.outdir, 'thb_surface_shapes_allonce.png')
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")

    if options.debug:
        plt.show()


if __name__ == "__main__":
    main()

