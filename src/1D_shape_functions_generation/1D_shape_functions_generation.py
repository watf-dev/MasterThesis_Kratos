#!/opt/homebrew/bin/python3
# Created: Jun, 23, 2026 15:52:37 by Wataru Fukuda

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

"""
    )
    parser.add_argument("--show", action="store_true", help="show the shape functions")
    parser.add_argument("--outdir", default="ShapeFunctions", help="output dir")
    options = parser.parse_args()
    import os
    import numpy

    model = KM.Model()
    mp = model.CreateModelPart("main")
    p = 2
    knots = [0.0, 0.0, 1, 2, 3, 4, 5, 6, 7, 7]
    knots_t = make_knots(knots)
    grev_t = greville(list(knots_t), p)

    node_id = 1
    points = KM.NodesVector()
    for it, t in enumerate(grev_t):
        points.append(mp.CreateNewNode(node_id, t, 0.0, 0.0))  # Line: x=t, y=0, z=0
        node_id += 1

    geom = KM.THBCurveGeometry3D(points, p, knots_t)

    geom.AddLevel(2,mp)
    geom.AddRefinementDomain(1, 3.0, 6.0) 
    geom.EliminateInactiveFunctions(mp)

    n_eval = 1000
    t_eval = numpy.linspace(0.0, max(knots) - 1e-10, n_eval)   # avoid t=1.0 (half-open right boundary)
    t_vec = KM.Vector(n_eval)
    for i, t in enumerate(t_eval):
        t_vec[i] = t

    N_matrix = geom.EvaluateShapeFunctions(t_vec)   # (n_eval × n_active_cps) Matrix
    n_active = geom.PointsNumber()
    N_numpy = numpy.array([[N_matrix[i, j] for j in range(n_active)] for i in range(n_eval)])

    pou_err = numpy.max(numpy.abs(N_numpy.sum(axis=1) - 1.0))
    print(f"Max POU error on {n_eval} dense pts: {pou_err:.2e}  "
          f"({'PASS' if pou_err < 1e-10 else 'FAIL'})")

    knot_spans = geom.SpansLocalSpace(0)

    cp_levels = []
    for l in range(geom.NumberOfLevels()):
        cp_levels.extend([l] * geom.NumberOfActiveFunctions(l))
    level_counts = [0] * geom.NumberOfLevels()

    os.makedirs(options.outdir, exist_ok=True)
    for i in range(n_active):
        lvl = cp_levels[i]
        local_idx = level_counts[lvl]
        level_counts[lvl] += 1
        # if truncated: filename = f"L{lvl}_N{local_idx}_truncated.txt", else: filename = f"L{lvl}_N{local_idx}.txt"
        filename = f"L{lvl}_N{local_idx}.txt"
        numpy.savetxt(os.path.join(options.outdir,filename), numpy.stack([t_eval,N_numpy[:, i]],axis=-1))

    if options.show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 5))
        # Map each packed CP index to its level
        level_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for j in range(n_active):
            lvl = cp_levels[j]
            local_idx = level_counts[lvl]
            level_counts[lvl] += 1
            color = level_colors[lvl % len(level_colors)]
            ax.plot(t_eval, N_numpy[:, j], color=color, lw=1.5,
                    label=f'L{lvl}-N{local_idx}')

        # Mark knot span boundaries
        for xi in knot_spans:
            ax.axvline(xi, color='gray', lw=0.7, ls='--', alpha=0.6)

        ax.axhline(0.0, color='k', lw=0.5)
        ax.set_xlim(0.0, max(knots))
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('parameter t')
        ax.set_ylabel('N(t)')
        ax.set_title('THB shape functions (vertical lines = knot spans)')
        ax.legend(fontsize=7, ncol=2, loc='upper right')

        plt.tight_layout()
        filename = "1D_shape_functions_matplotlib.png"
        plt.savefig(filename, dpi=150)
        plt.show()
        print(f"\nSaved: {filename}")


if __name__ == "__main__":
    main()

