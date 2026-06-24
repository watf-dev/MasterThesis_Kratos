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
    parser.add_argument("--show",   action="store_true", help="show the shape functions")
    parser.add_argument("--outdir", default="ShapeFunctions",        help="output dir for txt files")
    parser.add_argument("--tex",    default="1D_shape_functions.tex", help="output tex file")
    options = parser.parse_args()
    import os
    import numpy

    model = KM.Model()
    mp = model.CreateModelPart("main")
    p = 2
    knots = [0.0, 0.0, 1, 2, 3, 4, 5, 6, 6]
    knots_t = make_knots(knots)
    grev_t = greville(list(knots_t), p)

    node_id = 1
    points = KM.NodesVector()
    for it, t in enumerate(grev_t):
        points.append(mp.CreateNewNode(node_id, t, 0.0, 0.0))  # Line: x=t, y=0, z=0
        node_id += 1

    geom = KM.THBCurveGeometry3D(points, p, knots_t)

    geom.AddLevel(2,mp)
    geom.AddRefinementDomain(1, 2.0, 4.0) 
    # geom.AddRefinementDomain(2, 3.0, 4.0) 
    # geom.AddRefinementDomain(2, 2.5, 3.5) 
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
    cp_flat_indices = []
    for l in range(geom.NumberOfLevels()):
        flags = geom.GetActiveFunctions(l)
        for flat_idx in range(len(flags)):
            if flags[flat_idx]:
                cp_levels.append(l)
                cp_flat_indices.append(flat_idx)
    level_counts = [0] * geom.NumberOfLevels()

    # A level-l function is truncated if the midpoint of its support falls inside
    # a refinement domain at a finer level. This covers both conditions:
    #   1. The support overlaps Ω^{l+1} (truncation operator is applied).
    #   2. Active children exist in that overlap (operator is non-trivial),
    #      since the refinement domain is fully populated with active fine CPs.
    ref_domains = list(geom.RefinementDomains())
    finest_level = geom.NumberOfLevels() - 1
    degree = geom.PolynomialDegree(0)

    def is_truncated(lvl, flat_idx):
        if lvl >= finest_level:
            return False
        knots = list(geom.Levels()[lvl].Knots)
        supp_min = knots[flat_idx - 1] if flat_idx > 0 else knots[0]
        supp_max = knots[min(flat_idx + degree, len(knots) - 1)]
        midpoint = 0.5 * (supp_min + supp_max)
        tol = 1e-10
        for dom in ref_domains:
            if dom.Level > lvl and dom.MinT + tol < midpoint < dom.MaxT - tol:
                return True
        return False

    os.makedirs(options.outdir, exist_ok=True)
    for i in range(n_active):
        lvl = cp_levels[i]
        flat_idx = cp_flat_indices[i]
        local_idx = level_counts[lvl]
        level_counts[lvl] += 1

        N_col = N_numpy[:, i]
        tag = "_truncated" if is_truncated(lvl, flat_idx) else ""
        filename = f"L{lvl}_N{local_idx}{tag}.txt"

        nonzero_mask = numpy.abs(N_col) > 1e-12
        numpy.savetxt(
            os.path.join(options.outdir, filename),
            numpy.stack([t_eval[nonzero_mask], N_col[nonzero_mask]], axis=-1)
        )

    ### Generate tex file ###
    # Color per (level, truncated): non-truncated / truncated pairs per level.
    level_color_pairs = [
        ('blue',         'cyan'),    # level 0
        ('red',          'orange'),  # level 1
        ('green!70!black', 'green'), # level 2
        ('purple',       'violet'),  # level 3
    ]

    xmin = int(min(knots))
    xmax = int(max(knots))
    xtick_list = ','.join(str(v) for v in range(xmin, xmax + 1))

    addplot_lines = []
    seen_legend_keys = []  # ordered unique (lvl, truncated) for legend
    level_counts_tex = [0] * geom.NumberOfLevels()
    for i in range(n_active):
        lvl = cp_levels[i]
        flat_idx = cp_flat_indices[i]
        local_idx = level_counts_tex[lvl]
        level_counts_tex[lvl] += 1
        truncated = is_truncated(lvl, flat_idx)
        tag = "_truncated" if truncated else ""
        fname = f"{options.outdir}/L{lvl}_N{local_idx}{tag}.txt"
        pair = level_color_pairs[lvl % len(level_color_pairs)]
        color = pair[1] if truncated else pair[0]
        addplot_lines.append(
            f'  \\addplot[color={color}, forget plot] table [x index=0, y index=1] {{{fname}}};'
        )
        key = (lvl, truncated)
        if key not in seen_legend_keys:
            seen_legend_keys.append(key)

    legend_lines = []
    for (lvl, truncated) in seen_legend_keys:
        pair = level_color_pairs[lvl % len(level_color_pairs)]
        color = pair[1] if truncated else pair[0]
        label = f"Level {lvl} truncated" if truncated else f"Level {lvl}"
        legend_lines.append(f'  \\addlegendimage{{color={color}, thick}}')
        legend_lines.append(f'  \\addlegendentry{{\\textrm{{{label}}}}}')

    # One indicator line per refinement domain, stacked downward by level.
    draw_lines = []
    for dom in sorted(ref_domains, key=lambda d: d.Level):
        lvl = dom.Level
        color = level_color_pairs[lvl % len(level_color_pairs)][0]
        yshift = -1.5 - (lvl - 1) * 2
        draw_lines.append(
            f'    \\draw[{color}, ultra thick]'
            f' ([yshift={yshift}pt]axis cs:{dom.MinT},0)'
            f' -- ([yshift={yshift}pt]axis cs:{dom.MaxT},0);'
        )

    if draw_lines:
        # Shift x-tick labels down to clear the deepest indicator line.
        max_abs_yshift = max(abs(-1.5 - (dom.Level - 1) * 2) for dom in ref_domains)
        label_yshift = -(max_abs_yshift - 1)
        axis_options = (
            '[\n'
            f'  xticklabel style={{yshift={label_yshift:.1f}pt}},\n'
            '  after end axis/.code={\n'
            + '\n'.join(draw_lines) +
            '\n  }\n]'
        )
    else:
        axis_options = ''

    tex_content = r"""\documentclass[multi=minipage,border=0]{standalone}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{pgfplots,tikz}
\usepackage{geometry}
\usepackage[T1]{fontenc}
\usepackage[scaled]{helvet}
\usepackage{pgffor}
\renewcommand*\familydefault{\sfdefault}

\pgfplotsset{compat=newest}
\pgfplotsset{every axis/.append style={
  thick,
  solid,
  grid=major,
  axis equal image,
  scale only axis=true,
  %%=== x setting ===%%
  xmin=XMIN, xmax=XMAX,
  xtick={XTICK},
  xlabel={$\Xi$},
  %%=== y setting ===%%
  ymin=0, ymax=1,
  ytick={0,1},
  ylabel={$N_a$},
  %%=== legend setting ===%%
  legend columns=7,
  legend style={
    at={(0.5,-0.8)},
    anchor=north,
    font=\normalsize,
    legend cell align=left,
    /tikz/every even column/.append style={column sep=0.5cm},
    draw=none,
  },
}}

\begin{document}
\begin{tikzpicture}
\begin{axis}AXIS_OPTIONS
ADDPLOTS
LEGEND
\end{axis}
\end{tikzpicture}
\end{document}
"""
    tex_content = (tex_content
        .replace('XMIN',         str(xmin))
        .replace('XMAX',         str(xmax))
        .replace('XTICK',        xtick_list)
        .replace('ADDPLOTS',     '\n'.join(addplot_lines))
        .replace('LEGEND',       '\n'.join(legend_lines))
        .replace('AXIS_OPTIONS', axis_options)
        .replace('%%', '%'))

    with open(options.tex, 'w') as f:
        f.write(tex_content)
    print(f"Saved: {options.tex}")

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
        filename = "1D_shape_functions_matplotlib.pdf"
        plt.savefig(filename, dpi=150)
        plt.show()
        print(f"\nSaved: {filename}")


if __name__ == "__main__":
    main()

