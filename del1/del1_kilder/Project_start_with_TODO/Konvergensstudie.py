# -*- coding: utf-8 -*-
"""
Konvergensstudie for utkrageren i Cantilever_FEM_main.py med alle fire elementtypene
(T3, Q4, T6, Q9) og fire nettfinheter.

Størrelser som sammenlignes:
  v_tip       vertikal forskyvning i midtnoden på frie ende (x = L, y = H/2)
  sigma_x     bøyespenning i underkant midt på bjelken (x = L/2, y = 0)
  tau_xy      skjærspenning i nøytralaksen midt på bjelken (x = L/2, y = H/2)
  sigma_x,inn bøyespenning i hjørnet ved innspenningen (x = 0, y = 0) - singulært punkt

Spenninger i en node er gjennomsnittet av hjørnespenningene fra alle elementer som
deler noden (nodal midling). Nettene er valgt slik at alle målepunktene er hjørnenoder
for alle elementtypene, og alle elementtypene får like mange frihetsgrader.

Referanseverdier:
  sigma_x, tau_xy : bjelketeori, sigma = M*c/I og tau_max = 1.5*P/A
  v_tip           : Q9 på fineste nett (2D-skiveløsning). Timoshenko og Euler-Bernoulli
                    skrives ut for sammenligning.

Kjøres fra denne mappa:  python3 Konvergensstudie.py
Skriver: del1/resultater/konvergens.csv, del1/resultater/konvergens.md,
         del1/figurer/konvergens.png
"""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fem_models
import fem_utilities as fem_util

# ----- Problem (samme som Cantilever_FEM_main.py) ------------------------------
L, H, t = 10.0, 2.0, 0.1
P = 3.0e6
E, nu = 2.1e11, 0.3
endLoadXY = np.array([0.0, P])

meshes = [(9, 5), (17, 9), (33, 17), (65, 33)]   # (numNodesX, numNodesY)
elements = [(3, "T3"), (4, "Q4"), (6, "T6"), (9, "Q9")]

# ----- Analytiske verdier ------------------------------------------------------
I = t * H**3 / 12.0
A = t * H
G = E / (2.0 * (1.0 + nu))
v_EB = P * L**3 / (3.0 * E * I)
v_Timo = v_EB + P * L / (5.0 / 6.0 * G * A)
sx_theory = P * (L - L / 2) * (H / 2) / I      # underkant, x = L/2 (strekk)
tau_theory = 1.5 * P / A                       # nøytralaksen

here = os.path.dirname(os.path.abspath(__file__))
del1_dir = os.path.abspath(os.path.join(here, "..", ".."))
res_dir = os.path.join(del1_dir, "resultater")
fig_dir = os.path.join(del1_dir, "figurer")
os.makedirs(res_dir, exist_ok=True)


def node_at(model, x, y):
    c = np.array(model.coords)
    i = int(np.argmin((c[:, 0] - x)**2 + (c[:, 1] - y)**2))
    assert np.hypot(c[i, 0] - x, c[i, 1] - y) < 1e-9, "punktet er ikke en node"
    return i


def nodal_average_stress(model, corner_stresses, inode):
    vals = []
    for iel, nods in enumerate(model.elnods):
        n_corners = 3 if len(nods) in (3, 6) else 4
        for ic in range(n_corners):
            if nods[ic] == inode:
                vals.append(np.asarray(corner_stresses[iel][ic], dtype=float).ravel())
    assert vals, "noden er ikke et hjørne i noe element"
    return np.mean(vals, axis=0)


rows = []
for nx, ny in meshes:
    for n_el_nodes, name in elements:
        model = fem_models.CantileverModel(L, H, n_el_nodes, nx, ny, endLoadXY, None)
        K, R = model.assemble_stiffness_matrix_and_load_vector()
        r, R0 = fem_util.solveq(K, R, model.bcdofs)
        cs = model.compute_element_corner_stresses(r)

        n_tip = node_at(model, L, H / 2)
        s_mid_bot = nodal_average_stress(model, cs, node_at(model, L / 2, 0.0))
        s_mid_na = nodal_average_stress(model, cs, node_at(model, L / 2, H / 2))
        s_clamp = nodal_average_stress(model, cs, node_at(model, 0.0, 0.0))

        rows.append(dict(
            element=name, nodes_x=nx, nodes_y=ny, n_elements=len(model.elnods),
            ndof=2 * nx * ny,
            v_tip=r[2 * n_tip + 1, 0],
            sigma_x_mid=s_mid_bot[0],
            tau_xy_mid=s_mid_na[2],
            sigma_x_clamp=s_clamp[0],
            R_y=R0[1::2, 0].sum(),
        ))
        print("{:2s} {:3d}x{:<3d} ndof={:5d}  v={:.5f}  sx={:.4e}  tau={:.4e}  sx_inn={:.4e}".format(
            name, nx, ny, rows[-1]["ndof"], rows[-1]["v_tip"], rows[-1]["sigma_x_mid"],
            rows[-1]["tau_xy_mid"], rows[-1]["sigma_x_clamp"]))

v_ref = [r for r in rows if r["element"] == "Q9" and (r["nodes_x"], r["nodes_y"]) == meshes[-1]][0]["v_tip"]

for r in rows:
    r["err_v"] = abs(r["v_tip"] - v_ref) / v_ref
    r["err_sigma_x"] = abs(r["sigma_x_mid"] - sx_theory) / sx_theory
    r["err_tau_xy"] = abs(r["tau_xy_mid"] - tau_theory) / tau_theory

# ----- CSV ---------------------------------------------------------------------
csv_path = os.path.join(res_dir, "konvergens.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ----- Markdown-tabeller -------------------------------------------------------
names = [n for _, n in elements]


def get(el, mesh, key):
    return [r for r in rows if r["element"] == el and (r["nodes_x"], r["nodes_y"]) == mesh][0][key]


def table(title, key, err_key, fmt, ref_text):
    out = ["### " + title, "", ref_text, "",
           "| Noder | ndof | " + " | ".join(names) + " |",
           "|---|---:|" + "---:|" * len(names)]
    for m in meshes:
        cells = []
        for el in names:
            val = fmt.format(get(el, m, key))
            if err_key:
                val += " ({:.1f} %)".format(100 * get(el, m, err_key))
            cells.append(val)
        out.append("| {}×{} | {} | ".format(*m, 2 * m[0] * m[1]) + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


md = ["# Konvergensstudie – utkrager",
      "",
      "Generert av `del1_kilder/Project_start_with_TODO/Konvergensstudie.py`.",
      "",
      "Utkrager L = {:g} m, H = {:g} m, t = {:g} m, E = {:.1e} Pa, ν = {:g}, endelast P = {:.1e} N "
      "(jevnt fordelt på nodene i fri ende), fast innspent venstre ende.".format(L, H, t, E, nu, P),
      "Spenninger er nodalt midlet over elementene som deler noden. Tall i parentes er relativt avvik fra referansen.",
      "",
      "### Antall elementer",
      "",
      "| Noder | ndof | " + " | ".join(names) + " |",
      "|---|---:|" + "---:|" * len(names)]
for m in meshes:
    md.append("| {}×{} | {} | ".format(*m, 2 * m[0] * m[1]) +
              " | ".join(str(get(el, m, "n_elements")) for el in names) + " |")
md.append("")
md.append(table("Tuppforskyvning v [m] i (x, y) = (L, H/2)", "v_tip", "err_v", "{:.5f}",
                "Referanse: Q9 på fineste nett, v = {:.5f} m. Til sammenligning: Euler–Bernoulli {:.5f} m, "
                "Timoshenko (k = 5/6) {:.5f} m.".format(v_ref, v_EB, v_Timo)))
md.append(table("Bøyespenning σx [Pa] i (x, y) = (L/2, 0)", "sigma_x_mid", "err_sigma_x", "{:.3e}",
                "Referanse: bjelketeori σx = M·c/I = {:.3e} Pa.".format(sx_theory)))
md.append(table("Skjærspenning τxy [Pa] i (x, y) = (L/2, H/2)", "tau_xy_mid", "err_tau_xy", "{:.3e}",
                "Referanse: bjelketeori τmax = 1,5·P/A = {:.3e} Pa.".format(tau_theory)))
md.append(table("Bøyespenning σx [Pa] i innspenningshjørnet (x, y) = (0, 0)", "sigma_x_clamp", None, "{:.3e}",
                "Bjelketeori gir {:.3e} Pa, men hjørnet er et singulært punkt (fast innspenning i både x og y). "
                "Verdien vokser når nettet forfines og konvergerer ikke.".format(P * L * (H / 2) / I)))
md_path = os.path.join(res_dir, "konvergens.md")
with open(md_path, "w") as f:
    f.write("\n".join(md))

# ----- Figur: relativt avvik mot ndof (log-log), én rute per størrelse -------------
colors = {"T3": "#2a78d6", "Q4": "#eb6834", "T6": "#1baf7a", "Q9": "#eda100"}
markers = {"T3": "^", "Q4": "s", "T6": "D", "Q9": "o"}
ink, ink2, grid = "#1f1f1e", "#5f5e5a", "#e4e3df"

panels = [("err_v", "Tuppforskyvning v"),
          ("err_sigma_x", "Bøyespenning σx i (L/2, 0)"),
          ("err_tau_xy", "Skjærspenning τxy i (L/2, H/2)")]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink2,
                     "xtick.color": ink2, "ytick.color": ink2, "text.color": ink})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
for ax, (key, title) in zip(axes, panels):
    ends = []
    for el in names:
        sub = [r for r in rows if r["element"] == el]
        if key == "err_v":   # referansen selv har null avvik og kan ikke vises i log-plot
            sub = [r for r in sub if not (el == "Q9" and (r["nodes_x"], r["nodes_y"]) == meshes[-1])]
        x = [r["ndof"] for r in sub]
        y = [max(100 * r[key], 1e-12) for r in sub]   # vern mot log(0)
        ax.loglog(x, y, color=colors[el], marker=markers[el], markersize=7, linewidth=2,
                  markeredgecolor="white", markeredgewidth=1.5, label=el)
        ends.append([el, x[-1], y[-1]])
    # Direkte etiketter ved siste punkt, skjøvet fra hverandre så de ikke overlapper
    ends.sort(key=lambda e: e[2])
    min_gap = 0.075 * np.diff(np.log10(ax.get_ylim()))[0]
    ly = [np.log10(e[2]) for e in ends]
    for i in range(1, len(ly)):
        ly[i] = max(ly[i], ly[i - 1] + min_gap)
    for (el, xe, ye), yl in zip(ends, ly):
        ax.annotate(el, (xe, ye), xytext=(xe * 1.25, 10**yl), textcoords="data",
                    va="center", fontsize=9, color=ink)
    ax.set_title(title, loc="left", fontsize=11, color=ink)
    ax.set_xlabel("Antall frihetsgrader (ndof)")
    ax.grid(True, which="major", color=grid, linewidth=0.8)
    ax.grid(True, which="minor", color=grid, linewidth=0.4, alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.8)
axes[0].set_ylabel("Relativt avvik fra referanse [%]")
axes[0].legend(frameon=False, loc="lower left")
fig_path = os.path.join(fig_dir, "konvergens.png")
fig.savefig(fig_path, dpi=200)

print("\nSkrev", csv_path)
print("Skrev", md_path)
print("Skrev", fig_path)
