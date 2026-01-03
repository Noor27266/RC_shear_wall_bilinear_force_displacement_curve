# =============================================================================
# 📊 STEP 6: INPUT PARAMETERS & DATA RANGES DEFINITION  (THETA REMOVED ONLY)
# =============================================================================
R = {
    "lw": (400.0, 3500.0),
    "hw": (495.0, 5486.4),
    "tw": (26.0, 305.0),
    "fc": (13.38, 93.6),
    "fyt": (0.0, 1187.0),
    "fysh": (0.0, 1375.0),
    "fyl": (160.0, 1000.0),
    "fybl": (0.0, 900.0),
    "rt": (0.000545, 0.025139),
    "rsh": (0.0, 0.041888),
    "rl": (0.0, 0.029089),
    "rbl": (0.0, 0.031438),
    "axial": (0.0, 0.86),
    "b0": (45.0, 3045.0),
    "db": (0.0, 500.0),
    "s_db": (0.0, 47.65625),
    "AR": (0.388889, 5.833333),
    "M_Vlw": (0.388889, 4.1),
}

U = lambda s: rf"\;(\mathrm{{{s}}})"

# ✅ M/(Vlw) REMOVED from Geometry
GEOM = [
    (rf"$l_w{U('mm')}$", "lw", 1000.0, 1.0, None, "Length"),
    (rf"$h_w{U('mm')}$", "hw", 495.0, 1.0, None, "Height"),
    (rf"$t_w{U('mm')}$", "tw", 200.0, 1.0, None, "Thickness"),
    (rf"$b_0{U('mm')}$", "b0", 200.0, 1.0, None, "Boundary element width"),
    (rf"$d_b{U('mm')}$", "db", 400.0, 1.0, None, "Boundary element length"),
    (r"$AR$", "AR", 2.0, 0.01, None, "Aspect ratio"),
]

# ✅ M/(Vlw) ADDED at the END of Material Strengths (so it comes after fybl)
MATS = [
    (rf"$f'_c{U('MPa')}$", "fc", 40.0, 0.1, None, "Concrete strength"),
    (rf"$f_{{yt}}{U('MPa')}$", "fyt", 400.0, 1.0, None, "Transverse web yield strength"),
    (rf"$f_{{ysh}}{U('MPa')}$", "fysh", 400.0, 1.0, None, "Transverse boundary yield strength"),
    (rf"$f_{{yl}}{U('MPa')}$", "fyl", 400.0, 1.0, None, "Vertical web yield strength"),
    (rf"$f_{{ybl}}{U('MPa')}$", "fybl", 400.0, 1.0, None, "Vertical boundary yield strength"),
    (r"$M/(V_{l_w})$", "M_Vlw", 2.0, 0.01, None, "Shear span ratio"),  # ✅ moved here
]
