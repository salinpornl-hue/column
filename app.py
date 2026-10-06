"""
RC Column Pro – Biaxial & Sway Analysis
ACI 318-19 / SDM (MKS Unit System: ksc, ton, cm, ton-m)

UNIT SYSTEM (consistent throughout):
  Force       : ton   (1 ton = 1000 kgf)
  Moment      : ton-m
  Length      : cm
  Stress (fc) : ksc   (kgf/cm²)
  Stress (fy) : ksc
  Ec, Es      : ksc
  Area        : cm²
  Inertia     : cm⁴
  EI          : ksc·cm²  → Pc in kgf → /1000 → ton

All ACI formulae that reference √f'c use the MKS coefficient (0.53, 1.06, etc.)
rather than the SI coefficient (0.17, 0.33).  Conversions:
  0.53 √f'c [ksc] ≡ 0.17 √f'c [MPa]  (since 1 MPa ≈ 10.2 ksc → √10.2 ≈ 3.19)

BUNDLED BARS (ACI 318-19):
  §26.6.3.1  – Maximum 4 bars per bundle in columns.
  §26.6.3.2  – Each bundle must be enclosed by ties.
  §25.6.1.2  – Clear spacing between bundles ≥ max(4/3·dagg, d_b,eq)
               where d_b,eq = d_b · √n_per_bundle  (equivalent single-bar diameter).
  §25.6.1.5  – Development/splice lengths computed using d_b,eq.
  §25.7.2.1  – Tie spacing: 6·d_b (individual bar, NOT d_b,eq).
  Structural: bundle centroid positions are unchanged;
              area per position = n_per_bundle × A_bar_single.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
from scipy.interpolate import interp1d

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
PHI_FLEX   = 0.90   # Tension-controlled sections
PHI_COMP_T = 0.65   # Compression-controlled, tied
PHI_COMP_S = 0.75   # Compression-controlled, spiral/circular
PHI_SHEAR  = 0.75   # Shear & torsion
ES_KSC     = 2_040_000.0   # ksc  (≈ 200 GPa)
EPS_CU     = 0.003          # ACI ultimate concrete strain
EPS_TY_LIM = 0.005          # Tension-controlled limit

# ──────────────────────────────────────────────────────────────────────────────
# ENGINE
# ──────────────────────────────────────────────────────────────────────────────
class RCColumn:
    """
    Generates P-M interaction data and helper calculations for a rectangular
    or circular RC column.  All inputs/outputs in MKS (ksc, ton, cm).
    """

    def __init__(self, shape, layout, b, h, fc, fy, db_mm, n_bars,
                 nx, ny, cover_cm, n_per_bundle=1, tie_cm=0.9):
        self.shape  = shape
        self.layout = layout
        self.b, self.h = float(b), float(h)
        self.fc, self.fy = float(fc), float(fy)
        self.Es = ES_KSC

        # ── Bundled-bar parameters (ACI 318-19 §26.6.3, §25.6.1) ─────────────
        self.n_per_bundle = int(max(1, min(4, n_per_bundle)))   # 1–4 per ACI
        # Equivalent single-bar diameter for spacing & splice checks (§25.6.1.2)
        self.db_cm        = db_mm / 10.0
        self.db_eq_cm     = self.db_cm * math.sqrt(self.n_per_bundle)
        # Single-bar area and bundle area (area at each position)
        self.as_single    = math.pi * self.db_cm**2 / 4.0
        self.as_bundle    = self.n_per_bundle * self.as_single  # area per position

        # β₁  ACI 318-19 Table 22.2.2.4.3  (fc in ksc; threshold 280 ksc ≈ 28 MPa)
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (self.fc - 280.0) / 70.0))

        # Tension-controlled strain limit (ACI 318-19)
        self.eps_ty_lim = (self.fy / self.Es) + 0.003

        # Section geometry
        if shape == "Rectangular":
            self.Ag  = self.b * self.h
            self.Igx = self.b * self.h**3 / 12.0
            self.Igy = self.h * self.b**3 / 12.0
            self.rx  = 0.3 * self.h   # radius of gyration about X-axis (bending in h-direction)
            self.ry  = 0.3 * self.b   # radius of gyration about Y-axis
        else:  # Circular
            self.D   = self.h          # diameter = h input
            self.Ag  = math.pi * self.D**2 / 4.0
            self.Igx = self.Igy = math.pi * self.D**4 / 64.0
            self.rx  = self.ry = 0.25 * self.D

        # Concrete modulus  (Ec = 15100√f'c  for normal-weight concrete, MKS)
        self.Ec = 15_100.0 * math.sqrt(self.fc)

        # Cover to centroid of bundle:  cover + tie(≈9mm) + db_eq/2
        # For a bundle the geometric centroid shifts outward by ~db/4 for 2-bar,
        # db/3 for 3-bar, db/2.83 for 4-bar.  ACI §26.6.3 permits treating the
        # bundle centroid as equivalent to a single bar at its centroidal position,
        # so we use the same cover-to-centroid formula but with db_eq/2.
        self.tie_cm   = float(tie_cm)   # [REVIEW] was hard-coded 0.9 cm
        self.d_prime  = cover_cm + self.tie_cm + self.db_eq_cm / 2.0

        # Keep db_cm attribute (single bar) for tie-spacing (§25.7.2.1 uses db, not db_eq)
        # and for the exploded view rendering.

        # Bar (bundle position) layout — positions unchanged by bundling
        self.bars = self._place_bars(n_bars, nx, ny)
        self.n_positions = len(self.bars)        # number of bundle positions
        self.n_bars      = self.n_positions * self.n_per_bundle  # total individual bars
        # Area per position = bundle area; total steel area
        self.as_bar   = self.as_bundle           # keeps downstream solve_pm working
        self.total_as = self.n_positions * self.as_bundle
        self.rho      = self.total_as / self.Ag

        # Steel moment of inertia (used in EI calculation) — use bundle area per position
        self.Ise_x = sum(self.as_bundle * bar['y']**2 for bar in self.bars)
        self.Ise_y = sum(self.as_bundle * bar['x']**2 for bar in self.bars)

    # ── bar placement ──────────────────────────────────────────────────────────
    def _place_bars(self, n_bars, nx, ny):
        bars = []
        dp = self.d_prime
        if self.shape == "Rectangular":
            x_min = -self.b / 2 + dp
            x_max =  self.b / 2 - dp
            y_min = -self.h / 2 + dp
            y_max =  self.h / 2 - dp

            if self.layout == "2-Faces (Top/Bottom)":
                n_each = max(2, n_bars // 2)
                for x in np.linspace(x_min, x_max, n_each):
                    bars.append({'x': float(x), 'y': float(y_max)})
                    bars.append({'x': float(x), 'y': float(y_min)})

            else:  # 4-Faces
                nx = max(2, int(nx))
                ny = max(2, int(ny))
                for x in np.linspace(x_min, x_max, nx):
                    bars.append({'x': float(x), 'y': float(y_max)})
                    bars.append({'x': float(x), 'y': float(y_min)})
                if ny > 2:
                    for y in np.linspace(y_min, y_max, ny)[1:-1]:
                        bars.append({'x': float(x_min), 'y': float(y)})
                        bars.append({'x': float(x_max), 'y': float(y)})
        else:  # Circular
            Rs = self.D / 2.0 - dp
            for i in range(n_bars):
                theta = i * 2 * math.pi / n_bars
                bars.append({'x': Rs * math.sin(theta), 'y': Rs * math.cos(theta)})
        return bars

    # ── P-M interaction ────────────────────────────────────────────────────────
    def solve_pm(self, axis='X'):
        """
        Returns (DataFrame of P-M points, phi*Pn_max).
        Columns: c, Pn [ton], Mn [ton-m], phiPn [ton], phiMn [ton-m]
        Sign convention: compression positive.
        """
        is_circular = (self.shape == "Circular")
        phi_comp    = PHI_COMP_S if is_circular else PHI_COMP_T

        if is_circular:
            depth = self.D
            width = self.D
            get_coord = lambda bar: bar['y']
        else:
            if axis == 'X':
                depth, width = self.h, self.b
                get_coord = lambda bar: bar['y']
            else:
                depth, width = self.b, self.h
                get_coord = lambda bar: bar['x']

        y_bars = np.array([get_coord(b) for b in self.bars], dtype=float)
        d_bars = depth / 2.0 - y_bars   # distance from compression face
        dt     = float(np.max(d_bars))  # extreme tension steel depth

        results = []

        # Sweep c from deep compression (pure compression) to near-zero (near pure tension)
        c_vals = np.concatenate([
            np.linspace(depth * 10.0, dt + 0.001, 150),
            np.linspace(dt, 0.01 * depth, 150)
        ])
        c_vals = np.unique(np.clip(c_vals, 1e-6, None))

        for c in c_vals:
            a = min(self.beta1 * c, depth)

            # ── Concrete compression resultant ──────────────────────────────
            if is_circular:
                R = self.D / 2.0
                if a >= self.D:
                    Ac    = math.pi * R**2
                    y_bar = 0.0
                else:
                    ratio = max(-1.0, min(1.0, (R - a) / R))
                    theta_c = 2.0 * math.acos(ratio)
                    Ac    = (R**2 / 2.0) * (theta_c - math.sin(theta_c))
                    if Ac > 1e-10:
                        y_bar = (4.0 * R * math.sin(theta_c / 2.0)**3) / \
                                (3.0 * (theta_c - math.sin(theta_c)))
                    else:
                        y_bar = R
                Cc = 0.85 * self.fc * Ac           # kgf
                Mc = Cc * y_bar                     # kgf·cm  (from section centroid)
            else:
                Cc = 0.85 * self.fc * a * width     # kgf
                Mc = Cc * (depth / 2.0 - a / 2.0)  # kgf·cm

            # ── Steel forces ────────────────────────────────────────────────
            eps_s = EPS_CU * (c - d_bars) / c      # + = compression
            fs    = np.clip(eps_s * self.Es, -self.fy, self.fy)  # ksc
            # [REVIEW] bars inside the compression block displace concrete: use (fs - 0.85 f'c)
            in_block = d_bars <= a
            Fsi   = self.as_bar * (fs - np.where(in_block & (eps_s > 0), 0.85 * self.fc, 0.0))

            Pn_s  = float(np.sum(Fsi))              # kgf
            Mn_s  = float(np.sum(Fsi * y_bars))    # kgf·cm

            # ── Totals (convert to ton and ton-m) ──────────────────────────
            Pn = (Cc + Pn_s)  / 1_000.0            # ton
            Mn = (Mc + Mn_s)  / 100_000.0          # ton-m

            # ── φ factor (extreme tension strain in outermost steel) ────────
            # ACI 318-19 §21.2.2: εt,lim = fy/Es + 0.003  (fy-dependent, not constant)
            et   = EPS_CU * (dt - c) / c
            ey   = self.fy / self.Es
            eps_ty = self.eps_ty_lim   # fy/Es + 0.003  (correct per §21.2.2)
            if et >= eps_ty:
                phi = PHI_FLEX
            elif et <= ey:
                phi = phi_comp
            else:
                phi = phi_comp + (PHI_FLEX - phi_comp) * \
                      (et - ey) / (eps_ty - ey)

            results.append({
                'c': c, 'Pn': Pn, 'Mn': abs(Mn),
                'phiPn': phi * Pn, 'phiMn': phi * abs(Mn)
            })

        # ── Pure tension point ──────────────────────────────────────────────
        Pn_tension = -self.total_as * self.fy / 1_000.0  # ton (negative)
        results.append({'c': 0.0, 'Pn': Pn_tension, 'Mn': 0.0,
                        'phiPn': PHI_FLEX * Pn_tension, 'phiMn': 0.0})

        # ── Maximum axial compression (pure squash) ─────────────────────────
        Po = (0.85 * self.fc * (self.Ag - self.total_as) +
              self.fy * self.total_as) / 1_000.0          # ton
        phi_max_factor = 0.85 if is_circular else 0.80    # ACI 318-19 §22.4.2
        phi_pn_max     = phi_comp * phi_max_factor * Po

        results.append({'c': 1e6, 'Pn': Po, 'Mn': 0.0,
                        'phiPn': phi_comp * Po, 'phiMn': 0.0})

        df = pd.DataFrame(results).sort_values('Pn').reset_index(drop=True)
        # [REVIEW] ACI 22.4.2: design curve is CUT OFF horizontally at phi*Pn,max.
        # Keep the first point that reaches the cap (its phiMn is retained) and
        # close the curve with a horizontal segment to (phiMn = 0, phiPn = cap).
        over = df['phiPn'] >= phi_pn_max
        below = df[~over].drop_duplicates(subset=['phiPn'], keep='first')
        if over.any():
            first_cap = df[over].iloc[[0]].copy()
            first_cap['phiPn'] = phi_pn_max
            squash = df[over].iloc[[-1]].copy()
            squash['phiPn'] = phi_pn_max
            squash['phiMn'] = 0.0
            df = pd.concat([below, first_cap, squash], ignore_index=True)
        else:
            df = below
        df = df.sort_values(['phiPn', 'phiMn'], ascending=[True, False]).reset_index(drop=True)

        return df, phi_pn_max

    # ── Slenderness magnification (Non-Sway) ───────────────────────────────────
    def slenderness_magnifier(self, Pu_ton, K, Lu_m, axis, Cm, beta_d):
        """
        ACI 318-19 §6.6.4  –  Moment Magnification (Nonsway frames).
        Returns (kl/r, Pc [ton], δ, Ise [cm⁴], EI [ksc·cm²])
        """
        if K <= 0 or Lu_m <= 0:
            # Guard against degenerate sway inputs
            r  = self.rx if axis == 'X' else self.ry
            Ig = self.Igx if axis == 'X' else self.Igy
            Ise = self.Ise_x if axis == 'X' else self.Ise_y
            EI  = 0.2 * self.Ec * Ig + self.Es * Ise
            return 0.0, 1e9, 1.0, Ise, EI

        Lu_cm = Lu_m * 100.0
        r    = self.rx  if axis == 'X' else self.ry
        Ig   = self.Igx if axis == 'X' else self.Igy
        Ise  = self.Ise_x if axis == 'X' else self.Ise_y

        kl_r = K * Lu_cm / r

        # EI  ACI 318-19 Eq. (6.6.4.4.4a)
        EI = (0.2 * self.Ec * Ig + self.Es * Ise) / (1.0 + beta_d)

        # Euler critical load   [ton]
        Pc = (math.pi**2 * EI) / (K * Lu_cm)**2 / 1_000.0

        # δns  ACI 318-19 Eq. (6.6.4.5.2)
        denom = 1.0 - Pu_ton / (0.75 * Pc) if Pc > 0 else -1.0
        if denom <= 0:
            delta = 999.9
        else:
            delta = max(1.0, Cm / denom)

        return kl_r, Pc, delta, Ise, EI

    # ── Clear spacing check ────────────────────────────────────────────────────
    def check_clear_spacing(self, nx, ny):
        """
        ACI 318-19 §25.6.1.2 (bundles) / §25.2.3 (single bars):
          Clear spacing between bundles ≥ max(4/3·dagg, d_b,eq)
          where d_b,eq = d_b · √n_per_bundle.
          Simplified MKS minimum: max(2.5 cm, d_b,eq) — conservative for 4/3·dagg.

        For single bars (n_per_bundle = 1) this reduces to the standard check.
        The 'actual' measured clear gap is centre-to-centre spacing minus
        d_b,eq (the equivalent bundle footprint).
        """
        # Minimum clear spacing uses equivalent diameter (§25.6.1.2)
        min_req = max(4.0, 1.5 * self.db_eq_cm)   # [REVIEW] ACI 25.2.3 (columns), was max(2.5, db)

        # Number of bundle positions (not individual bars)
        n_pos = self.n_positions

        if self.shape == "Rectangular":
            dp = self.d_prime
            sx = sy = 999.0
            if self.layout == "2-Faces (Top/Bottom)":
                n_each = n_pos // 2   # positions per face
                if n_each > 1:
                    # centre-to-centre between bundle positions minus bundle footprint
                    sx = (self.b - 2 * dp) / (n_each - 1) - self.db_eq_cm
            else:
                _nx = max(2, int(nx)); _ny = max(2, int(ny))
                if _nx > 1:
                    sx = (self.b - 2 * dp) / (_nx - 1) - self.db_eq_cm
                if _ny > 1:
                    sy = (self.h - 2 * dp) / (_ny - 1) - self.db_eq_cm
            actual = min(sx, sy)
        else:
            Rs    = self.D / 2.0 - self.d_prime
            chord = 2.0 * Rs * math.sin(math.pi / n_pos)
            actual = chord - self.db_eq_cm   # chord minus bundle footprint

        return actual, min_req, actual >= min_req

    # ── PCA exponent α ─────────────────────────────────────────────────────────
    def get_alpha(self, Pu_ton):
        """
        Interpolated PCA α between 1.15 (low axial) and 1.55 (high axial).
        For circular sections α = 2.0 (Bresler bilinear is exact for circles).
        """
        if self.shape == "Circular":
            return 2.0
        Po_ton  = (0.85 * self.fc * (self.Ag - self.total_as) +
                   self.fy * self.total_as) / 1_000.0
        phi_Po  = PHI_COMP_T * Po_ton
        if phi_Po <= 0:
            return 1.15
        ratio = max(0.0, min(1.0, Pu_ton / phi_Po))
        return 1.15 + (ratio - 0.1) * (1.55 - 1.15) / 0.9 if ratio > 0.1 else 1.15

    # ── 3D surface ──────────────────────────────────────────────────────────────
    def generate_3d_surface(self, df_x, df_y, alpha, n_p=30, n_t=25):
        """
        Builds the biaxial failure surface using the PCA load-contour method.
        Vectorized with NumPy for fast Plotly rendering.
        """
        p_lo = max(df_x['phiPn'].min(), df_y['phiPn'].min()) + 0.01
        p_hi = min(df_x['phiPn'].max(), df_y['phiPn'].max()) - 0.01
        
        if p_lo >= p_hi:
            return np.array([]), np.array([]), np.array([])

        p_steps = np.linspace(p_lo, p_hi, n_p)
        theta = np.linspace(0, math.pi / 2.0, n_t)
        
        # Create 2D arrays for vectorization
        T, P = np.meshgrid(theta, p_steps)
        
        # NumPy interpolation requires strictly ascending x-coordinates
        mx_cap = np.interp(P, df_x['phiPn'], df_x['phiMn'])
        my_cap = np.interp(P, df_y['phiPn'], df_y['phiMn'])
        
        # Suppress warnings for division by zero inside the contour boundary
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = ((np.cos(T) / mx_cap)**alpha + (np.sin(T) / my_cap)**alpha) ** (1.0 / alpha)
            R = np.where(denom > 0, 1.0 / denom, 0.0)
            
        X = R * np.cos(T)
        Y = R * np.sin(T)
        Z = P
        
        return X, Y, Z


# ──────────────────────────────────────────────────────────────────────────────
# SHEAR & TORSION HELPER  (all in MKS: ksc, ton, cm)
# ──────────────────────────────────────────────────────────────────────────────
def shear_torsion_check(engine, Pu_ton, Vux_ton, Vuy_ton, Tu_tonm,
                        tie_dia_mm, tie_legs, is_seismic, Lu_x_m):
    """
    ACI 318-19 shear check with exact Table 22.5.5.1 equation via MPa conversion.
    """
    fc   = engine.fc
    fy   = engine.fy
    phi  = PHI_SHEAR
    Pu_kgf = Pu_ton * 1_000.0

    d_tie  = tie_dia_mm / 10.0  # cm
    At     = math.pi * d_tie**2 / 4.0  # cm²
    Av     = tie_legs * At              # cm² per stirrup set

    if engine.shape == "Rectangular":
        dx = engine.h - engine.d_prime
        dy = engine.b - engine.d_prime
        bwx = engine.b
        bwy = engine.h
        Acp = engine.b * engine.h
        pcp = 2.0 * (engine.b + engine.h)
        min_dim = min(engine.b, engine.h)
    else:
        dx = dy = 0.8 * engine.D
        bwx = bwy = engine.D
        Acp = engine.Ag
        pcp = math.pi * engine.D
        min_dim = engine.D

    # ── Concrete shear capacity (ACI 318-19 Table 22.5.5.1) ────────────
    # Convert parameters to MPa for the standard ACI formula
    fc_mpa = fc / 10.197
    stress_pu_mpa = (Pu_kgf / engine.Ag) / 10.197
    
    # ACI limit: Nu/(6*Ag) cannot exceed 0.05*f'c (in MPa) [ACI 22.5.5.1.1]
    stress_pu_mpa = min(stress_pu_mpa, 0.05 * fc_mpa)

    # Vc stress in MPa: 0.17·√f'c + Nu/(6·Ag)
    vc_stress_mpa = 0.17 * math.sqrt(fc_mpa) + (stress_pu_mpa / 6.0)
    
    # Convert back to MKS stress (ksc) and calculate capacity
    vc_stress_ksc = vc_stress_mpa * 10.197
    
    Vcx_kgf = vc_stress_ksc * bwx * dx
    Vcy_kgf = vc_stress_ksc * bwy * dy
    Vcx_ton = Vcx_kgf / 1_000.0
    Vcy_ton = Vcy_kgf / 1_000.0

    # ── Torsion threshold  ACI 318-19 §22.7.4.1 (MKS) ─────────────
    Tth_kgcm = phi * 0.265 * math.sqrt(fc) * (Acp**2 / pcp)
    Tth_tonm = Tth_kgcm / 100_000.0
    torsion_critical = Tu_tonm > Tth_tonm

    # ── Spacing limits ──────────────────────────────────────
    s_max_x = min(dx / 2.0, 60.0)
    s_max_y = min(dy / 2.0, 60.0)

    s_seismic = 999.0
    if is_seismic:
        # ACI 318-19 §18.7.5.3 (SMF): s <= min(b_min/4, 6·d_b of INDIVIDUAL bar, s_o)
        # NOTE: s_o = 100 + (350-hx)/3 [mm] (100..150 mm) is capped at 15 cm here - verify hx.
        s_seismic = min(min_dim / 4.0, 6.0 * engine.db_cm, 15.0)

    # ── Required stirrup spacing ──────────────────────────
    Vsx_req_ton = max(0.0, Vux_ton / phi - Vcx_ton)
    Vsy_req_ton = max(0.0, Vuy_ton / phi - Vcy_ton)

    # [REVIEW] Added: tighter max spacing when Vs > 0.33*sqrt(f'c)[MPa]*bw*d  (≈1.06*sqrt(f'c) ksc)
    # and minimum shear reinforcement Av,min (0.062*sqrt(f'c)/0.35 MPa -> 0.2*sqrt(f'c)/3.57 ksc)
    vs_lim_x = 1.06 * math.sqrt(fc) * bwx * dx / 1_000.0
    vs_lim_y = 1.06 * math.sqrt(fc) * bwy * dy / 1_000.0
    if Vsx_req_ton > vs_lim_x: s_max_x = min(dx / 4.0, 30.0)
    if Vsy_req_ton > vs_lim_y: s_max_y = min(dy / 4.0, 30.0)
    av_min_coef = max(0.2 * math.sqrt(fc), 3.57)
    if Vux_ton > 0.5 * phi * Vcx_ton: s_max_x = min(s_max_x, Av * fy / (av_min_coef * bwx))
    if Vuy_ton > 0.5 * phi * Vcy_ton: s_max_y = min(s_max_y, Av * fy / (av_min_coef * bwy))

    def req_spacing(Vs_ton, Av_cm2, d_cm):
        Vs_kgf = Vs_ton * 1_000.0
        if Vs_kgf < 1.0: return 999.0
        return (Av_cm2 * fy * d_cm) / Vs_kgf

    sx_shear = req_spacing(Vsx_req_ton, Av, dx)
    sy_shear = req_spacing(Vsy_req_ton, Av, dy)

    s_gov = min(sx_shear, sy_shear, s_max_x, s_max_y, s_seismic)
    s_design = max(5.0, math.floor(s_gov / 2.5) * 2.5)

    Vsx_prov_ton = (Av * fy * dx) / (s_design * 1_000.0)
    Vsy_prov_ton = (Av * fy * dy) / (s_design * 1_000.0)
    phiVnx = phi * (Vcx_ton + Vsx_prov_ton)
    phiVny = phi * (Vcy_ton + Vsy_prov_ton)

    Vs_max_x_ton = 2.12 * math.sqrt(fc) * bwx * dx / 1_000.0
    Vs_max_y_ton = 2.12 * math.sqrt(fc) * bwy * dy / 1_000.0
    section_adequate_x = Vsx_req_ton <= Vs_max_x_ton
    section_adequate_y = Vsy_req_ton <= Vs_max_y_ton

    return {
        'dx': dx, 'dy': dy, 'bwx': bwx, 'bwy': bwy,
        'Av': Av, 'At': At, 'd_tie': d_tie,
        'Vcx_ton': Vcx_ton, 'Vcy_ton': Vcy_ton,
        'Vsx_prov_ton': Vsx_prov_ton, 'Vsy_prov_ton': Vsy_prov_ton,
        'phiVnx': phiVnx, 'phiVny': phiVny,
        'Tth_tonm': Tth_tonm, 'torsion_critical': torsion_critical,
        's_design': s_design, 's_seismic': s_seismic,
        's_max_x': s_max_x, 's_max_y': s_max_y,
        'sx_shear': sx_shear, 'sy_shear': sy_shear,
        'Vs_max_x_ton': Vs_max_x_ton, 'Vs_max_y_ton': Vs_max_y_ton,
        'section_adequate_x': section_adequate_x,
        'section_adequate_y': section_adequate_y,
        'x_ok': phiVnx >= Vux_ton,
        'y_ok': phiVny >= Vuy_ton,
        'Acp': Acp, 'pcp': pcp,
        'min_dim': min_dim,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LAP SPLICE (ACI 318-19 §25.5 / §25.6.1.5 for bundles)
# ──────────────────────────────────────────────────────────────────────────────
def lap_splice_lengths(fc_ksc, fy_ksc, db_cm, n_per_bundle=1):
    """
    Lap splice lengths in cm (MKS input, ACI 318-19 evaluated in SI then converted).

    [REVIEW] Development length, ACI 318-19 Eq. 25.4.2.4a (SI, MPa / mm):
        ld = fy·ψt·ψe·ψs / (1.1·λ·√f'c·[(cb+Ktr)/db]) · db   ≥ 300 mm
    (the previous code used the US-customary 3/40 coefficient with MPa values,
     which gave ld ≈ 2 db and always returned the 30 cm minimum.)
    Assumptions: λ = ψt = ψe = ψs = 1, (cb+Ktr)/db = 2.5 (upper limit - verify!).

    Class B tension splice = 1.3·ld.
    Compression splice (ACI 25.5.5.1): fy ≤ 420 MPa: 0.071·fy·db ; else (0.13·fy − 24)·db ; ≥ 300 mm.
    Bundled bars: the splice length of the INDIVIDUAL bar is increased by 20 % (3-bar)
    or 33 % (4-bar); NOT by √n.  Individual splices in a bundle must not overlap.
    """
    n      = max(1, min(4, int(n_per_bundle)))
    fy_mpa = fy_ksc / 10.197
    fc_mpa = fc_ksc / 10.197
    db_mm  = db_cm * 10.0
    cb_ktr_over_db = 2.5

    ld_single_cm = max(fy_mpa / (1.1 * math.sqrt(fc_mpa) * cb_ktr_over_db) * db_mm / 10.0, 30.0)
    l_splice_B_single = max(1.3 * ld_single_cm, 30.0)

    if fy_mpa <= 420.0:
        lc_single_mm = 0.071 * fy_mpa * db_mm
    else:
        lc_single_mm = (0.13 * fy_mpa - 24.0) * db_mm
    l_compression_single = max(lc_single_mm / 10.0, 30.0)

    pct = {1: 1.00, 2: 1.00, 3: 1.20, 4: 1.33}[n]
    db_eq_cm         = db_cm * math.sqrt(n)           # spacing / cover checks only
    ld_bundle_cm     = max(ld_single_cm * pct, 30.0)
    l_splice_B_bundle    = max(1.3 * ld_bundle_cm, 30.0)
    l_compression_bundle = max(l_compression_single * pct, 30.0)

    return {
        'ld_single':               ld_single_cm,
        'l_splice_B_single':       l_splice_B_single,
        'l_compression_single':    l_compression_single,
        'db_eq_cm':                db_eq_cm,
        'ld_bundle_sqrt_n':        ld_bundle_cm,   # kept for backward compatibility (= % method)
        'ld_bundle_pct':           ld_bundle_cm,
        'ld_bundle_gov':           ld_bundle_cm,
        'l_splice_B_bundle':       l_splice_B_bundle,
        'l_compression_bundle':    l_compression_bundle,
        'lap_splice_not_permitted': False,
        'bundle_special_care':     (n >= 3),
        'n_per_bundle':            n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RC Column Pro", layout="wide", page_icon="🏗️")
st.title("🏗️ RC Column Pro — Biaxial & Sway Analysis (ACI 318-19, MKS)")

col1, col_main = st.columns([1, 2.5])

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL – INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with col1:
    with st.expander("1. Section & Reinforcement", expanded=True):
        shape = st.radio("Section Shape", ["Rectangular", "Circular"], horizontal=True, key="shape")
        fc    = st.number_input("f'c (ksc)", value=280, min_value=140, max_value=700, key="fc_main")
        fy    = st.number_input("fy  (ksc)", value=4000, min_value=2400, max_value=6000, key="fy_main")

        if shape == "Rectangular":
            c1, c2 = st.columns(2)
            b = c1.number_input("Width b (cm) [X]", value=40, min_value=20, key="b_main")
            h = c2.number_input("Depth h (cm) [Y]", value=60, min_value=20, key="h_main")
            layout = st.selectbox("Rebar Layout", ["4-Faces (Uniform)", "2-Faces (Top/Bottom)"], key="layout_main")
            if layout == "2-Faces (Top/Bottom)":
                n_bars = st.number_input("Total Bars (even)", 4, 40, 8, step=2, key="n_bars_2f")
                nx = ny = 0
            else:
                c3, c4 = st.columns(2)
                nx = c3.number_input("Bars on X-faces (nx)", 2, 20, 3, key="nx_main",
                                     help="Bars along top & bottom faces (including corners)")
                ny = c4.number_input("Bars on Y-faces (ny)", 2, 20, 4, key="ny_main",
                                     help="Bars along left & right faces (including corners)")
                # nx bars on top + nx on bottom + (ny-2) on left + (ny-2) on right
                # Corners are shared with top/bottom, so excluded from side count.
                n_bars = 2 * nx + 2 * (ny - 2)
                st.caption(f"→ **{n_bars} bundle positions** "
                           f"({nx} top + {nx} bottom + {ny-2} left + {ny-2} right)")
        else:
            b = h = st.number_input("Diameter D (cm)", value=50, min_value=30, key="D_main")
            layout   = "Circular"
            n_bars   = st.number_input("Total Bars (≥ 6)", 6, 60, 8, key="n_bars_circ")
            nx = ny  = 0

        db     = st.selectbox("Bar Size (mm)", [16, 20, 25, 28, 32], index=2, key="db_main")
        cover  = st.number_input("Clear Cover (cm)", value=4.0, min_value=4.0, key="cover_main")

        st.markdown("---")
        st.markdown("**🔗 Bundled Bars (ACI 318-19 §26.6.3)**")
        n_per_bundle = st.radio(
            "Bars per Bundle",
            [1, 2, 3, 4],
            index=0,
            horizontal=True,
            key="n_per_bundle_main",
            help=(
                "ACI 318-19 §26.6.3.1 – max 4 bars per bundle in columns.\n"
                "Each bundle counts as 1 position; total steel = positions × bars/bundle.\n"
                "Lap: ld ของเส้นเดี่ยว ×1.20 (3 เส้น) / ×1.33 (4 เส้น), ห้ามซ้อนทับกันในกลุ่ม."
            ),
            format_func=lambda x: {1: "1 (single)", 2: "2-bar", 3: "3-bar", 4: "4-bar"}[x]
        )
        if n_per_bundle > 1:
            db_eq_display = db / 10.0 * math.sqrt(n_per_bundle)
            st.info(
                f"**Bundle summary:** {n_per_bundle} × DB{db} per position\n\n"
                f"d_b,eq = {db:.0f}×√{n_per_bundle} = **{db_eq_display*10:.1f} mm** "
                f"(ใช้กับ spacing/cover เท่านั้น — lap ใช้ ld ของเส้นเดี่ยว × 1.20 / 1.33)"
            )
            if n_per_bundle == 4:
                st.warning("⚠️ **4-bar bundle:** lap แต่ละเส้นแยกกัน (ไม่ซ้อนทับกันในกลุ่มเดียว) "
                           "ด้วย ld × 1.33 — ห้าม lap ทั้งกลุ่มพร้อมกัน; พิจารณา coupler (ตรวจเลขข้อกับมาตรฐานฉบับที่ใช้)")

    with st.expander("2. Loads & Frame Type", expanded=True):
        Pu    = st.number_input("Factored Axial Pu (ton)", value=150.0, min_value=0.0, key="Pu_main")
        c5, c6 = st.columns(2)
        Mux   = c5.number_input("Mux (ton-m)", value=15.0, key="Mux_main", help="Moment about X-axis (bending in h-direction)")
        Muy   = c6.number_input("Muy (ton-m)", value=10.0, key="Muy_main", help="Moment about Y-axis (bending in b-direction)")

        st.markdown("---")
        frame_type = st.radio("Frame Type", ["Non-Sway (Braced)", "Sway (Unbraced)"], horizontal=True, key="frame_type_main")

        if frame_type == "Non-Sway (Braced)":
            cx1, cx2, cx3 = st.columns(3)
            Lu_x  = cx1.number_input("Lu X (m)", value=4.0, step=0.5, key="Lu_x_ns")
            K_x   = cx2.number_input("K  X", value=1.0, step=0.1, min_value=0.5, key="K_x_main")
            Cm_x  = cx3.number_input("Cm X", value=1.0, step=0.05, min_value=0.4, key="Cm_x_main")

            cy1, cy2, cy3 = st.columns(3)
            Lu_y  = cy1.number_input("Lu Y (m)", value=4.0, step=0.5, key="Lu_y_ns")
            K_y   = cy2.number_input("K  Y", value=1.0, step=0.1, min_value=0.5, key="K_y_main")
            Cm_y  = cy3.number_input("Cm Y", value=1.0, step=0.05, min_value=0.4, key="Cm_y_main")

            beta_d = st.slider("β_d (Sustained Load Ratio)", 0.0, 1.0, 0.6, key="beta_d_ns")
            delta_sx = delta_sy = 1.0

        else:  # Sway
            sway_method = st.radio("δs Method", ["Stability Index Q", "ΣPu & ΣPc", "Direct Input"], horizontal=True, key="sway_method")
            if sway_method == "Stability Index Q":
                Q_val     = st.number_input("Q", 0.0, 0.99, 0.05, step=0.01, key="Q_sway")
                delta_sx  = delta_sy = max(1.0, 1.0 / (1.0 - Q_val))
                st.info(f"δs = {delta_sx:.3f}")
            elif sway_method == "ΣPu & ΣPc":
                cs1, cs2 = st.columns(2)
                sum_Pu = cs1.number_input("ΣPu (ton)", 0.1, value=500.0, key="sum_Pu_sway")
                sum_Pc = cs2.number_input("ΣPc (ton)", 0.1, value=2000.0, key="sum_Pc_sway")
                limit  = 0.75 * sum_Pc
                if sum_Pu >= limit:
                    st.error("⚠️ ΣPu ≥ 0.75ΣPc → Frame unstable!")
                    delta_sx = delta_sy = 999.0
                else:
                    delta_sx = delta_sy = max(1.0, 1.0 / (1.0 - sum_Pu / limit))
                    st.info(f"δs = {delta_sx:.3f}")
            else:
                cs1, cs2 = st.columns(2)
                delta_sx = cs1.number_input("δs X", value=1.2, step=0.05, min_value=1.0, key="delta_sx_dir")
                delta_sy = cs2.number_input("δs Y", value=1.2, step=0.05, min_value=1.0, key="delta_sy_dir")

            if max(delta_sx, delta_sy) > 1.5:   # [REVIEW] ACI 6.6.4.6.2
                st.warning('⚠️ δs > 1.5 → ACI 318-19 §6.6.4.6.2 ต้องใช้ second-order analysis (ไม่ใช้ magnifier)')
            # Sway frames: non-sway magnifier still required for non-sway component
            cx1, cx2, cx3 = st.columns(3)
            Lu_x  = cx1.number_input("Lu X (m)", value=4.0, step=0.5, key="Lu_x_sw")
            K_x   = cx2.number_input("K  X (NS)", value=0.5, step=0.05, min_value=0.1, key="K_x_sw")
            Cm_x  = cx3.number_input("Cm X", value=1.0, step=0.05, min_value=0.4, key="Cm_x_sw")
            cy1, cy2, cy3 = st.columns(3)
            Lu_y  = cy1.number_input("Lu Y (m)", value=4.0, step=0.5, key="Lu_y_sw")
            K_y   = cy2.number_input("K  Y (NS)", value=0.5, step=0.05, min_value=0.1, key="K_y_sw")
            Cm_y  = cy3.number_input("Cm Y", value=1.0, step=0.05, min_value=0.4, key="Cm_y_sw")
            beta_d = st.slider("β_d", 0.0, 1.0, 0.6, key="beta_d_sw")

    with st.expander("3. Shear, Torsion & Seismic", expanded=True):
        st.subheader("🛡️ Shear Design")
        cv5, cv6 = st.columns(2)
        vux_ton = cv5.number_input("Vux (ton)", value=5.0, step=1.0, help="เฉือนที่เกิดคู่กับ Mux — แรงขนานด้านลึก h (bw=b, d=h−d')", key="vux_main")
        vuy_ton = cv6.number_input("Vuy (ton)", value=5.0, step=1.0, help="เฉือนที่เกิดคู่กับ Muy — แรงขนานด้านลึก b (bw=h, d=b−d')", key="vuy_main")

        c7, c8 = st.columns(2)
        tie_dia  = c7.selectbox("Tie ⌀ (mm)", [9, 12, 16], index=0, format_func=lambda x: f"RB{x}" if x < 10 else f"DB{x}", key="tie_dia_main")
        tie_legs = c8.number_input("Stirrup Legs", 2, 10, 2, key="tie_legs_main")

        st.markdown("---")
        tu_tonm    = st.number_input("Tu (ton-m)", value=0.0, step=0.5, key="tu_main")
        is_seismic = st.toggle("Seismic Detailing (SMF)", value=True, key="seismic_main")


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE & CALCULATION
# ══════════════════════════════════════════════════════════════════════════════
engine  = RCColumn(shape, layout, b, h, fc, fy, db, n_bars, nx, ny, cover, n_per_bundle,
                   tie_cm=tie_dia / 10.0)
df_x, phi_pn_max = engine.solve_pm(axis='X')
df_y, _          = engine.solve_pm(axis='Y')

# ── Minimum eccentricity moments  ACI 318-19 §6.6.4.5.4 ──────────────────────
e_min_x   = Pu * (0.015 + 0.03 * h / 100.0)
e_min_y   = Pu * (0.015 + 0.03 * b / 100.0)
Mu_x_dsgn = max(abs(Mux), e_min_x)   # [REVIEW] use |Mux|
Mu_y_dsgn = max(abs(Muy), e_min_y)   # [REVIEW] use |Muy|

# ── Moment magnification ──────────────────────────────────────────────────────
kl_rx, Pcx, del_x, Ise_x, EIx = engine.slenderness_magnifier(
    Pu, K_x, Lu_x, 'X', Cm_x, beta_d)
kl_ry, Pcy, del_y, Ise_y, EIy = engine.slenderness_magnifier(
    Pu, K_y, Lu_y, 'Y', Cm_y, beta_d)

if frame_type == "Non-Sway (Braced)":
    Mcx = del_x * Mu_x_dsgn if kl_rx > 22.0 else Mu_x_dsgn
    Mcy = del_y * Mu_y_dsgn if kl_ry > 22.0 else Mu_y_dsgn
else:  # Sway: M2 = δs·M2s + δns·M2ns  (simplified: apply δs to total Mu)
    Mcx_sway = delta_sx * Mu_x_dsgn
    Mcy_sway = delta_sy * Mu_y_dsgn
    Mcx = del_x * Mcx_sway if kl_rx > 22.0 else Mcx_sway
    Mcy = del_y * Mcy_sway if kl_ry > 22.0 else Mcy_sway

# ── Shear / torsion ───────────────────────────────────────────────────────────
shear = shear_torsion_check(engine, Pu, vux_ton, vuy_ton,
                             tu_tonm, tie_dia, tie_legs, is_seismic, Lu_x)

# ── Biaxial interaction (PCA load-contour) ───────────────────────────────────
error_status  = None
is_safe       = False
demand_ratio  = 999.0
phi_Mnox = phi_Mnoy = 0.0
alpha = 1.5

if Pu > phi_pn_max:
    error_status = (f"Axial load Pu = {Pu:.1f} t exceeds section capacity "
                    f"φPn,max = {phi_pn_max:.1f} t")
else:


    try:
        # np.interp clamps values outside the bounds to the boundary values
        phi_Mnox = float(np.interp(Pu, df_x['phiPn'], df_x['phiMn']))
        phi_Mnoy = float(np.interp(Pu, df_y['phiPn'], df_y['phiMn']))
    

        if phi_Mnox <= 0 or phi_Mnoy <= 0:
            error_status = "Pu is outside the valid range of the P-M interaction curve."
        else:
            alpha        = engine.get_alpha(Pu)
            demand_ratio = ((Mcx / phi_Mnox)**alpha +
                            (Mcy / phi_Mnoy)**alpha)
            is_safe      = demand_ratio <= 1.0
    except Exception as e:
        error_status = f"Interpolation error: {e}"

actual_space, min_req_space, space_ok = engine.check_clear_spacing(nx, ny)
rho_pct = engine.rho * 100.0
rho_max = 6.0 if is_seismic else 8.0     # [REVIEW] ACI 18.7.4.1: ≤ 6% for SMF columns
rho_ok  = 1.0 <= rho_pct <= rho_max

# ── Lap splice lengths ────────────────────────────────────────────────────────
splice_data = lap_splice_lengths(fc, fy, engine.db_cm, n_per_bundle)
# Convenience aliases used in UI (prefer bundle values when bundled)
l_splice_B    = splice_data['l_splice_B_bundle']    if n_per_bundle > 1 else splice_data['l_splice_B_single']
l_compression = splice_data['l_compression_bundle'] if n_per_bundle > 1 else splice_data['l_compression_single']


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL – RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with col_main:
    st.markdown("### 📋 Executive Design Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    bundle_tag = f" ({n_per_bundle}×)" if n_per_bundle > 1 else ""
    m1.metric(f"Steel ρ{bundle_tag}",   f"{rho_pct:.2f} %",
              "✅ OK" if rho_ok else "❌ Fail",
              delta_color="normal" if rho_ok else "inverse")
    spacing_label = "Bundle Spacing" if n_per_bundle > 1 else "Clear Spacing"
    m2.metric(spacing_label, f"{actual_space:.2f} cm",
              "✅ OK" if space_ok else "⚠️ Tight",
              delta_color="normal" if space_ok else "inverse")
    m3.metric("Mcx (Magnified)", f"{Mcx:.1f} t-m",
              f"×{max(Mcx / Mu_x_dsgn, 1.0):.2f}", delta_color="off")
    m4.metric("Mcy (Magnified)", f"{Mcy:.1f} t-m",
              f"×{max(Mcy / Mu_y_dsgn, 1.0):.2f}", delta_color="off")
    shear_label = "SMF Tie Spacing" if is_seismic else "Shear Tie Spacing"
    s_fin = shear['s_design']
    s_ok  = shear['x_ok'] and shear['y_ok']
    m5.metric(shear_label, f"@ {s_fin:.1f} cm",
              "✅ OK" if s_ok else "❌ Fail",
              delta_color="normal" if s_ok else "inverse")

    st.markdown("---")

    # Alerts
    # Bundled-bar code checks
    if n_per_bundle > 1:
        st.info(
            f"📦 **Bundled bars active:** {n_per_bundle} bars/bundle × "
            f"{engine.n_positions} positions = **{engine.n_bars} total bars** | "
            f"Total Ast = **{engine.total_as:.2f} cm²** | ρ = **{rho_pct:.2f}%**\n\n"
            f"d_b,eq = {engine.db_eq_cm*10:.1f} mm | "
            f"Cover to bundle centroid = {engine.d_prime:.2f} cm"
        )
    if n_per_bundle == 3:
        st.warning("⚠️ **3-bar bundle:** ความยาว lap เพิ่ม 20% (ต่อเส้น) และห้ามซ้อนทับกันในกลุ่มเดียว.")
    if shear['torsion_critical']:
        st.warning(f"🌪️ **Torsion Design Required:** Tu = {tu_tonm:.2f} t-m > "
                   f"Tth = {shear['Tth_tonm']:.3f} t-m. "
                   "Provide closed stirrups with 135° hooks + extra longitudinal bars.")
    if not space_ok:
        st.warning(f"⚠️ Bar spacing {actual_space:.2f} cm < minimum {min_req_space:.2f} cm "
                   "— Honeycombing risk. Reduce bar count or increase section width.")
    if not rho_ok:
        st.warning(f"⚠️ ρ = {rho_pct:.2f}% is {'below 1.0%' if rho_pct < 1.0 else f'above {rho_max:.1f}%'}")
    if del_x > 10 or del_y > 10:
        st.error("⚠️ Slenderness magnifier > 10 — Column is critically slender. "
                 "Increase section or reduce height.")

    if error_status:
        st.error(f"### ❌ CAPACITY EXCEEDED\n{error_status}")
    elif is_safe:
        st.success(f"### ✅ SAFE — Biaxial Demand Ratio = **{demand_ratio:.3f}** ≤ 1.0  (α = {alpha:.3f})")
    else:
        st.error(f"### ❌ UNSAFE — Biaxial Demand Ratio = **{demand_ratio:.3f}** > 1.0  (α = {alpha:.3f})")

    # ══════════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════════
    # ── Input → Design → Magnified moment table (explains why plotted Mc ≠ typed Mu) ──
    st.markdown("##### 🔎 ที่มาของโมเมนต์ที่ใช้ตรวจสอบและ plot  (ค่าที่กรอก → ค่าออกแบบ → ค่าหลังขยาย)")
    _mom = pd.DataFrame([
        {"แกน": "X  (Mux · ดัดรอบแกน X · ความลึก h)", "Mu ที่กรอก": abs(Mux), "Mu,min = Pu·e_min": e_min_x,
         "Mu ออกแบบ = max": Mu_x_dsgn, "kl/r": kl_rx, "ตัวคูณรวม": Mcx / max(Mu_x_dsgn, 1e-9), "Mc ที่ plot (t-m)": Mcx},
        {"แกน": "Y  (Muy · ดัดรอบแกน Y · ความลึก b)", "Mu ที่กรอก": abs(Muy), "Mu,min = Pu·e_min": e_min_y,
         "Mu ออกแบบ = max": Mu_y_dsgn, "kl/r": kl_ry, "ตัวคูณรวม": Mcy / max(Mu_y_dsgn, 1e-9), "Mc ที่ plot (t-m)": Mcy},
    ]).round(3)
    st.dataframe(_mom, use_container_width=True, hide_index=True)
    st.caption("กราฟทุกกราฟ plot **Mc** (โมเมนต์หลังขยายด้วย δ และไม่ต่ำกว่า Mu,min) เป็นจุด ✚ ส่วนค่าที่กรอกจริง (Mu) แสดงเป็นวงกลมโปร่ง ○ "
               "— Mc จึงมากกว่า Mu ได้เมื่อ kl/r > 22 หรือ Mu,min ควบคุม")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 Overview & 3D",
        "📊 P-M Interaction",
        "🧊 Section Detail",
        "🌪️ Shear & Seismic",
        "📝 Calc Report",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 – 3D surface + 2D contour + P-Mx / P-My projections
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 🌐 3D Biaxial Failure Surface (PCA Load-Contour)")

        if not error_status:
            try:
                mx_m, my_m, p_m = engine.generate_3d_surface(df_x, df_y, alpha)
                if mx_m.size:
                    fig3d = go.Figure()
                    fig3d.add_trace(go.Surface(
                        x=mx_m, y=my_m, z=p_m,
                        colorscale='Plasma', opacity=0.75,
                        showscale=False, name='Capacity Surface',
                        lighting=dict(ambient=0.6, diffuse=0.8,
                                      roughness=0.4, specular=0.5)))
                    mc = '#2ecc71' if is_safe else '#e74c3c'
                    fig3d.add_trace(go.Scatter3d(
                        x=[Mcx], y=[Mcy], z=[Pu],
                        mode='markers+text',
                        marker=dict(size=9, color=mc, symbol='diamond',
                                    line=dict(width=2, color='white')),
                        text=["Design Mc"], textposition="top center",
                        name='Design Mc (หลังขยาย)'))
                    fig3d.add_trace(go.Scatter3d(
                        x=[abs(Mux)], y=[abs(Muy)], z=[Pu], mode='markers+text',
                        marker=dict(size=7, color='#7f8c8d', symbol='circle-open',
                                    line=dict(width=3, color='#7f8c8d')),
                        text=["Input Mu"], textposition="bottom center",
                        name='Input Mu (ไม่ขยาย)'))
                    fig3d.add_trace(go.Scatter3d(
                        x=[Mcx, Mcx], y=[Mcy, Mcy], z=[0, Pu],
                        mode='lines', line=dict(color=mc, width=3, dash='dot'),
                        showlegend=False))
                    fig3d.update_layout(
                        scene=dict(
                            xaxis_title='Mx (t-m)',
                            yaxis_title='My (t-m)',
                            zaxis_title='P (ton)',
                            aspectmode='manual',
                            aspectratio=dict(x=1, y=1, z=1.2),
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
                        margin=dict(l=0, r=0, b=0, t=0), height=560)
                    st.plotly_chart(fig3d, use_container_width=True)
            except Exception as e:
                st.info(f"3D surface: {e}")
        else:
            st.error("Cannot draw 3D surface — Pu exceeds section capacity.")

        st.markdown("---")
        st.markdown(f"#### 🎯 2D PCA Contour at Pu = {Pu:.2f} ton")

        col_a, col_b, col_c = st.columns([1, 1, 1.5])
        col_a.metric("Demand Ratio", f"{demand_ratio:.3f}",
                     "SAFE" if is_safe else "UNSAFE",
                     delta_color="inverse")
        col_b.metric("α (PCA Exponent)", f"{alpha:.3f}")

        with col_c:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=min(demand_ratio, 2.0),
                title={'text': "Capacity Utilisation", 'font': {'size': 13}},
                gauge={'axis': {'range': [0, 1.5]},
                       'bar': {'color': '#2c3e50'},
                       'steps': [
                           {'range': [0, 0.8],  'color': 'rgba(46,204,113,0.3)'},
                           {'range': [0.8, 1.0], 'color': 'rgba(241,196,15,0.3)'},
                           {'range': [1.0, 1.5], 'color': 'rgba(231,76,60,0.3)'}],
                       'threshold': {'line': {'color': 'red', 'width': 4},
                                     'thickness': 0.75, 'value': 1.0}}))
            fig_g.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

        # 2D PCA contour
        if phi_Mnox > 0 and phi_Mnoy > 0:
            mx_c = np.linspace(0, phi_Mnox, 120)
            my_c = phi_Mnoy * np.maximum(0, 1 - (mx_c / phi_Mnox)**alpha)**(1.0 / alpha)
            mc = '#2ecc71' if is_safe else '#e74c3c'

            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=mx_c, y=my_c, mode='lines', name=f'Capacity (α={alpha:.2f})',
                line=dict(color='#8e44ad', width=3),
                fill='tozeroy', fillcolor='rgba(142,68,173,0.10)'))
            fig_c.add_trace(go.Scatter(
                x=[phi_Mnox, 0], y=[0, phi_Mnoy], mode='markers+text',
                name='Uniaxial Caps',
                marker=dict(color='#2c3e50', size=9, symbol='square'),
                text=[f'φMnox={phi_Mnox:.1f}', f'φMnoy={phi_Mnoy:.1f}'],
                textposition=['top right', 'top right']))
            fig_c.add_trace(go.Scatter(
                x=[Mcx], y=[Mcy], mode='markers+text', name='Design point Mc (หลังขยาย)',
                marker=dict(color=mc, size=14, symbol='cross',
                            line=dict(width=2, color='white')),
                text=["Design Point"], textposition="top right"))
            fig_c.add_trace(go.Scatter(
                x=[abs(Mux)], y=[abs(Muy)], mode='markers+text', name='Input Mu (ที่กรอก)',
                marker=dict(color='#7f8c8d', size=11, symbol='circle-open', line=dict(width=3)),
                text=["Input Mu"], textposition="bottom right"))
            fig_c.add_shape(type="line", x0=0, y0=0, x1=Mcx, y1=Mcy,
                            line=dict(color=mc, width=2, dash='dashdot'))

            mx_rng = max(phi_Mnox, Mcx, abs(Mux)) * 1.2
            my_rng = max(phi_Mnoy, Mcy, abs(Muy)) * 1.2
            fig_c.update_layout(
                xaxis=dict(title='Magnified Mcx (ton-m)', range=[0, mx_rng],
                           showgrid=True, gridcolor='rgba(0,0,0,0.06)',
                           zeroline=True, zerolinewidth=2),
                yaxis=dict(title='Magnified Mcy (ton-m)', range=[0, my_rng],
                           showgrid=True, gridcolor='rgba(0,0,0,0.06)',
                           zeroline=True, zerolinewidth=2),
                plot_bgcolor='white', paper_bgcolor='white', height=450,
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.85)',
                            bordercolor='gray', borderwidth=1),
                margin=dict(l=40, r=40, t=20, b=40))
            st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📈 Uniaxial P-M Projections")
        col_pmx, col_pmy = st.columns(2)

        def pm_side_chart(df, Mc, label, color, Mu_in=None):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['phiMn'], y=df['phiPn'], mode='lines',
                line=dict(color=color, width=2.5),
                fill='tozerox', fillcolor=f'rgba({",".join(str(c) for c in [41,128,185])},0.12)',
                name='Capacity'))
            fig.add_trace(go.Scatter(
                x=[Mc], y=[Pu], mode='markers',
                marker=dict(color='#e74c3c', size=12, symbol='cross',
                            line=dict(width=2, color='white')),
                name='Mc', hovertemplate="Mc (หลังขยาย): %{x:.2f} t-m<extra></extra>"))
            if Mu_in is not None:
                fig.add_trace(go.Scatter(
                    x=[abs(Mu_in)], y=[Pu], mode='markers',
                    marker=dict(color='#7f8c8d', size=11, symbol='circle-open', line=dict(width=3)),
                    name='Mu', hovertemplate="Mu (ที่กรอก): %{x:.2f} t-m<extra></extra>"))
            fig.add_shape(type="line", x0=0, y0=Pu, x1=Mc, y1=Pu,
                          line=dict(color="#e74c3c", width=1, dash="dot"))
            fig.add_shape(type="line", x0=Mc, y0=df['phiPn'].min() * 1.05, x1=Mc, y1=Pu,
                          line=dict(color="#e74c3c", width=1, dash="dot"))
            p_lo2 = df['phiPn'].min() * 1.1 if df['phiPn'].min() < 0 else -10
            p_hi2 = df['phiPn'].max() * 1.1
            fig.update_layout(
                title=dict(text=f"<b>{label}</b>", font=dict(size=13)),
                xaxis=dict(title=f'{label.split()[0]} (ton-m)', rangemode='tozero',
                           showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                           zeroline=True, zerolinewidth=2),
                yaxis=dict(title='Axial φPn (ton)', range=[p_lo2, p_hi2],
                           showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                           zeroline=True, zerolinewidth=2),
                plot_bgcolor='white', paper_bgcolor='white',
                height=380, showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20))
            return fig

        with col_pmx:
            st.plotly_chart(pm_side_chart(df_x, Mcx, "P-Mx (about X-axis, depth h)", "#2980b9", Mux),
                            use_container_width=True)
        with col_pmy:
            st.plotly_chart(pm_side_chart(df_y, Mcy, "P-My (about Y-axis, depth b)", "#27ae60", Muy),
                            use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 – Full P-M diagram with ACI limits
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 📊 P-M Interaction Diagram")
        show_bounds = st.toggle("Show ACI ρ-limits (min & max)", value=True, key="show_bounds_t2")
        show_keys   = st.toggle("Label Key Points", value=True, key="show_keys_t2")

        fig_pm = go.Figure()

        if show_bounds:
            def make_ref_df(target_rho):
                """
                Compute the P-M envelope for this SAME section geometry (b, h, fc, fy, cover)
                but with steel ratio forced to target_rho, using the same bar diameter (db)
                as the design. This guarantees d_prime is identical, so the design curve
                (at the actual ρ) will always lie between the 1% and 8% reference curves.
                n_per_bundle=1 always — these are code-limit reference curves, not design.
                """
                try:
                    Ag   = (math.pi * h**2 / 4) if shape == "Circular" else float(b) * float(h)
                    tAs  = target_rho * Ag
                    # Use the actual bar size from the design
                    ref_n = max(4, round(tAs / engine.as_single))
                    if shape == "Rectangular":
                        ref_nx = max(2, round(math.sqrt(ref_n * b / h)))
                        ref_ny = max(2, round((ref_n - 2 * ref_nx) / 2) + 2)
                        re = RCColumn(shape, "4-Faces (Uniform)",
                                      b, h, fc, fy, db, 0, ref_nx, ref_ny, cover, n_per_bundle=1, tie_cm=tie_dia / 10.0)
                    else:
                        re = RCColumn(shape, "Circular",
                                      b, h, fc, fy, db, max(6, ref_n), 0, 0, cover, n_per_bundle=1, tie_cm=tie_dia / 10.0)
                    rd, _ = re.solve_pm(axis='X')
                    return rd, re.rho * 100
                except Exception:
                    return pd.DataFrame(), target_rho * 100

            with st.spinner("Computing boundary curves…"):
                df_1, rho_1_actual = make_ref_df(0.01)
                df_8, rho_8_actual = make_ref_df(rho_max / 100.0)

            if not df_1.empty and not df_8.empty:
                # Ensure ρ=8% curve is always plotted OUTSIDE ρ=1% (sanity check)
                xp = list(df_8['phiMn']) + list(df_1['phiMn'])[::-1]
                yp = list(df_8['phiPn']) + list(df_1['phiPn'])[::-1]
                xp.append(xp[0]); yp.append(yp[0])
                fig_pm.add_trace(go.Scatter(
                    x=xp, y=yp, fill='toself',
                    fillcolor='rgba(46,204,113,0.10)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Optimal Zone (ρ min–max)', hoverinfo='skip'))
                for df_lim, nm, clr in [
                        (df_1, f'Min (ρ={rho_1_actual:.2f}%)', 'rgba(149,165,166,0.9)'),
                        (df_8, f'Max (ρ={rho_8_actual:.2f}%)', 'rgba(231,76,60,0.6)')]:
                    fig_pm.add_trace(go.Scatter(
                        x=df_lim['phiMn'], y=df_lim['phiPn'],
                        name=nm, mode='lines',
                        line=dict(color=clr, width=1.5), hoverinfo='skip'))

        for dfpm, nm, clr in [(df_x, 'X-Axis', '#2980b9'),
                               (df_y, 'Y-Axis', '#27ae60')]:
            fig_pm.add_trace(go.Scatter(
                x=dfpm['phiMn'], y=dfpm['phiPn'],
                name=nm, mode='lines', line=dict(color=clr, width=3.5),
                hovertemplate=f"<b>{nm}</b><br>φMn: %{{x:.2f}} t-m<br>φPn: %{{y:.2f}} ton<extra></extra>"))

        if show_keys and not df_x.empty:
            bal_idx = df_x['phiMn'].idxmax()
            anns = [
                dict(x=0, y=df_x['phiPn'].max(), text="Pure Compression",
                     showarrow=True, arrowhead=2, ax=60, ay=0,
                     font=dict(size=10, color='#7f8c8d')),
                dict(x=df_x.loc[bal_idx,'phiMn'], y=df_x.loc[bal_idx,'phiPn'],
                     text="Balance Point",
                     showarrow=True, arrowhead=2, ax=40, ay=-35,
                     font=dict(size=10, color='#7f8c8d')),
                dict(x=0, y=df_x['phiPn'].min(), text="Pure Tension",
                     showarrow=True, arrowhead=2, ax=60, ay=0,
                     font=dict(size=10, color='#7f8c8d')),
            ]
            fig_pm.update_layout(annotations=anns)

        fig_pm.add_trace(go.Scatter(
            x=[Mcx, Mcy], y=[Pu, Pu], mode='markers',
            marker=dict(color=['#e74c3c', '#e67e22'], size=14,
                        symbol='cross', line=dict(width=2, color='white')),
            name='Design Mc (Mcx บนเส้น X, Mcy บนเส้น Y)',
            hovertemplate="<b>Demand</b><br>Mc: %{x:.2f} t-m<br>Pu: %{y:.2f} ton<extra></extra>"))
        fig_pm.add_trace(go.Scatter(
            x=[abs(Mux), abs(Muy)], y=[Pu, Pu], mode='markers', name='Input Mux, Muy (ที่กรอก)',
            marker=dict(size=11, symbol='circle-open', color=['#e74c3c', '#e67e22'],
                        line=dict(width=3, color=['#e74c3c', '#e67e22']))))
        for Mc, clr in [(Mcx, '#e74c3c'), (Mcy, '#e67e22')]:
            fig_pm.add_shape(type="line", x0=0, y0=Pu, x1=Mc, y1=Pu,
                             line=dict(color=clr, width=1, dash='dot'))
            fig_pm.add_shape(type="line", x0=Mc, y0=0, x1=Mc, y1=Pu,
                             line=dict(color=clr, width=1, dash='dot'))

        all_pn = pd.concat([df_x['phiPn'], df_y['phiPn']])
        y_lo = all_pn.min() * 1.1 if all_pn.min() < 0 else -10
        fig_pm.update_layout(
            xaxis=dict(title='Design Moment φMn (ton-m)', rangemode='tozero',
                       showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                       zeroline=True, zerolinewidth=2),
            yaxis=dict(title='Design Axial φPn (ton)',
                       range=[y_lo, all_pn.max() * 1.1],
                       showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                       zeroline=True, zerolinewidth=2),
            plot_bgcolor='white', paper_bgcolor='white', height=660,
            hovermode='closest',
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='center', x=0.5,
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
            margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig_pm, use_container_width=True)

        if show_bounds:
            st.info(
                f"**ρ-limit bands explained:** Both boundary curves use the same section "
                f"(b={b} cm, h={h} cm, f'c={fc} ksc, fy={fy} ksc, DB{db}, cover={cover} cm) "
                f"but with steel forced to ρ = {rho_1_actual:.2f}% (grey, ACI §10.6.1.1 min) "
                f"and ρ = {rho_8_actual:.2f}% (red, ACI §10.6.1.1 max). "
                f"The design curve (ρ = **{rho_pct:.2f}%**) should always lie between them. "
                f"Reference curves use n_per_bundle = 1 regardless of the bundle setting."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 – Section detail drawing
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🧊 Cross-Section & BIM Cage")
        bx = [bar['x'] for bar in engine.bars]
        by = [bar['y'] for bar in engine.bars]
        cv2 = cover + engine.tie_cm / 2.0   # [REVIEW] tie centre-line, not clear cover

        dark = '#020617'; blue = '#38bdf8'; red = '#ef4444'; gold = '#fbbf24'
        orange = '#f97316'; violet = '#a78bfa'

        # Section outline coordinates (shared by all 2D sub-tabs)
        if shape == "Rectangular":
            xc = [-b/2, b/2, b/2, -b/2, -b/2]
            yc = [-h/2, -h/2, h/2, h/2, -h/2]
            xt = [-(b/2-cv2), (b/2-cv2), (b/2-cv2), -(b/2-cv2), -(b/2-cv2)]
            yt = [-(h/2-cv2), -(h/2-cv2), (h/2-cv2), (h/2-cv2), -(h/2-cv2)]
        else:
            th = np.linspace(0, 2*math.pi, 120)
            xc, yc = (b/2)*np.cos(th), (b/2)*np.sin(th)
            xt, yt = (b/2-cv2)*np.cos(th), (b/2-cv2)*np.sin(th)

        lim = max(b, h) / 2 + max(b, h) * 0.35

        def _add_outline(fig):
            fig.add_trace(go.Scatter(x=xc, y=yc, mode='lines', name='Concrete',
                                     line=dict(color=blue, width=3),
                                     fill='toself', fillcolor='rgba(56,189,248,0.1)'))
            fig.add_trace(go.Scatter(x=xt, y=yt, mode='lines', name='Ties',
                                     line=dict(color=gold, width=1.5, dash='dash')))

        def _add_dims(fig):
            if shape == "Rectangular":
                yd = -h/2 - max(b, h)*0.18
                fig.add_shape(type="line", x0=-b/2, y0=yd, x1=b/2, y1=yd,
                              line=dict(color='#94a3b8', width=1.5))
                fig.add_annotation(x=0, y=yd, text=f"b = {b} cm", showarrow=False,
                                   yshift=12, font=dict(color='white', size=12))
                xd = -b/2 - max(b, h)*0.18
                fig.add_shape(type="line", x0=xd, y0=-h/2, x1=xd, y1=h/2,
                              line=dict(color='#94a3b8', width=1.5))
                fig.add_annotation(x=xd, y=0, text=f"h = {h} cm", showarrow=False,
                                   xshift=-15, textangle=-90, font=dict(color='white', size=12))
            else:
                yd = -b/2 - b*0.2
                fig.add_shape(type="line", x0=-b/2, y0=yd, x1=b/2, y1=yd,
                              line=dict(color='#94a3b8', width=1.5))
                fig.add_annotation(x=0, y=yd, text=f"D = {b} cm", showarrow=False,
                                   yshift=12, font=dict(color='white', size=12))

        def _add_specs(fig, extra=""):
            fig.add_annotation(
                xref='paper', yref='paper', x=0.98, y=0.02,
                text=(f"<b>SPECS</b><br>f'c = {fc} ksc<br>fy = {fy} ksc<br>"
                      f"DB{db} ({n_per_bundle}/bundle)<br>"
                      f"Ast = {engine.total_as:.2f} cm²<br>ρ = {rho_pct:.2f}%"
                      + (f"<br>{extra}" if extra else "")),
                showarrow=False, align='right',
                bgcolor='rgba(15,23,42,0.85)', bordercolor='#334155',
                borderpad=10, font=dict(color=blue, size=11))

        def _base_layout(fig):
            fig.update_layout(
                plot_bgcolor=dark, paper_bgcolor=dark,
                xaxis=dict(visible=False, range=[-lim, lim]),
                yaxis=dict(visible=False, range=[-lim, lim],
                           scaleanchor='x', scaleratio=1),
                height=650, margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(font=dict(color='white'), orientation='h',
                            y=1.05, x=0.5, xanchor='center'))

        # ── Choose sub-tabs based on whether bundles are active ───────────────
        if n_per_bundle == 1:
            sub_tabs = st.tabs(["2D Section", "3D Cage"])
            section_2d_idx, cage_3d_idx = 0, 1
            section_eq_idx = section_exp_idx = None
        else:
            sub_tabs = st.tabs(["2D — Equivalent Circle", "2D — Exploded Bundle", "3D Cage"])
            section_eq_idx, section_exp_idx, cage_3d_idx = 0, 1, 2
            section_2d_idx = None

        # ── Standard single-bar 2D view ───────────────────────────────────────
        if section_2d_idx is not None:
            with sub_tabs[section_2d_idx]:
                show_dim  = st.toggle("Show Dimensions", value=True, key="show_dim_t3")
                show_lid  = st.toggle("Bar Labels",      value=True, key="show_lid_t3")
                show_spec = st.toggle("Material Specs",  value=True, key="show_spec_t3")

                fig2d = go.Figure()
                _add_outline(fig2d)
                fig2d.add_trace(go.Scatter(
                    x=bx, y=by,
                    mode='markers+text' if show_lid else 'markers',
                    marker=dict(color=dark, size=13, line=dict(color=red, width=2.5)),
                    text=[str(i+1) for i in range(len(bx))],
                    textposition='top center', textfont=dict(color='white', size=9),
                    name='Rebars'))
                if show_dim: _add_dims(fig2d)
                if show_spec: _add_specs(fig2d)
                _base_layout(fig2d)
                st.plotly_chart(fig2d, use_container_width=True)

        # ── Bundled: Equivalent-circle view ──────────────────────────────────
        if section_eq_idx is not None:
            with sub_tabs[section_eq_idx]:
                show_dim_eq  = st.toggle("Show Dimensions", value=True, key="show_dim_eq")
                show_lid_eq  = st.toggle("Bundle Labels",   value=True, key="show_lid_eq")
                show_spec_eq = st.toggle("Material Specs",  value=True, key="show_spec_eq")

                st.caption(
                    f"Each marker represents one **{n_per_bundle}-bar bundle** drawn as a "
                    f"single equivalent circle of diameter d_b,eq = "
                    f"{engine.db_eq_cm*10:.1f} mm. "
                    "This is how the section is treated analytically."
                )
                # Scale marker size to approximate equivalent diameter on screen
                # Use a reference: single DB25 → size≈13 → scale proportionally
                eq_marker_size = max(13, int(13 * engine.db_eq_cm / (db / 10.0)))

                fig_eq = go.Figure()
                _add_outline(fig_eq)
                fig_eq.add_trace(go.Scatter(
                    x=bx, y=by,
                    mode='markers+text' if show_lid_eq else 'markers',
                    marker=dict(
                        color=orange,
                        size=eq_marker_size,
                        line=dict(color='white', width=2),
                        symbol='circle'),
                    text=[f"B{i+1}" for i in range(len(bx))],
                    textposition='top center',
                    textfont=dict(color='white', size=9),
                    name=f'{n_per_bundle}-bar bundle (equiv. circle)'))
                if show_dim_eq: _add_dims(fig_eq)
                if show_spec_eq:
                    _add_specs(fig_eq, f"d_b,eq = {engine.db_eq_cm*10:.1f} mm")
                _base_layout(fig_eq)
                st.plotly_chart(fig_eq, use_container_width=True)

        # ── Bundled: Exploded view ────────────────────────────────────────────
        if section_exp_idx is not None:
            with sub_tabs[section_exp_idx]:
                show_dim_exp  = st.toggle("Show Dimensions", value=True, key="show_dim_exp")
                show_lid_exp  = st.toggle("Bar Labels",      value=True, key="show_lid_exp")
                show_spec_exp = st.toggle("Material Specs",  value=True, key="show_spec_exp")

                st.caption(
                    f"Each bundle position is **exploded** to show the "
                    f"{n_per_bundle} individual DB{db} bars arranged tightly within "
                    "the bundle. Offset ≈ 1 bar diameter between individual bar centroids."
                )
                # Build offset patterns for 1–4 bars within a bundle
                # Offsets in units of db_cm, rotated 45° for corner aesthetics
                db_c = engine.db_cm
                _offset_patterns = {
                    1: [(0, 0)],
                    2: [(-0.5, 0), (0.5, 0)],
                    3: [(-0.7, -0.4), (0.7, -0.4), (0.0, 0.7)],
                    4: [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)],
                }
                offsets = _offset_patterns[n_per_bundle]

                fig_exp = go.Figure()
                _add_outline(fig_exp)

                # Plot individual bars within each bundle position
                all_indiv_x, all_indiv_y, all_indiv_text = [], [], []
                colors_cycle = [red, violet, orange, '#4ade80']
                for pos_i, (px, py) in enumerate(zip(bx, by)):
                    for bar_j, (ox, oy) in enumerate(offsets):
                        ix = px + ox * db_c
                        iy = py + oy * db_c
                        all_indiv_x.append(ix)
                        all_indiv_y.append(iy)
                        all_indiv_text.append(
                            f"B{pos_i+1}-{bar_j+1}" if show_lid_exp else ""
                        )
                # Single trace for all individual bars (faster rendering)
                fig_exp.add_trace(go.Scatter(
                    x=all_indiv_x, y=all_indiv_y,
                    mode='markers+text' if show_lid_exp else 'markers',
                    marker=dict(color=red, size=10,
                                line=dict(color='white', width=1.5)),
                    text=all_indiv_text,
                    textposition='top center',
                    textfont=dict(color='white', size=7),
                    name=f'DB{db} individual bars'))
                # Draw bundle outlines (dashed circle around each bundle group)
                for px, py in zip(bx, by):
                    theta_circ = np.linspace(0, 2*math.pi, 40)
                    r_bundle = engine.db_eq_cm / 2.0
                    bcirc_x = px + r_bundle * np.cos(theta_circ)
                    bcirc_y = py + r_bundle * np.sin(theta_circ)
                    fig_exp.add_trace(go.Scatter(
                        x=bcirc_x, y=bcirc_y, mode='lines',
                        line=dict(color=orange, width=1, dash='dot'),
                        showlegend=False))

                if show_dim_exp: _add_dims(fig_exp)
                if show_spec_exp:
                    _add_specs(fig_exp, f"{n_per_bundle}×DB{db}/bundle")
                _base_layout(fig_exp)
                st.plotly_chart(fig_exp, use_container_width=True)

        # ── 3D cage ───────────────────────────────────────────────────────────
        with sub_tabs[cage_3d_idx]:
            L_col = min(max(Lu_x, 1.0) * 100.0, 600.0)   # [REVIEW] follows Lu input
            fig3d_cage = go.Figure()
            # For bundled bars, render individual bars with offsets in 3D too
            db_c = engine.db_cm
            _offset_patterns_3d = {
                1: [(0, 0)],
                2: [(-0.5, 0), (0.5, 0)],
                3: [(-0.7, -0.4), (0.7, -0.4), (0.0, 0.7)],
                4: [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)],
            }
            offsets_3d = _offset_patterns_3d[n_per_bundle]
            bar_color_3d = red if n_per_bundle == 1 else orange

            for pos_i, (px, py) in enumerate(zip(bx, by)):
                for bar_j, (ox, oy) in enumerate(offsets_3d):
                    ix = px + ox * db_c
                    iy = py + oy * db_c
                    lbl = f"Bundle {pos_i+1}" if bar_j == 0 else None
                    fig3d_cage.add_trace(go.Scatter3d(
                        x=[ix, ix], y=[iy, iy], z=[0, L_col], mode='lines',
                        line=dict(color=bar_color_3d, width=4 if n_per_bundle == 1 else 3),
                        name=lbl if lbl else f'Bar {pos_i+1}-{bar_j+1}',
                        showlegend=(bar_j == 0)))

            n_ties_3d = max(5, int(L_col / shear['s_design']))
            for z in np.linspace(shear['s_design'], L_col - shear['s_design'], n_ties_3d):
                fig3d_cage.add_trace(go.Scatter3d(
                    x=list(xt) if shape == "Rectangular" else list(xc),
                    y=list(yt) if shape == "Rectangular" else list(yc),
                    z=[z] * (len(xt) if shape == "Rectangular" else len(xc)),
                    mode='lines', line=dict(color=gold, width=3), showlegend=False))
            fig3d_cage.update_layout(
                scene=dict(aspectmode='data',
                           xaxis_title='X (cm)', yaxis_title='Y (cm)',
                           zaxis_title='Height (cm)'),
                margin=dict(l=0, r=0, t=0, b=0), height=580, paper_bgcolor=dark)
            st.plotly_chart(fig3d_cage, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 – Shear & Seismic
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🛡️ Shear, Torsion & Seismic Detailing (ACI 318-19, MKS)")
        s = shear

        # Dashboard metrics
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("φVnx (X-shear cap.)", f"{s['phiVnx']:.2f} ton",
                    "✅ OK" if s['x_ok'] else "❌ Fail",
                    delta_color="normal" if s['x_ok'] else "inverse")
        r1c2.metric("φVny (Y-shear cap.)", f"{s['phiVny']:.2f} ton",
                    "✅ OK" if s['y_ok'] else "❌ Fail",
                    delta_color="normal" if s['y_ok'] else "inverse")
        r1c3.metric("Tu",  f"{tu_tonm:.2f} ton-m",
                    "Critical" if s['torsion_critical'] else "Ignorable",
                    delta_color="inverse" if s['torsion_critical'] else "off")
        r1c4.metric("Tth", f"{s['Tth_tonm']:.3f} ton-m")

        st.markdown("---")

        with st.expander("📝 Full Shear Calculation (Step-by-Step)", expanded=True):
            st.markdown(f"""
#### Unit System Reminder
All calculations use **MKS**: stress in ksc, force in kgf (→ ton), length in cm.
MKS coefficient 0.53 ≡ SI coefficient 0.17 (√(10.2 ksc/MPa) ≈ 3.19; 0.53/3.19 ≈ 0.17).

---

#### 1. Effective Depths & Widths
* **dx** (depth for X-shear, bending in h-direction) = h − d' = {engine.h} − {engine.d_prime:.2f} = **{s['dx']:.2f} cm**
* **dy** (depth for Y-shear, bending in b-direction) = b − d' = {engine.b} − {engine.d_prime:.2f} = **{s['dy']:.2f} cm**
* **bwx** (web width for X-shear) = {s['bwx']} cm
* **bwy** (web width for Y-shear) = {s['bwy']} cm

#### 2. Concrete Shear Capacity (ACI 318-19 Table 22.5.5.1)
Exact formula applied via MPa conversion (ACI SI form, then converted to MKS):
$$V_c = \\left[0.543\\sqrt{{f'_c(ksc)}} + \\frac{{N_u\\,(kgf)}}{{6\\,A_g(cm^2)}}\\right] b_w\\,d \\quad [kgf]$$
*(0.543 = 0.17 × √10.197; N_u/(6·A_g) is the axial term in ksc — additive, not multiplicative)*
* f'c = {fc} ksc → {fc/10.197:.3f} MPa; Ag = {engine.Ag:.0f} cm²
* Nu = {Pu*1000:.0f} kgf; Nu/(6·Ag) = {Pu*1000/(6*engine.Ag):.4f} ksc
* **Vcx** = [0.543×√{fc} + {Pu*1000/(6*engine.Ag):.4f}] × {s['bwx']} × {s['dx']:.2f} / 1000 = **{s['Vcx_ton']:.3f} ton**
* **Vcy** = [0.543×√{fc} + {Pu*1000/(6*engine.Ag):.4f}] × {s['bwy']} × {s['dy']:.2f} / 1000 = **{s['Vcy_ton']:.3f} ton**

#### 3. Transverse Steel Provided
* Tie: {f"RB{tie_dia}" if tie_dia<10 else f"DB{tie_dia}"}, legs = {tie_legs}
* At = π×{s['d_tie']:.2f}²/4 = {s['At']:.3f} cm²
* Av = {tie_legs} × {s['At']:.3f} = **{s['Av']:.3f} cm²**

#### 4. Required Stirrup Spacing from Shear Demand
$$V_s = \\frac{{A_v f_y d}}{{s}} \\implies s = \\frac{{A_v f_y d}}{{V_s,req}}$$
* **X-demand Vs,req** = Vux/φ − Vcx = {vux_ton:.2f}/{PHI_SHEAR} − {s['Vcx_ton']:.3f} = {max(0,vux_ton/PHI_SHEAR - s['Vcx_ton']):.3f} ton
  → sx,shear = {s['Av']:.3f}×{fy}×{s['dx']:.2f} / ({max(1e-9,max(0,vux_ton/PHI_SHEAR-s['Vcx_ton']))*1000:.0f}) = **{s['sx_shear']:.1f} cm**
* **Y-demand Vs,req** = Vuy/φ − Vcy = {vuy_ton:.2f}/{PHI_SHEAR} − {s['Vcy_ton']:.3f} = {max(0,vuy_ton/PHI_SHEAR - s['Vcy_ton']):.3f} ton
  → sy,shear = **{s['sy_shear']:.1f} cm**

#### 5. Spacing Limits
| Limit | Value |
|---|---|
| Code max (d/2, 60 cm) X | {s['s_max_x']:.1f} cm |
| Code max (d/2, 60 cm) Y | {s['s_max_y']:.1f} cm |
| Seismic SMF limit | {s['s_seismic']:.1f} cm |
| **Governing design spacing** | **{s['s_design']:.1f} cm** |

#### 6. Provided Shear Capacity at s = {s['s_design']:.1f} cm
$$\\phi V_n = \\phi(V_c + V_s) = {PHI_SHEAR}\\left(V_c + \\frac{{A_v f_y d}}{{s}}\\right)$$
* **φVnx** = {PHI_SHEAR}×({s['Vcx_ton']:.3f} + {s['Vsx_prov_ton']:.3f}) = **{s['phiVnx']:.3f} ton** {'✅ ≥' if s['x_ok'] else '❌ <'} Vux = {vux_ton:.2f} ton
* **φVny** = {PHI_SHEAR}×({s['Vcy_ton']:.3f} + {s['Vsy_prov_ton']:.3f}) = **{s['phiVny']:.3f} ton** {'✅ ≥' if s['y_ok'] else '❌ <'} Vuy = {vuy_ton:.2f} ton

#### 7. Torsion Threshold Check (ACI 318-19 §22.7.4.1)
$$T_{{th}} = \\phi\\,0.265\\sqrt{{f'_c}}\\frac{{A_{{cp}}^2}}{{p_{{cp}}}}$$
* Acp = {s['Acp']:.2f} cm²,  pcp = {s['pcp']:.2f} cm
* Tth = {PHI_SHEAR}×0.265×√{fc}×{s['Acp']:.2f}²/{s['pcp']:.2f} / 100000 = **{s['Tth_tonm']:.4f} ton-m**
* Tu = {tu_tonm:.2f} ton-m → {"**Critical — provide torsional reinforcement!**" if s['torsion_critical'] else "Negligible."}

#### 8. Lap Splice Lengths (ACI 318-19 §25.5 / §25.6.1.5)
{"" if n_per_bundle == 1 else f"**Bundle adjustment:** d_b,eq = d_b·√n = {engine.db_cm*10:.0f}·√{n_per_bundle} = **{engine.db_eq_cm*10:.1f} mm**"}

| Parameter | Single bar (DB{db}) | {"Bundle (" + str(n_per_bundle) + "×DB" + str(db) + ")" if n_per_bundle > 1 else "—"} | Governing |
|---|---|---|---|
| Tension ld | {splice_data['ld_single']:.0f} cm (base) | {"N/A" if n_per_bundle == 1 else f"{splice_data['ld_bundle_gov']:.0f} cm (ACI +% rule)"} | {splice_data['ld_bundle_gov']:.0f} cm |
| Class B Splice (1.3·ld) | {splice_data['l_splice_B_single']:.0f} cm | {"—" if n_per_bundle == 1 else f"{splice_data['l_splice_B_bundle']:.0f} cm"} | **{l_splice_B:.0f} cm** |
| Compression Splice | {splice_data['l_compression_single']:.0f} cm | {"—" if n_per_bundle == 1 else f"{splice_data['l_compression_bundle']:.0f} cm"} | **{l_compression:.0f} cm** |

{"⚠️ **Bundled bars:** lap ต่อเส้น (ld ×1.20/1.33), ไม่ซ้อนทับกันในกลุ่มเดียว" if n_per_bundle >= 3 else ""}
* **Selected for design:** {"Class B Tension (SMF requirement)" if is_seismic else "Compression splice (Ordinary frame)"}
  → **{l_splice_B if is_seismic else l_compression:.0f} cm**
""")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 – Full calculation report
    # ─────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### 📝 Detailed Calculation Report  *(ACI 318-19 / MKS)*")
        st.caption("All values substituted numerically. Units: ton, ton-m, cm, ksc.")

        # ── 1. Section & Material Properties ─────────────────────────────────
        with st.expander("1. Section & Material Properties", expanded=True):
            st.markdown("#### 1a. Gross Section Geometry")
            if shape == "Rectangular":
                st.latex(rf"A_g = b \times h = {b} \times {h} = {engine.Ag:.2f} \text{{ cm}}^2")
                st.latex(rf"I_{{gx}} = \frac{{b \cdot h^3}}{{12}} = \frac{{{b} \times {h}^3}}{{12}} = {engine.Igx:,.2f} \text{{ cm}}^4")
                st.latex(rf"I_{{gy}} = \frac{{h \cdot b^3}}{{12}} = \frac{{{h} \times {b}^3}}{{12}} = {engine.Igy:,.2f} \text{{ cm}}^4")
                st.latex(rf"r_x = 0.30 \cdot h = 0.30 \times {h} = {engine.rx:.2f} \text{{ cm}}")
                st.latex(rf"r_y = 0.30 \cdot b = 0.30 \times {b} = {engine.ry:.2f} \text{{ cm}}")
            else:
                st.latex(rf"A_g = \frac{{\pi D^2}}{{4}} = \frac{{\pi \times {h}^2}}{{4}} = {engine.Ag:.2f} \text{{ cm}}^2")
                st.latex(rf"I_g = \frac{{\pi D^4}}{{64}} = \frac{{\pi \times {h}^4}}{{64}} = {engine.Igx:,.2f} \text{{ cm}}^4")
                st.latex(rf"r = 0.25 \cdot D = 0.25 \times {h} = {engine.rx:.2f} \text{{ cm}}")

            st.markdown("#### 1b. Material Properties")
            st.latex(rf"f'_c = {fc} \text{{ ksc}}, \quad f_y = {fy} \text{{ ksc}}, \quad E_s = {engine.Es:,.0f} \text{{ ksc}}")
            st.latex(rf"E_c = 15100\sqrt{{f'_c}} = 15100 \times \sqrt{{{fc}}} = 15100 \times {math.sqrt(fc):.4f} = {engine.Ec:,.0f} \text{{ ksc}}")
            st.latex(rf"\beta_1 = 0.85 - 0.05\,\frac{{f'_c - 280}}{{70}} = 0.85 - 0.05 \times \frac{{{fc}-280}}{{70}} = {engine.beta1:.4f}")
            st.latex(rf"\varepsilon_{{ty,\lim}} = \frac{{f_y}}{{E_s}} + 0.003 = \frac{{{fy}}}{{{engine.Es:,.0f}}} + 0.003 = {engine.eps_ty_lim:.5f}")

            st.markdown("#### 1c. Reinforcement")
            st.latex(rf"d_b = {engine.db_cm*10:.1f} \text{{ mm}} \quad A_{{bar}} = \frac{{\pi d_b^2}}{{4}} = \frac{{\pi \times {engine.db_cm:.3f}^2}}{{4}} = {engine.as_single:.4f} \text{{ cm}}^2")
            if n_per_bundle > 1:
                st.latex(rf"d_{{b,eq}} = d_b \cdot \sqrt{{n}} = {engine.db_cm*10:.1f} \times \sqrt{{{n_per_bundle}}} = {engine.db_eq_cm*10:.3f} \text{{ mm}} \quad (\text{{ACI \S25.6.1.2}})")
                st.latex(rf"A_{{bundle}} = n \times A_{{bar}} = {n_per_bundle} \times {engine.as_single:.4f} = {engine.as_bundle:.4f} \text{{ cm}}^2\text{{/position}}")
            st.latex(rf"d' = c_{{cover}} + d_{{tie}} + \frac{{d_{{b,eq}}}}{{2}} = {cover} + {engine.tie_cm:.2f} + \frac{{{engine.db_eq_cm:.4f}}}{{2}} = {engine.d_prime:.4f} \text{{ cm}}")
            st.latex(rf"n_{{positions}} = {engine.n_positions}, \quad n_{{bars,total}} = {engine.n_bars}")
            st.latex(rf"A_{{st}} = n_{{pos}} \times A_{{bundle}} = {engine.n_positions} \times {engine.as_bundle:.4f} = {engine.total_as:.4f} \text{{ cm}}^2")
            st.latex(rf"\rho = \frac{{A_{{st}}}}{{A_g}} = \frac{{{engine.total_as:.4f}}}{{{engine.Ag:.2f}}} = {engine.rho:.6f} = {rho_pct:.3f}\%")
            if rho_ok:
                st.success(f"✅ ρ = {rho_pct:.3f}% is within ACI §10.6.1.1 / 18.7.4.1 limits: 1.0% ≤ ρ ≤ {rho_max:.1f}%")
            else:
                st.error(f"❌ ρ = {rho_pct:.3f}% violates ACI §10.6.1.1 / 18.7.4.1 limits: 1.0% ≤ ρ ≤ {rho_max:.1f}%")

        # ── 2. Maximum Axial Capacity ─────────────────────────────────────────
        with st.expander("2. Maximum Axial Capacity (ACI 318-19 §22.4.2)", expanded=True):
            is_circ2 = (shape == "Circular")
            phi_c2   = 0.75 if is_circ2 else 0.65
            fac_c2   = 0.85 if is_circ2 else 0.80
            Po_val   = (0.85 * fc * (engine.Ag - engine.total_as) + fy * engine.total_as) / 1000
            phi_Pn_max_val = phi_c2 * fac_c2 * Po_val
            st.markdown(f"Section type: **{'Circular (spiral)' if is_circ2 else 'Rectangular (tied)'}** → φ = {phi_c2}, reduction factor = {fac_c2}")
            st.latex(
                rf"P_o = \frac{{0.85 f'_c (A_g - A_{{st}}) + f_y A_{{st}}}}{{1000}}"
                rf"= \frac{{0.85 \times {fc} \times ({engine.Ag:.2f} - {engine.total_as:.4f}) + {fy} \times {engine.total_as:.4f}}}{{1000}}"
                rf"= \frac{{{0.85*fc*(engine.Ag-engine.total_as):.2f} + {fy*engine.total_as:.2f}}}{{1000}}"
                rf"= {Po_val:.3f} \text{{ ton}}")
            st.latex(
                rf"\phi P_{{n,\max}} = \phi \times {fac_c2} \times P_o"
                rf"= {phi_c2} \times {fac_c2} \times {Po_val:.3f}"
                rf"= {phi_Pn_max_val:.3f} \text{{ ton}}")
            if Pu <= phi_Pn_max_val:
                st.success(f"✅ Pu = {Pu:.2f} t ≤ φPn,max = {phi_Pn_max_val:.3f} t")
            else:
                st.error(f"❌ Pu = {Pu:.2f} t > φPn,max = {phi_Pn_max_val:.3f} t — section too small!")

        # ── 3. Minimum Eccentricity ───────────────────────────────────────────
        with st.expander("3. Minimum Eccentricity Moments (ACI 318-19 §6.6.4.5.4)", expanded=True):
            st.markdown("ACI requires a minimum design moment regardless of analysis results:")
            st.latex(rf"e_{{min,x}} = 0.015 + 0.03\,\frac{{h}}{{100}} = 0.015 + 0.03 \times \frac{{{h}}}{{100}} = {0.015+0.03*h/100:.5f} \text{{ m}}")
            st.latex(rf"M_{{u,min,x}} = P_u \times e_{{min,x}} = {Pu:.2f} \times {0.015+0.03*h/100:.5f} = {e_min_x:.4f} \text{{ t-m}}")
            st.latex(rf"M_{{ux,design}} = \max(M_{{ux}},\, M_{{u,min,x}}) = \max({Mux:.3f},\, {e_min_x:.4f}) = {Mu_x_dsgn:.4f} \text{{ t-m}}")
            st.latex(rf"e_{{min,y}} = 0.015 + 0.03\,\frac{{b}}{{100}} = 0.015 + 0.03 \times \frac{{{b}}}{{100}} = {0.015+0.03*b/100:.5f} \text{{ m}}")
            st.latex(rf"M_{{u,min,y}} = P_u \times e_{{min,y}} = {Pu:.2f} \times {0.015+0.03*b/100:.5f} = {e_min_y:.4f} \text{{ t-m}}")
            st.latex(rf"M_{{uy,design}} = \max(M_{{uy}},\, M_{{u,min,y}}) = \max({Muy:.3f},\, {e_min_y:.4f}) = {Mu_y_dsgn:.4f} \text{{ t-m}}")

        # ── 4. Slenderness & Moment Magnification ─────────────────────────────
        with st.expander("4. Slenderness & Moment Magnification (ACI 318-19 §6.6.4)", expanded=True):
            st.markdown(f"**Frame type:** {frame_type}")

            for axis_lbl, K_v, Lu_v, r_v, Ig_v, Ise_v, EI_v, Pc_v, kl_r_v, Cm_v, beta_d_v, del_v, Mc_v, Mu_dsgn_v in [
                ('X', K_x, Lu_x, engine.rx, engine.Igx, engine.Ise_x, EIx, Pcx, kl_rx, Cm_x, beta_d, del_x, Mcx, Mu_x_dsgn),
                ('Y', K_y, Lu_y, engine.ry, engine.Igy, engine.Ise_y, EIy, Pcy, kl_ry, Cm_y, beta_d, del_y, Mcy, Mu_y_dsgn),
            ]:
                st.markdown(f"---\n##### Axis {axis_lbl}")
                if frame_type == "Non-Sway (Braced)":
                    st.latex(
                        rf"\frac{{kl}}{{r_{axis_lbl}}} = \frac{{K \cdot L_u \cdot 100}}{{r}} = "
                        rf"\frac{{{K_v} \times {Lu_v} \times 100}}{{{r_v:.3f}}} = {kl_r_v:.3f}")
                    if kl_r_v <= 22:
                        st.success(f"kl/r = {kl_r_v:.3f} ≤ 22 → Slenderness effects negligible (ACI §6.2.5). δ{axis_lbl} = 1.0, M_c{axis_lbl} = {Mu_dsgn_v:.4f} t-m")
                    else:
                        st.warning(f"kl/r = {kl_r_v:.3f} > 22 → Slenderness magnification required (ACI §6.6.4.5)")
                        st.latex(
                            rf"EI_{axis_lbl} = \frac{{0.2 E_c I_{{g{axis_lbl}}} + E_s I_{{se,{axis_lbl}}}}}{{1 + \beta_d}}"
                            rf"= \frac{{0.2 \times {engine.Ec:,.0f} \times {Ig_v:,.2f} + {engine.Es:,.0f} \times {Ise_v:,.2f}}}{{1 + {beta_d_v}}}"
                            rf"= \frac{{{0.2*engine.Ec*Ig_v:,.2f} + {engine.Es*Ise_v:,.2f}}}{{{1+beta_d_v}}}"
                            rf"= {EI_v:,.0f} \text{{ ksc·cm}}^2")
                        st.latex(
                            rf"P_{{c{axis_lbl}}} = \frac{{\pi^2 EI_{axis_lbl}}}{{(K L_u)^2}}"
                            rf"= \frac{{\pi^2 \times {EI_v:,.0f}}}{{({K_v} \times {Lu_v} \times 100)^2}}"
                            rf"= \frac{{{math.pi**2*EI_v:,.2f}}}{{{(K_v*Lu_v*100)**2:,.0f}}}"
                            rf"= {Pc_v:.3f} \text{{ ton}}")
                        denom_val = 1.0 - Pu / (0.75 * Pc_v)
                        st.latex(
                            rf"\delta_{axis_lbl} = \frac{{C_m}}{{1 - P_u / (0.75\,P_{{c{axis_lbl}}})}} "
                            rf"= \frac{{{Cm_v}}}{{1 - {Pu:.2f} / (0.75 \times {Pc_v:.3f})}}"
                            rf"= \frac{{{Cm_v}}}{{{denom_val:.6f}}}"
                            rf"= {del_v:.4f} \geq 1.0")
                        st.latex(
                            rf"M_{{c{axis_lbl}}} = \delta_{axis_lbl} \times M_{{u{axis_lbl},design}}"
                            rf"= {del_v:.4f} \times {Mu_dsgn_v:.4f} = {Mc_v:.4f} \text{{ t-m}}")
                else:
                    ds_v = delta_sx if axis_lbl == 'X' else delta_sy
                    st.latex(
                        rf"M_{{c{axis_lbl}}} = \delta_{{s{axis_lbl}}} \times M_{{u{axis_lbl},design}}"
                        rf"= {ds_v:.4f} \times {Mu_dsgn_v:.4f} = {Mc_v:.4f} \text{{ t-m}}")
                    if kl_r_v > 22:
                        st.latex(
                            rf"\text{{Additional non-sway component: }} \delta_{{ns,{axis_lbl}}} = {del_v:.4f}"
                            rf"\implies M_{{c{axis_lbl}}} = {del_v:.4f} \times {Mc_v/del_v:.4f} = {Mc_v:.4f} \text{{ t-m}}")

        # ── 5. Uniaxial Moment Capacities ─────────────────────────────────────
        with st.expander("5. Uniaxial Moment Capacities at Pu (from P-M diagram)", expanded=True):
            st.markdown(
                f"Read φMnox and φMnoy from the P-M interaction curve at **Pu = {Pu:.2f} ton** "
                f"by linear interpolation of the computed (φPn, φMn) data points."
            )
            if phi_Mnox > 0 and phi_Mnoy > 0:
                st.latex(rf"\phi M_{{nox}} = {phi_Mnox:.4f} \text{{ t-m}} \quad \text{{(from X-axis P-M at Pu={Pu:.2f} t)}}")
                st.latex(rf"\phi M_{{noy}} = {phi_Mnoy:.4f} \text{{ t-m}} \quad \text{{(from Y-axis P-M at Pu={Pu:.2f} t)}}")
            else:
                st.error("Cannot evaluate — Pu is outside the P-M curve range.")

        # ── 6. Biaxial Bending Check ───────────────────────────────────────────
        with st.expander("6. Biaxial Bending Check — PCA Load-Contour (ACI 318-19 Commentary)", expanded=True):
            st.markdown("**PCA Load-Contour Method** (Bresler, 1960 / ACI 318 Commentary R6.6.5):")
            st.latex(
                rf"\left(\frac{{M_{{cx}}}}{{\phi M_{{nox}}}}\right)^{{\alpha}} + "
                rf"\left(\frac{{M_{{cy}}}}{{\phi M_{{noy}}}}\right)^{{\alpha}} \leq 1.0")

            # Alpha derivation
            Po_a = (0.85*fc*(engine.Ag-engine.total_as)+fy*engine.total_as)/1000
            phi_Po_a = (0.75 if shape=="Circular" else 0.65)*Po_a
            ratio_a  = max(0.0, min(1.0, Pu/phi_Po_a)) if phi_Po_a > 0 else 0
            st.markdown(f"**α exponent** (PCA interpolation, ACI Commentary):")
            st.latex(
                rf"\frac{{P_u}}{{\phi P_o}} = \frac{{{Pu:.3f}}}{{{phi_Po_a:.3f}}} = {ratio_a:.4f}")
            st.latex(
                rf"\alpha = 1.15 + (ratio - 0.10) \times \frac{{1.55 - 1.15}}{{0.90}} = "
                rf"1.15 + ({ratio_a:.4f} - 0.10) \times 0.4\overline{{4}} = {alpha:.4f}")

            if phi_Mnox > 0 and phi_Mnoy > 0:
                term_x  = (Mcx / phi_Mnox)**alpha
                term_y  = (Mcy / phi_Mnoy)**alpha
                st.markdown("**Demand ratio calculation:**")
                st.latex(
                    rf"\left(\frac{{M_{{cx}}}}{{\phi M_{{nox}}}}\right)^{{\alpha}} = "
                    rf"\left(\frac{{{Mcx:.4f}}}{{{phi_Mnox:.4f}}}\right)^{{{alpha:.4f}}} = "
                    rf"{Mcx/phi_Mnox:.6f}^{{{alpha:.4f}}} = {term_x:.6f}")
                st.latex(
                    rf"\left(\frac{{M_{{cy}}}}{{\phi M_{{noy}}}}\right)^{{\alpha}} = "
                    rf"\left(\frac{{{Mcy:.4f}}}{{{phi_Mnoy:.4f}}}\right)^{{{alpha:.4f}}} = "
                    rf"{Mcy/phi_Mnoy:.6f}^{{{alpha:.4f}}} = {term_y:.6f}")
                st.latex(
                    rf"\text{{Demand Ratio}} = {term_x:.6f} + {term_y:.6f} = {demand_ratio:.6f}")
                if is_safe:
                    st.success(f"✅ {demand_ratio:.4f} ≤ 1.0 → **SAFE**")
                else:
                    st.error(f"❌ {demand_ratio:.4f} > 1.0 → **UNSAFE** — increase section or steel")
            else:
                st.error("Cannot evaluate — Pu out of range.")

        # ── 7. Shear Capacity ─────────────────────────────────────────────────
        with st.expander("7. Shear Capacity (ACI 318-19 Table 22.5.5.1)", expanded=True):
            s = shear
            st.markdown("**Effective depths (distance from compression face to tension steel centroid):**")
            st.latex(rf"d_x = h - d' = {engine.h:.2f} - {engine.d_prime:.4f} = {s['dx']:.4f} \text{{ cm}}")
            st.latex(rf"d_y = b - d' = {engine.b:.2f} - {engine.d_prime:.4f} = {s['dy']:.4f} \text{{ cm}}")

            fc_mpa_val  = fc / 10.197
            Nu_kgf_val  = Pu * 1000
            Nu_mpa_val  = min(Nu_kgf_val / engine.Ag / 10.197, 0.05 * fc_mpa_val)
            vc_mpa_val  = 0.17 * math.sqrt(fc_mpa_val) + Nu_mpa_val / 6.0
            vc_ksc_val  = vc_mpa_val * 10.197

            st.markdown("**Concrete shear stress (ACI Table 22.5.5.1 — converted to MKS):**")
            st.latex(
                rf"f'_c = {fc} \text{{ ksc}} \div 10.197 = {fc_mpa_val:.4f} \text{{ MPa}}")
            st.latex(
                rf"\frac{{N_u}}{{A_g}} = \frac{{{Nu_kgf_val:.0f}}}{{{engine.Ag:.2f}}} = {Nu_kgf_val/engine.Ag:.4f} \text{{ ksc}} "
                rf"\div 10.197 = {Nu_kgf_val/engine.Ag/10.197:.4f} \text{{ MPa}} "
                rf"\leq 0.05 f'_c = {0.05*fc_mpa_val:.4f} \text{{ MPa}}")
            st.latex(
                rf"v_c = 0.17\sqrt{{f'_c}} + \frac{{N_u}}{{6 A_g}} = "
                rf"0.17 \times \sqrt{{{fc_mpa_val:.4f}}} + \frac{{{Nu_mpa_val:.6f}}}{{6}} = "
                rf"0.17 \times {math.sqrt(fc_mpa_val):.4f} + {Nu_mpa_val/6:.6f} = "
                rf"{vc_mpa_val:.6f} \text{{ MPa}} = {vc_ksc_val:.6f} \text{{ ksc}}")

            st.markdown("**Concrete shear force:**")
            st.latex(
                rf"V_{{cx}} = v_c \times b_{{wx}} \times d_x = "
                rf"{vc_ksc_val:.6f} \times {s['bwx']} \times {s['dx']:.4f} = "
                rf"{vc_ksc_val*s['bwx']*s['dx']:.2f} \text{{ kgf}} = {s['Vcx_ton']:.4f} \text{{ ton}}")
            st.latex(
                rf"V_{{cy}} = v_c \times b_{{wy}} \times d_y = "
                rf"{vc_ksc_val:.6f} \times {s['bwy']} \times {s['dy']:.4f} = "
                rf"{vc_ksc_val*s['bwy']*s['dy']:.2f} \text{{ kgf}} = {s['Vcy_ton']:.4f} \text{{ ton}}")

            d_tie_cm = tie_dia / 10
            At_val   = math.pi * d_tie_cm**2 / 4
            Av_val   = tie_legs * At_val
            st.markdown("**Transverse steel:**")
            st.latex(
                rf"A_t = \frac{{\pi d_{{tie}}^2}}{{4}} = \frac{{\pi \times {d_tie_cm:.3f}^2}}{{4}} = {At_val:.4f} \text{{ cm}}^2")
            st.latex(
                rf"A_v = n_{{legs}} \times A_t = {tie_legs} \times {At_val:.4f} = {Av_val:.4f} \text{{ cm}}^2")

            st.markdown("**Required spacing from shear demand:**")
            Vsx_req = max(0.0, vux_ton / PHI_SHEAR - s['Vcx_ton'])
            Vsy_req = max(0.0, vuy_ton / PHI_SHEAR - s['Vcy_ton'])
            st.latex(
                rf"V_{{s,req,x}} = \frac{{V_{{ux}}}}{{\phi}} - V_{{cx}} = "
                rf"\frac{{{vux_ton:.3f}}}{{0.75}} - {s['Vcx_ton']:.4f} = "
                rf"{vux_ton/PHI_SHEAR:.4f} - {s['Vcx_ton']:.4f} = {Vsx_req:.4f} \text{{ ton}}")
            if Vsx_req > 0:
                st.latex(
                    rf"s_{{req,x}} = \frac{{A_v f_y d_x}}{{V_{{s,req,x}} \times 1000}} = "
                    rf"\frac{{{Av_val:.4f} \times {fy} \times {s['dx']:.4f}}}{{{Vsx_req:.4f} \times 1000}} = "
                    rf"\frac{{{Av_val*fy*s['dx']:.2f}}}{{{Vsx_req*1000:.2f}}} = {s['sx_shear']:.2f} \text{{ cm}}")
            else:
                st.latex(rf"V_{{s,req,x}} = 0 \implies s_{{req,x}} = \text{{no stirrups required beyond minimum}}")

            st.markdown("**Governing design spacing:**")
            st.latex(
                rf"s_{{design}} = \min\!\left(s_{{req}},\, \frac{{d}}{{2}},\, 60\text{{cm}},\, s_{{seismic}}\right) = "
                rf"\min({s['sx_shear']:.1f},\, {s['s_max_x']:.1f},\, {s['s_seismic']:.1f}\text{{ cm}}) = "
                rf"{s['s_design']:.1f} \text{{ cm (rounded to 2.5 cm)}}")

            st.markdown("**Provided capacity check:**")
            Vsx_prov = Av_val * fy * s['dx'] / (s['s_design'] * 1000)
            Vsy_prov = Av_val * fy * s['dy'] / (s['s_design'] * 1000)
            st.latex(
                rf"V_{{s,x}} = \frac{{A_v f_y d_x}}{{s \times 1000}} = "
                rf"\frac{{{Av_val:.4f} \times {fy} \times {s['dx']:.4f}}}{{{s['s_design']:.1f} \times 1000}} = {Vsx_prov:.4f} \text{{ ton}}")
            phi_Vnx = PHI_SHEAR * (s['Vcx_ton'] + Vsx_prov)
            phi_Vny = PHI_SHEAR * (s['Vcy_ton'] + Vsy_prov)
            st.latex(
                rf"\phi V_{{nx}} = \phi (V_{{cx}} + V_{{s,x}}) = 0.75 \times ({s['Vcx_ton']:.4f} + {Vsx_prov:.4f}) = "
                rf"0.75 \times {s['Vcx_ton']+Vsx_prov:.4f} = {phi_Vnx:.4f} \text{{ ton}}")
            if s['x_ok']:
                st.success(f"✅ φVnx = {phi_Vnx:.4f} t ≥ Vux = {vux_ton:.3f} t")
            else:
                st.error(f"❌ φVnx = {phi_Vnx:.4f} t < Vux = {vux_ton:.3f} t — increase ties or section")
            st.latex(
                rf"\phi V_{{ny}} = 0.75 \times ({s['Vcy_ton']:.4f} + {Vsy_prov:.4f}) = {phi_Vny:.4f} \text{{ ton}}")
            if s['y_ok']:
                st.success(f"✅ φVny = {phi_Vny:.4f} t ≥ Vuy = {vuy_ton:.3f} t")
            else:
                st.error(f"❌ φVny = {phi_Vny:.4f} t < Vuy = {vuy_ton:.3f} t")

            st.markdown("**Maximum Vs check (ACI §22.5.1.2):**")
            st.latex(
                rf"V_{{s,\max,x}} = \frac{{0.66\sqrt{{f'_c[\text{{MPa}}]}}\,b_{{wx}}\,d_x \times 100}}{{9810}} = "
                rf"\frac{{0.66 \times {math.sqrt(fc_mpa_val):.4f} \times {s['bwx']} \times 10 \times {s['dx']:.4f} \times 10}}{{9810}} = "
                rf"{s['Vs_max_x_ton']:.4f} \text{{ ton}}")
            if s['section_adequate_x']:
                st.success(f"✅ Section adequate for X-shear (Vs,req ≤ Vs,max)")
            else:
                st.error("❌ Section too small for X-shear — increase dimensions")

        # ── 8. Torsion Threshold ──────────────────────────────────────────────
        with st.expander("8. Torsion Threshold (ACI 318-19 §22.7.4.1)", expanded=False):
            s = shear
            st.latex(
                rf"T_{{th}} = \phi \times 0.265\sqrt{{f'_c}} \times \frac{{A_{{cp}}^2}}{{p_{{cp}}}}"
                rf"= 0.75 \times 0.265 \times \sqrt{{{fc}}} \times \frac{{{s['Acp']:.2f}^2}}{{{s['pcp']:.2f}}}")
            Tth_kgcm = 0.75 * 0.265 * math.sqrt(fc) * s['Acp']**2 / s['pcp']
            st.latex(
                rf"= 0.75 \times 0.265 \times {math.sqrt(fc):.4f} \times \frac{{{s['Acp']**2:.2f}}}{{{s['pcp']:.2f}}}"
                rf"= {Tth_kgcm:.2f} \text{{ kgf·cm}} = {Tth_kgcm/100000:.6f} \text{{ ton-m}}")
            st.latex(rf"T_{{th}} = {s['Tth_tonm']:.6f} \text{{ ton-m}}")
            if s['torsion_critical']:
                st.error(f"❌ Tu = {tu_tonm:.4f} t-m > Tth = {s['Tth_tonm']:.6f} t-m → Torsion design required!")
            else:
                st.success(f"✅ Tu = {tu_tonm:.4f} t-m ≤ Tth = {s['Tth_tonm']:.6f} t-m → Torsion negligible")

        # ── 9. Lap Splice ─────────────────────────────────────────────────────
        with st.expander("9. Lap Splice Lengths (ACI 318-19 §25.5 / §25.6.1.5)", expanded=False):
            fy_mpa_s  = fy / 10.197
            fc_mpa_s  = fc / 10.197
            db_mm_s   = engine.db_cm * 10
            ld_mm_s   = fy_mpa_s / (1.1 * math.sqrt(fc_mpa_s) * 2.5) * db_mm_s
            ld_cm_s   = splice_data['ld_single']
            st.markdown("**Single-bar development length (ACI 318-19 §25.4.2.4, SI form):**")
            st.markdown("Assumptions: λ=1.0, ψt=ψe=ψs=1.0, (cb+Ktr)/db = 2.5 (ค่าสูงสุดที่ยอมให้ใช้ — ตรวจระยะหุ้ม/ระยะเรียง/ปลอก)")
            st.latex(r"l_d = \frac{f_y\,\psi_t\psi_e\psi_s}{1.1\,\lambda\sqrt{f'_c}\,[(c_b+K_{tr})/d_b]}\,d_b \ge 300\ \text{mm}")
            st.latex(rf"l_d = \frac{{{fy_mpa_s:.1f}}}{{1.1 \times \sqrt{{{fc_mpa_s:.2f}}} \times 2.5}} \times {db_mm_s:.1f} = {ld_mm_s:.1f}\ \text{{mm}} \Rightarrow {ld_cm_s:.1f}\ \text{{cm}}")
            st.latex(rf"l_{{splice,B}} = 1.3 \times l_d = {splice_data['l_splice_B_single']:.1f}\ \text{{cm}}")
            st.latex(rf"l_{{compr}} = {splice_data['l_compression_single']:.1f}\ \text{{cm}}\quad (\text{{ACI 25.5.5.1}})")
            if n_per_bundle > 1:
                st.markdown(f"---\n**Bundle adjustment — {n_per_bundle}-bar bundle (ต่อเส้นเดี่ยว × { {2:1.0,3:1.2,4:1.33}[n_per_bundle]}):**")
                st.latex(rf"l_{{d,bundle}} = {splice_data['ld_bundle_gov']:.1f}\ \text{{cm}},\quad l_{{splice,B,bundle}} = {splice_data['l_splice_B_bundle']:.1f}\ \text{{cm}}")

            st.markdown("---\n**Summary:**")
            col_s1, col_s2 = st.columns(2)
            col_s1.metric("Class B Tension Splice", f"{l_splice_B:.1f} cm")
            col_s2.metric("Compression Splice", f"{l_compression:.1f} cm")

        # ── 10. Detailing Summary Table ────────────────────────────────────────
        with st.expander("10. Detailing Summary (ACI 318-19 §10.6 / §25.7)", expanded=False):
            bundle_splice_note = f"{splice_data['l_splice_B_bundle']:.1f} cm"
            st.markdown(f"""
| Item | Computed | Limit | Code Ref. | Status |
|---|---|---|---|---|
| Steel ratio ρ | {rho_pct:.3f}% | 1.0% – {rho_max:.1f}% | ACI §10.6.1.1 | {"✅" if rho_ok else "❌"} |
| Bars per bundle | {n_per_bundle} | ≤ 4 | ACI §26.6.3.1 | ✅ |
| Bundle positions | {engine.n_positions} | — | — | — |
| Individual bars | {engine.n_bars} | — | — | — |
| Clear spacing (bundles) | {actual_space:.3f} cm | ≥ {min_req_space:.3f} cm | ACI §25.2.3/§25.6.1.2 | {"✅" if space_ok else "❌"} |
| Effective depth d' | {engine.d_prime:.3f} cm | — | — | — |
| Design tie spacing | {shear['s_design']:.1f} cm | ≤ {shear['s_max_x']:.1f} cm (d/2) | ACI §25.7.2.1 | {"✅" if shear['s_design'] <= shear['s_max_x'] else "❌"} |
| Seismic tie (6·d_b) | {6*engine.db_cm:.2f} cm | ≤ 15 cm | ACI §18.7.5.3 | {"✅" if 6*engine.db_cm <= 15 else "❌"} |
| φVnx | {shear['phiVnx']:.3f} ton | ≥ {vux_ton:.3f} ton | ACI §22.5 | {"✅" if shear['x_ok'] else "❌"} |
| φVny | {shear['phiVny']:.3f} ton | ≥ {vuy_ton:.3f} ton | ACI §22.5 | {"✅" if shear['y_ok'] else "❌"} |
| Single-bar tension splice | {splice_data['l_splice_B_single']:.1f} cm | ≥ 30 cm | ACI §25.5.2.1 | ✅ |
| Bundle tension splice | {bundle_splice_note} | — | ACI §25.6.1.5 | {"⚠️" if n_per_bundle >= 3 else "—"} |
| Single-bar compression splice | {splice_data['l_compression_single']:.1f} cm | ≥ 30 cm | ACI §25.5.5.1 | ✅ |
| Torsion critical? | {"Yes" if shear['torsion_critical'] else "No"} | Tu ≤ Tth | ACI §22.7.4.1 | {"❌" if shear['torsion_critical'] else "✅"} |
""")
