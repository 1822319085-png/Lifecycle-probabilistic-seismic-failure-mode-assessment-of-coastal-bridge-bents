# -*- coding: utf-8 -*-
"""
lifecycle_page.py

Lifecycle probabilistic seismic failure mode assessment GUI.

This version makes the GUI sampling / deterioration / transition logic
consistent with the current lifetime-transition code:

1. One unified Latin Hypercube Sampling design is generated once.
2. All static structural/material variables and all corrosion latent
   variables are sampled once and remain fixed during the 0-100 year life.
3. Normal variables with bounds use strict truncated-normal inverse CDF.
4. Lognormal variables with bounds use strict conditional-CDF truncation
   (no post-sampling clipping mass at the bounds).
5. Uniform variables use the specified lower/upper bounds directly.
6. N uses DiscreteUniform over {2, 3, 4}.
7. Pitting factor R uses Gumbel Type-I (mu0, alpha0) parameterization:
       mu0 = 5.56, alpha0 = 1.16
       scipy loc = 5.56, scale = 1 / 1.16
8. Corrosion paths use one fixed set of corrosion parameters for each sample
   and are forced non-decreasing by maximum.accumulate.
9. Scour uses a fixed LHS quantile U_SD for each sample:
       SD_i(t) = F_t^{-1}(U_SD,i)
   using the continuous time-varying scour equation and monotonic enforcement.
10. Individual-sample failure-mode transition time is extracted using:
       - minimum persistence = 3 years
       - probability margin = 0.05
       - minimum transition year = 1
11. Two transition-time histograms are displayed below the annual failure
    mode probability curve:
       - To FFF = CFF->FFF + CSF->FFF
       - To CSF = CFF->CSF + FFF->CSF
    No fitted distribution curves are used.
12. GUI_structure.png is displayed directly below the transition-time plots.
"""

import io
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MaxNLocator
from scipy import stats, special
from scipy.stats import qmc

warnings.filterwarnings("ignore")


# ============================================================
# 1. Global settings -- synchronized with lifetime-transition code
# ============================================================
RANDOM_SEED = 2026
N_SAMPLES = 4000
YEARS_FULL = np.arange(0, 101, 1, dtype=int)

MIN_PERSIST_YEARS = 3
USE_TRANSITION_PROB_MARGIN = True
TRANSITION_PROB_MARGIN = 0.05
MIN_TRANSITION_YEAR = 1

FORCE_SCOUR_MONOTONIC = True
FORCE_CORROSION_MONOTONIC = True
SD_DISCRETIZE_STEP = None  # continuous scour depth; no 0.5 m discretization
SD_COV_DEFAULT = 0.27
TRANSITION_HIST_BIN_WIDTH = 5

DISPLAY_NAME = {
    "FFF": "FFF",
    "PFF": "CFF",
    "CFF": "CFF",
    "PSF": "CSF",
    "CSF": "CSF",
}

# Current transition-code defaults for Case2
CURRENT_CASE2_DEFAULTS = {
    "N": ("DiscreteUniform", 3.0, None, 2.0, 4.0),
    "Dp": ("Normal", 1.2, 1.2 * 0.10, 0.6, 1.8),
    "rho_pl": ("Normal", 0.010, 0.010 * 0.27, 0.005, 0.015),
    "alpha": ("Normal", 0.15, 0.15 * 0.12, 0.05, 0.25),
    "S_Dp": ("Normal", 3.0, 3.0 * 0.15, 2.5, 3.5),
    "Dr": ("Uniform", 0.55, None, 0.35, 0.75),
    "Hp_Dc": ("Normal", 3.0, 3.0 * 0.26, 1.0, 5.0),
    "Dc_Dp": ("Normal", 2.0, 2.0 * 0.10, 1.5, 3.0),
    "rho_cl": ("Normal", 0.010, 0.010 * 0.27, 0.005, 0.015),
    "rho_ps": ("Normal", 0.008, 0.008 * 0.42, 0.003, 0.013),
    "fyl": ("Lognormal", 400.0, 400.0 * 0.106, 300.0, 500.0),
    "fc": ("Lognormal", 40.0, 40.0 * 0.20, 20.0, 60.0),
    "rho_cs": ("Normal", 0.008, 0.008 * 0.42, 0.003, 0.013),
    "t": ("Normal", 0.05, 0.05 * 0.20, 0.04, 0.08),
    "d_l": ("Normal", 0.025, 0.025 * 0.10, 0.018, 0.032),
    "fyt": ("Lognormal", 350.0, 350.0 * 0.106, 250.0, 450.0),
    "d_t": ("Normal", 0.016, 0.016 * 0.10, 0.010, 0.020),
    "Acs": ("Normal", 7.758, 1.360, 1e-6, None),
    "ecs": ("Normal", 0.0, 1.105, None, None),
    "Ccr": ("Normal", 0.900, 0.150, 1e-6, None),
    "D0": ("Normal", 473.0, 43.20, 1e-6, None),
    "kc": ("Normal", 0.800, 0.100, 1e-6, None),
    "kt": ("Normal", 0.850, 0.024, 1e-6, None),
    "ke": ("Normal", 1.000, 0.300, 1e-6, None),
    "n_val": ("Beta", 0.250, 0.050, None, None),
    "X1": ("Lognormal", 1.000, 0.050, None, None),
    "lam_corr": ("Deterministic", 2.000, 0.000, None, None),
    "R": ("Gumbel", 5.560, 1.160, 1e-6, None),
    "lambda_SD": ("Deterministic", 2.000, 0.000, None, None),
    "B_val": ("Deterministic", 2.260, 0.0, None, None),
    "p_val": ("Deterministic", 1.093, 0.0, None, None),
    "q_val": ("Deterministic", 0.021, 0.0, None, None),
    "r_val": ("Deterministic", 0.269, 0.0, None, None),
    "s_val": ("Deterministic", 2.135, 0.0, None, None),
}


# ============================================================
# 2. Model loading and label helpers
# ============================================================
@st.cache_resource
def load_numpy_assets(model_path: str = "model_assets_numpy.pkl"):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def canonical_label(label):
    s = str(label)
    if s == "CFF":
        return "PFF"
    if s == "CSF":
        return "PSF"
    return s


def display_label(label):
    return DISPLAY_NAME.get(str(label), DISPLAY_NAME.get(canonical_label(label), str(label)))


def get_label_names(assets):
    if "label_names" in assets:
        return np.array(assets["label_names"]).astype(str)
    if "le" in assets:
        return np.array(assets["le"].classes_).astype(str)
    return np.array(["FFF", "PFF", "PSF"], dtype=str)


def softmax_stable(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def sigmoid_stable(z):
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def forward_softmax_numpy(X_raw, assets):
    scaler = assets["scaler"]
    weights = assets["weights"]

    a = scaler.transform(X_raw)
    for w, b in weights[:-1]:
        a = np.maximum(0.0, np.dot(a, w) + b)

    w_out, b_out = weights[-1]
    logits = np.dot(a, w_out) + b_out
    probs = softmax_stable(logits)
    pred_idx = np.argmax(probs, axis=1)
    return pred_idx, probs


def forward_sigmoid_numpy(X_scaled, weights):
    a = X_scaled
    for w, b in weights[:-1]:
        a = np.maximum(0.0, np.dot(a, w) + b)
    w_out, b_out = weights[-1]
    z = np.dot(a, w_out) + b_out
    return sigmoid_stable(z.reshape(-1))


def forward_hierarchical_numpy(X_raw, assets):
    scaler = assets["scaler"]
    X_scaled = scaler.transform(X_raw)
    label_names = get_label_names(assets)

    idx_fff = int(assets["idx_fff"])
    idx_pff = int(assets["idx_pff"])
    idx_psf = int(assets["idx_psf"])
    pff_threshold = float(assets["pff_threshold"])
    psf_threshold = float(assets["psf_threshold"])

    p_pff = forward_sigmoid_numpy(X_scaled, assets["stage1_weights"])
    p_psf_cond = forward_sigmoid_numpy(X_scaled, assets["stage2_weights"])

    probs = np.zeros((len(X_raw), len(label_names)), dtype=float)
    probs[:, idx_pff] = p_pff
    probs[:, idx_fff] = (1.0 - p_pff) * (1.0 - p_psf_cond)
    probs[:, idx_psf] = (1.0 - p_pff) * p_psf_cond

    pred_idx = np.full(len(X_raw), idx_fff, dtype=int)
    is_pff = p_pff >= pff_threshold
    pred_idx[is_pff] = idx_pff
    non_pff = ~is_pff
    pred_idx[non_pff & (p_psf_cond >= psf_threshold)] = idx_psf
    pred_idx[non_pff & (p_psf_cond < psf_threshold)] = idx_fff
    return pred_idx, probs


def predict_model(X_raw, assets):
    if "stage1_weights" in assets and "stage2_weights" in assets:
        return forward_hierarchical_numpy(X_raw, assets)
    return forward_softmax_numpy(X_raw, assets)


# ============================================================
# 3. Matplotlib / CSS
# ============================================================
def setup_matplotlib_font():
    candidate_files = [
        Path(__file__).parent / "fonts" / "times.ttf",
        Path(__file__).parent / "fonts" / "Times New Roman.ttf",
        Path(__file__).parent / "times.ttf",
        Path(__file__).parent / "Times New Roman.ttf",
    ]

    for font_file in candidate_files:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
            prop = font_manager.FontProperties(fname=str(font_file))
            font_name = prop.get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.serif"] = [font_name]
            return font_name, prop

    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" in available:
        prop = font_manager.FontProperties(family="Times New Roman")
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        return "Times New Roman", prop

    fallback = "STIXGeneral"
    prop = font_manager.FontProperties(family=fallback)
    plt.rcParams["font.family"] = fallback
    plt.rcParams["font.serif"] = [fallback, "DejaVu Serif"]
    return fallback, prop


GLOBAL_FONT_NAME, GLOBAL_FONT_PROP = setup_matplotlib_font()
plt.rcParams["font.family"] = "Times New Roman" if GLOBAL_FONT_NAME == "Times New Roman" else GLOBAL_FONT_NAME
plt.rcParams["font.serif"] = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


def inject_css():
    st.markdown(
        """
        <style>
        html, body, *,
        [data-testid="stAppViewContainer"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        .stText, .stMarkdown,
        p, span, label, button, input, textarea,
        div, h1, h2, h3, h4, h5, h6,
        table, th, td, li, ul, ol {
            font-family: 'Times New Roman', serif !important;
        }
        .block-container {
            padding-top: 1.0rem;
            padding-bottom: 2.0rem;
            max-width: 98% !important;
        }
        hr {
            margin-top: 5px;
            margin-bottom: 10px;
            border-top: 1px solid #ddd;
        }
        div[data-baseweb="input"] input {
            text-align: center !important;
            font-family: 'Times New Roman', serif !important;
            font-size: 16px !important;
        }
        div[data-baseweb="select"] div {
            font-family: 'Times New Roman', serif !important;
            font-size: 16px !important;
        }
        ul[data-baseweb="menu"] li, [role="listbox"] li {
            font-family: 'Times New Roman', serif !important;
            font-size: 16px !important;
        }
        .section-header {
            color: #800020;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 5px;
            font-family: 'Times New Roman', serif;
        }
        .col-header {
            text-align: center;
            color: #333;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Times New Roman', serif;
        }
        .plot-title {
            text-align: center;
            font-family: 'Times New Roman', serif;
            font-weight: bold;
            font-size: 17px;
            margin-bottom: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def find_all_crossovers(
    years,
    prob_a,
    prob_b,
    label_a,
    label_b,
):
    """Find all intersections of two annual probability curves."""
    crossovers = []
    diff = np.asarray(prob_a) - np.asarray(prob_b)

    for i in range(len(years) - 1):
        if diff[i] == 0:
            if i > 0 and diff[i - 1] != 0:
                slope_a = prob_a[i + 1] - prob_a[i]
                slope_b = prob_b[i + 1] - prob_b[i]

                if diff[i - 1] > 0 and slope_a < slope_b:
                    crossovers.append(
                        (
                            round(float(years[i]), 2),
                            f"{label_a} to {label_b}",
                        )
                    )
                elif diff[i - 1] < 0 and slope_a > slope_b:
                    crossovers.append(
                        (
                            round(float(years[i]), 2),
                            f"{label_b} to {label_a}",
                        )
                    )

        elif diff[i] * diff[i + 1] < 0:
            t_cross = (
                years[i]
                - diff[i]
                * (years[i + 1] - years[i])
                / (diff[i + 1] - diff[i])
            )

            slope_a = prob_a[i + 1] - prob_a[i]
            slope_b = prob_b[i + 1] - prob_b[i]

            if slope_a < slope_b:
                desc = f"{label_a} to {label_b}"
            else:
                desc = f"{label_b} to {label_a}"

            crossovers.append(
                (round(float(t_cross), 2), desc)
            )

    return crossovers

def apply_academic_style(ax):
    ax.xaxis.label.set_fontproperties(GLOBAL_FONT_PROP)
    ax.yaxis.label.set_fontproperties(GLOBAL_FONT_PROP)
    ax.title.set_fontproperties(GLOBAL_FONT_PROP)
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(GLOBAL_FONT_PROP)
        label.set_fontsize(9)

    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontproperties(GLOBAL_FONT_PROP)
            text.set_fontsize(8.5)

    ax.tick_params(
        axis="both",
        direction="in",
        top=True,
        right=True,
        labelsize=9,
        width=0.8,
        length=3.5,
    )


def set_axis_labels(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontproperties=GLOBAL_FONT_PROP, fontsize=11)
    ax.set_ylabel(ylabel, fontproperties=GLOBAL_FONT_PROP, fontsize=11)


# ============================================================
# 4. Sampling -- same rules as current lifetime-transition code
# ============================================================
def get_samples(u_array, dist_type, mean, std, p_min, p_max):
    u_array = np.clip(
        np.asarray(u_array, dtype=float),
        1e-10,
        1.0 - 1e-10
    )

    if dist_type == "Deterministic":
        return np.full_like(u_array, float(mean), dtype=float)

    if dist_type == "DiscreteUniform":
        lo = int(round(p_min))
        hi = int(round(p_max))
        values = np.arange(lo, hi + 1, dtype=float)
        idx = np.floor(u_array * len(values)).astype(int)
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    if dist_type == "Uniform":
        # IMPORTANT: identical to current transition code:
        # Uniform uses lower/upper bounds directly; mean/std are ignored.
        lower = float(p_min)
        upper = float(p_max)
        return lower + u_array * (upper - lower)

    lower = -np.inf if p_min is None else float(p_min)
    upper = np.inf if p_max is None else float(p_max)

    if dist_type == "Normal":
        if mean is None or std is None or std <= 0:
            raise ValueError(f"Invalid Normal parameters: mean={mean}, std={std}")

        if np.isneginf(lower) and np.isposinf(upper):
            s = stats.norm.ppf(u_array, loc=mean, scale=std)
        else:
            a = (lower - mean) / std
            b = (upper - mean) / std
            s = stats.truncnorm.ppf(
                u_array, a, b, loc=mean, scale=std
            )
            s = np.clip(s, lower, upper)

    elif dist_type == "Lognormal":
        if mean is None or mean <= 0 or std is None or std < 0:
            raise ValueError(
                f"Invalid Lognormal parameters: mean={mean}, std={std}"
            )

        if std == 0:
            s = np.full_like(u_array, float(mean), dtype=float)
        else:
            sigma2 = np.log(1.0 + (std / mean) ** 2)
            sigma_ln = np.sqrt(sigma2)
            mu_ln = np.log(mean) - 0.5 * sigma2
            base_dist = stats.lognorm(
                s=sigma_ln,
                scale=np.exp(mu_ln)
            )

            if p_min is None and p_max is None:
                s = base_dist.ppf(u_array)
            else:
                lower_cdf = (
                    0.0
                    if p_min is None or lower <= 0.0
                    else float(base_dist.cdf(lower))
                )
                upper_cdf = (
                    1.0
                    if p_max is None or np.isposinf(upper)
                    else float(base_dist.cdf(upper))
                )
                u_trunc = lower_cdf + u_array * (upper_cdf - lower_cdf)
                u_trunc = np.clip(u_trunc, 1e-12, 1.0 - 1e-12)
                s = base_dist.ppf(u_trunc)
                s = np.clip(s, lower, upper)

    elif dist_type == "Beta":
        var = std ** 2
        temp = mean * (1.0 - mean) / var - 1.0
        a = mean * temp
        b = (1.0 - mean) * temp
        if a <= 0 or b <= 0:
            raise ValueError(
                f"Invalid Beta parameters: mean={mean}, std={std}"
            )
        s = stats.beta.ppf(u_array, a, b)

    elif dist_type == "Gumbel":
        # R follows F(r)=exp{-exp[-alpha0*(r-mu0)]}
        mu0 = float(mean)
        alpha0 = float(std)
        s = stats.gumbel_r.ppf(
            u_array,
            loc=mu0,
            scale=1.0 / alpha0
        )
        if p_min is not None or p_max is not None:
            s = np.clip(s, lower, upper)

    else:
        raise ValueError(f"Unsupported distribution: {dist_type}")

    if not np.all(np.isfinite(s)):
        raise ValueError(f"{dist_type} sampling produced NaN/Inf.")

    return np.asarray(s, dtype=float)


def generate_samples_dict(all_inputs, n_samples=N_SAMPLES, seed=RANDOM_SEED):
    sampler = qmc.LatinHypercube(
        d=len(all_inputs),
        seed=seed
    )
    U = sampler.random(n=n_samples)

    samples_dict = {}
    for idx, p in enumerate(all_inputs):
        if p["id"] == "SD_val":
            # SD_val is only a fixed LHS quantile driver for the scour path.
            samples_dict[p["id"]] = U[:, idx].copy()
            continue

        samples_dict[p["id"]] = get_samples(
            U[:, idx],
            p["dist"],
            p["mean"],
            p["std"],
            p["min"],
            p["max"],
        )

    return samples_dict, U


# ============================================================
# 5. Corrosion and scour path generation
# ============================================================
def compute_T_init(
    cover_mm, D0, k_e, k_t, k_c, n,
    C_cr, C0_arr, X1
):
    valid_mask = (
        (C0_arr > C_cr)
        & (C0_arr > 0)
        & (C_cr > 0)
    )

    safe_C0 = np.where(valid_mask, C0_arr, 2.0)
    safe_Ccr = np.where(valid_mask, C_cr, 1.0)
    ratio = np.clip(
        safe_Ccr / safe_C0,
        1e-12,
        1.0 - 1e-12
    )
    inv_erf = special.erfinv(1.0 - ratio)

    denom = (
        4.0
        * k_e
        * k_t
        * k_c
        * D0
        * (1.0 ** n)
        * (inv_erf ** 2)
    )

    valid_denom = denom > 0
    safe_denom = np.where(valid_denom, denom, 1.0)
    base_val = np.maximum(
        cover_mm ** 2 / safe_denom,
        1e-12
    )

    T = X1 * (
        base_val
        ** (
            1.0
            / np.maximum(1.0 - n, 1e-6)
        )
    )
    return np.where(
        valid_mask & valid_denom,
        T,
        np.inf
    )


def pitting_corrosion_matrix(
    d_rein_mm,
    T_init_arr,
    cover_mm,
    years_arr,
    R_arr,
    lambda_arr,
    w_b=0.5,
):
    i_corr0 = (
        37.8
        * lambda_arr
        * (1.0 - w_b) ** (-1.64)
        / cover_mm
    )

    A0 = np.pi * d_rein_mm ** 2 / 4.0
    corr_rate = np.zeros(
        (len(d_rein_mm), len(years_arr)),
        dtype=float
    )

    for y_idx, yr in enumerate(years_arr):
        t_p = np.where(
            np.isinf(T_init_arr),
            0.0,
            np.maximum(0.0, yr - T_init_arr)
        )

        diam_loss_uniform = (
            2.0
            * (
                0.0116
                * 0.85
                * i_corr0
                * (t_p ** 0.71)
                / 0.71
            )
        )

        d_rem_uniform = np.maximum(
            0.0,
            d_rein_mm - diam_loss_uniform
        )
        Au = np.pi * d_rem_uniform ** 2 / 4.0

        p_val = R_arr * (diam_loss_uniform / 2.0)
        A_rem = np.copy(A0)
        mask_p = p_val > 0

        if np.any(mask_p):
            p_v = np.minimum(
                p_val[mask_p],
                d_rein_mm[mask_p]
            )
            dr = d_rein_mm[mask_p]
            inner_val = np.maximum(
                0.0,
                1.0 - (p_v / dr) ** 2
            )
            a = 2.0 * p_v * np.sqrt(inner_val)

            theta1 = 2.0 * np.arcsin(
                np.clip(a / dr, -1.0, 1.0)
            )
            A1 = 0.5 * (
                theta1 * (0.5 * dr) ** 2
                - a * (0.5 * dr - p_v ** 2 / dr)
            )

            theta2 = np.where(
                p_v > 1e-12,
                2.0 * np.arcsin(
                    np.clip(a / (2.0 * p_v), -1.0, 1.0)
                ),
                0.0,
            )
            A2 = 0.5 * (
                theta2 * p_v ** 2
                - a * p_v ** 2 / dr
            )

            ADP = np.where(
                p_v <= dr / np.sqrt(2.0),
                A0[mask_p] - A1 - A2,
                A1 - A2,
            )

            Ap = (
                1.0 - a / (2.0 * dr)
            ) * (
                Au[mask_p] - A0[mask_p]
            ) + ADP

            A_rem[mask_p] = np.clip(
                Ap,
                0.0,
                A0[mask_p]
            )

        corr_rate[:, y_idx] = np.clip(
            1.0 - A_rem / A0,
            0.0,
            1.0
        )

    if FORCE_CORROSION_MONOTONIC:
        corr_rate = np.maximum.accumulate(
            corr_rate,
            axis=1
        )

    return corr_rate


def generate_corrosion_paths(samples_dict, years_arr):
    t_samples = np.asarray(samples_dict["t"], dtype=float)
    dl_samples = np.asarray(samples_dict["d_l"], dtype=float)
    dt_samples = np.asarray(samples_dict["d_t"], dtype=float)

    C0 = (
        np.asarray(samples_dict["Acs"], dtype=float) * 0.5
        + np.asarray(samples_dict["ecs"], dtype=float)
    )

    tc_mm = t_samples * 1000.0
    cover_stir_mm = np.maximum(
        tc_mm - dt_samples * 1000.0,
        15.0
    )

    T_init_long = compute_T_init(
        tc_mm,
        samples_dict["D0"],
        samples_dict["ke"],
        samples_dict["kt"],
        samples_dict["kc"],
        samples_dict["n_val"],
        samples_dict["Ccr"],
        C0,
        samples_dict["X1"],
    )

    T_init_stir = compute_T_init(
        cover_stir_mm,
        samples_dict["D0"],
        samples_dict["ke"],
        samples_dict["kt"],
        samples_dict["kc"],
        samples_dict["n_val"],
        samples_dict["Ccr"],
        C0,
        samples_dict["X1"],
    )

    corr_long = pitting_corrosion_matrix(
        dl_samples * 1000.0,
        T_init_long,
        tc_mm,
        years_arr,
        samples_dict["R"],
        samples_dict["lam_corr"],
    )

    corr_stir = pitting_corrosion_matrix(
        dt_samples * 1000.0,
        T_init_stir,
        cover_stir_mm,
        years_arr,
        samples_dict["R"],
        samples_dict["lam_corr"],
    )

    return (
        corr_long,
        corr_stir,
        T_init_long,
        T_init_stir,
    )


def generate_scour_paths(
    samples_dict,
    U,
    all_inputs,
    years_arr,
    sd_cov,
):
    """
    Continuous stochastic scour depth with a fixed lifecycle LHS quantile.

    For each sample i, U_SD,i is generated once and retained for all years.
    The yearly distribution is:

        SD_mean(t) = lambda_SD * B *
                     {p[1-exp(-q t)] + r[1-exp(-s t)]}

        SD(t) ~ Truncated Normal(mean=SD_mean(t),
                                 std=COV_SD*SD_mean(t),
                                 range=[0, 8] m)

    No 0.5 m discretization is used.
    """
    input_ids = [p["id"] for p in all_inputs]
    sd_idx = input_ids.index("SD_val")
    U_SD = np.clip(U[:, sd_idx], 1e-10, 1.0 - 1e-10)

    p_arr = np.asarray(samples_dict["p_val"], dtype=float)
    q_arr = np.asarray(samples_dict["q_val"], dtype=float)
    r_arr = np.asarray(samples_dict["r_val"], dtype=float)
    s_arr = np.asarray(samples_dict["s_val"], dtype=float)
    B_arr = np.asarray(samples_dict["B_val"], dtype=float)
    lambda_sd_arr = np.asarray(samples_dict["lambda_SD"], dtype=float)

    raw = np.zeros((len(U_SD), len(years_arr)), dtype=float)

    for y_idx, yr in enumerate(years_arr):
        sd_mean = (
            lambda_sd_arr * B_arr *
            (p_arr * (1.0 - np.exp(-q_arr * yr))
             + r_arr * (1.0 - np.exp(-s_arr * yr)))
        )

        zero_mask = np.isclose(sd_mean, 0.0, atol=1e-14)
        dynamic_std = np.maximum(sd_mean * float(sd_cov), 1e-10)
        a = (0.0 - sd_mean) / dynamic_std
        b = (8.0 - sd_mean) / dynamic_std

        sd_samples = stats.truncnorm.ppf(
            U_SD,
            a,
            b,
            loc=sd_mean,
            scale=dynamic_std,
        )
        sd_samples = np.where(zero_mask, 0.0, sd_samples)
        raw[:, y_idx] = np.clip(sd_samples, 0.0, 8.0)

    if FORCE_SCOUR_MONOTONIC:
        raw = np.maximum.accumulate(raw, axis=1)

    return raw


# ============================================================
# 6. 20D assembly and lifecycle prediction
# ============================================================
def build_base_feature_matrix(samples_dict, n_samples):
    X_fixed = np.zeros((n_samples, 20), dtype=float)

    mapping = [
        ("N", 0),
        ("Dp", 1),
        ("rho_pl", 2),
        ("alpha", 3),
        ("S_Dp", 4),
        ("Dr", 5),
        ("Hp_Dc", 7),
        ("Dc_Dp", 8),
        ("rho_cl", 9),
        ("rho_ps", 10),
        ("fyl", 11),
        ("fc", 12),
        ("rho_cs", 13),
        ("t", 16),
        ("d_l", 17),
        ("fyt", 18),
        ("d_t", 19),
    ]

    for key, col_idx in mapping:
        X_fixed[:, col_idx] = samples_dict[key]

    return X_fixed


def run_lifecycle_prediction(
    assets,
    all_inputs,
    sd_cov,
    n_samples=N_SAMPLES,
    seed=RANDOM_SEED,
):
    label_names = get_label_names(assets)

    samples_dict, U = generate_samples_dict(
        all_inputs,
        n_samples=n_samples,
        seed=seed,
    )

    (
        corr_long,
        corr_stir,
        T_init_long,
        T_init_stir,
    ) = generate_corrosion_paths(
        samples_dict,
        YEARS_FULL
    )

    scour_depths = generate_scour_paths(
        samples_dict,
        U,
        all_inputs,
        YEARS_FULL,
        sd_cov=sd_cov,
    )

    X_base = build_base_feature_matrix(
        samples_dict,
        n_samples
    )

    n_classes = len(label_names)
    n_years = len(YEARS_FULL)

    mode_paths = np.zeros(
        (n_samples, n_years),
        dtype=int
    )
    prob_paths = np.zeros(
        (n_samples, n_years, n_classes),
        dtype=float
    )
    annual_probs = np.zeros(
        (n_years, n_classes),
        dtype=float
    )

    for y_idx, yr in enumerate(YEARS_FULL):
        X_year = X_base.copy()
        X_year[:, 6] = scour_depths[:, y_idx]
        X_year[:, 14] = corr_stir[:, y_idx]
        X_year[:, 15] = corr_long[:, y_idx]

        pred_idx, probs = predict_model(
            X_year,
            assets
        )

        mode_paths[:, y_idx] = pred_idx
        prob_paths[:, y_idx, :] = probs
        annual_probs[y_idx, :] = (
            np.bincount(
                pred_idx,
                minlength=n_classes
            ).astype(float)
            / n_samples
        )

    transition_df = get_transition_table(
        mode_paths,
        prob_paths,
        label_names,
        YEARS_FULL,
    )

    hist_df = get_grouped_transition_histogram(
        transition_df,
        bin_width=TRANSITION_HIST_BIN_WIDTH,
    )

    return {
        "label_names": label_names,
        "samples_dict": samples_dict,
        "U": U,
        "T_init_long": T_init_long,
        "T_init_stir": T_init_stir,
        "corr_long": corr_long,
        "corr_stir": corr_stir,
        "scour_depths": scour_depths,
        "mode_paths": mode_paths,
        "prob_paths": prob_paths,
        "annual_probs": annual_probs,
        "transition_df": transition_df,
        "hist_df": hist_df,
    }


# ============================================================
# 7. Individual transition-time extraction
# ============================================================
def get_transition_table(
    mode_paths,
    prob_paths,
    label_names,
    years_arr,
):
    n_samples, n_years = mode_paths.shape
    label_names = np.asarray(label_names).astype(str)

    initial_modes = mode_paths[:, 0].copy()
    transition_years = np.full(
        n_samples,
        years_arr[-1],
        dtype=int
    )
    transition_flag = np.zeros(
        n_samples,
        dtype=bool
    )
    transition_modes = initial_modes.copy()
    transition_margin = np.full(
        n_samples,
        np.nan,
        dtype=float
    )

    search_end = (
        n_years
        if MIN_PERSIST_YEARS <= 1
        else n_years - MIN_PERSIST_YEARS + 1
    )

    for i in range(n_samples):
        initial = initial_modes[i]

        for j in range(1, search_end):
            yr = int(years_arr[j])

            if yr < MIN_TRANSITION_YEAR:
                continue

            new_mode = mode_paths[i, j]
            if new_mode == initial:
                continue

            if MIN_PERSIST_YEARS <= 1:
                segment_modes = np.array([new_mode])
                segment_probs = prob_paths[i, j:j + 1, :]
            else:
                segment_modes = mode_paths[
                    i,
                    j:j + MIN_PERSIST_YEARS
                ]
                segment_probs = prob_paths[
                    i,
                    j:j + MIN_PERSIST_YEARS,
                    :
                ]

            if not np.all(segment_modes == new_mode):
                continue

            if USE_TRANSITION_PROB_MARGIN:
                margins = (
                    segment_probs[:, new_mode]
                    - segment_probs[:, initial]
                )
                if not np.all(
                    margins >= TRANSITION_PROB_MARGIN
                ):
                    continue
                margin_at_j = float(margins[0])
            else:
                margin_at_j = float(
                    prob_paths[i, j, new_mode]
                    - prob_paths[i, j, initial]
                )

            transition_years[i] = yr
            transition_flag[i] = True
            transition_modes[i] = new_mode
            transition_margin[i] = margin_at_j
            break

    initial_labels_raw = label_names[initial_modes]
    transition_labels_raw = label_names[transition_modes]

    initial_labels = np.array(
        [canonical_label(x) for x in initial_labels_raw],
        dtype=object
    )
    transition_labels = np.array(
        [canonical_label(x) for x in transition_labels_raw],
        dtype=object
    )

    transition_type = []
    for i in range(n_samples):
        if transition_flag[i]:
            transition_type.append(
                f"{initial_labels[i]}->{transition_labels[i]}"
            )
        else:
            transition_type.append(
                "No transition within 100 years"
            )

    return pd.DataFrame({
        "Sample_ID": np.arange(1, n_samples + 1),
        "Initial_Mode": initial_labels,
        "Initial_Mode_Display": [
            display_label(x)
            for x in initial_labels
        ],
        "Transition_Mode": transition_labels,
        "Transition_Mode_Display": [
            display_label(x)
            for x in transition_labels
        ],
        "Transition_Type": transition_type,
        "Transition_Year": transition_years,
        "Transition_Flag": transition_flag,
        "Transition_Probability_Margin": transition_margin,
    })


def get_grouped_transition_histogram(
    transition_df,
    bin_width=5,
):
    df_tr = transition_df[
        transition_df["Transition_Flag"]
    ].copy()

    groups = {
        "To_FFF": [
            "PFF->FFF",
            "PSF->FFF",
        ],
        "To_CSF": [
            "PFF->PSF",
            "FFF->PSF",
        ],
    }

    display = {
        "To_FFF":
            "To FFF (CFF→FFF + CSF→FFF)",
        "To_CSF":
            "To CSF (CFF→CSF + FFF→CSF)",
    }

    edges = np.arange(
        0,
        100 + bin_width,
        bin_width,
        dtype=float
    )

    rows = []
    for group_name, types in groups.items():
        years = df_tr.loc[
            df_tr["Transition_Type"].isin(types),
            "Transition_Year",
        ].to_numpy(dtype=float)

        counts, _ = np.histogram(
            years,
            bins=edges
        )
        n_group = len(years)

        for k, count in enumerate(counts):
            rows.append({
                "Target_Group": group_name,
                "Target_Group_Display": display[group_name],
                "Bin_Start": edges[k],
                "Bin_End": edges[k + 1],
                "Bin_Center": 0.5 * (
                    edges[k] + edges[k + 1]
                ),
                "Count": int(count),
                "Probability_within_group":
                    count / n_group
                    if n_group > 0
                    else 0.0,
                "Probability_in_all_samples":
                    count / len(transition_df),
                "N_group": int(n_group),
            })

    return pd.DataFrame(rows)


# ============================================================
# 8. UI builders
# ============================================================
def render_param_section(
    title,
    params_config,
    use_std=False,
):
    if title:
        st.markdown(
            f"<div class='section-header'>{title}</div>",
            unsafe_allow_html=True
        )

    cols = st.columns(
        [1.2, 2.8, 1.6, 1.1, 1.1, 1.2]
    )
    headers = [
        "Parameter",
        "Description",
        "Distribution",
        "Mean",
        "St. dev. / α" if use_std else "COV",
        "Range",
    ]

    for i, h in enumerate(headers):
        cols[i].markdown(
            f"<div class='col-header'>{h}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    user_vals = []

    for cfg in params_config:
        (
            p_id,
            html_name,
            desc,
            rng_str,
            p_min,
            p_max,
            p_mean,
            p_dist,
            p_disp,
            p_step,
            p_fmt,
            dist_opts,
        ) = cfg

        c1, c2, c3, c4, c5, c6 = st.columns(
            [1.2, 2.8, 1.6, 1.1, 1.1, 1.2]
        )

        c1.markdown(
            f"<div style='text-align:center;font-weight:bold;"
            f"padding-top:8px'>{html_name}</div>",
            unsafe_allow_html=True
        )
        c2.markdown(
            f"<div style='text-align:center;color:#444;"
            f"font-size:14px;padding-top:8px'>{desc}</div>",
            unsafe_allow_html=True
        )

        with c3:
            dist_val = st.selectbox(
                label=f"{p_id}_dist",
                options=dist_opts,
                index=dist_opts.index(p_dist),
                label_visibility="collapsed",
            )

        mean_disabled = (
            p_mean is None
            or dist_val in ("Uniform", "DiscreteUniform")
        )

        with c4:
            kwargs = {}
            if p_min is not None:
                kwargs["min_value"] = float(p_min)
            if p_max is not None:
                kwargs["max_value"] = float(p_max)

            mean_val = st.number_input(
                label=f"{p_id}_mean",
                value=0.0 if p_mean is None else float(p_mean),
                step=float(p_step),
                format=p_fmt,
                disabled=mean_disabled,
                label_visibility="collapsed",
                **kwargs,
            )

        disp_disabled = (
            dist_val in (
                "Deterministic",
                "Uniform",
                "DiscreteUniform",
            )
        )

        with c5:
            disp_val = st.number_input(
                label=f"{p_id}_disp",
                min_value=0.0,
                value=(
                    0.0
                    if disp_disabled
                    else float(p_disp)
                ),
                step=0.05,
                format="%.3f",
                disabled=disp_disabled,
                label_visibility="collapsed",
            )

        c6.markdown(
            f"<div style='text-align:center;color:#666;"
            f"font-size:15px;padding-top:8px'>{rng_str}</div>",
            unsafe_allow_html=True
        )

        if use_std:
            # Corrosion-table dispersion column is interpreted directly as
            # standard deviation, except for Gumbel where it is alpha0.
            # In both cases the current GUI value must be passed through.
            std_val = float(disp_val)
        else:
            if dist_val in (
                "Deterministic",
                "Uniform",
                "DiscreteUniform",
            ):
                std_val = 0.0
            else:
                std_val = float(mean_val) * float(disp_val)

        user_vals.append({
            "id": p_id,
            "mean": float(mean_val),
            "std": std_val,
            "raw_disp": float(
                p_disp
                if p_id == "SD_val"
                else disp_val
            ),
            "dist": dist_val,
            "min": p_min,
            "max": p_max,
        })

    st.write("")
    return user_vals


# ============================================================
# 9. Plot helpers
# ============================================================
def make_initiation_plot(
    T_init_long,
    T_init_stir,
):
    """Original histogram colors + retained lognormal fitted PDFs."""
    fig, ax = plt.subplots(
        figsize=(6, 3.3),
        dpi=220
    )

    t_long = T_init_long[
        np.isfinite(T_init_long)
        & (T_init_long <= 100)
    ]
    t_stir = T_init_stir[
        np.isfinite(T_init_stir)
        & (T_init_stir <= 100)
    ]

    # Restore the original GUI palette.
    color_hist_s = "#CBE5F5"
    color_line_s = "#0000FF"
    color_hist_l = "#FADBDC"
    color_line_l = "#FF0000"

    ax.hist(
        t_stir,
        bins=80,
        rwidth=1.0,
        density=True,
        alpha=0.8,
        color=color_hist_s,
        edgecolor="gray",
        linewidth=0.5,
        label="Transverse frequency",
    )
    ax.hist(
        t_long,
        bins=80,
        rwidth=1.0,
        density=True,
        alpha=0.6,
        color=color_hist_l,
        edgecolor="gray",
        linewidth=0.5,
        label="Longitudinal frequency",
    )

    # Keep the fitted distributions for this figure.
    stir_fit = t_stir[t_stir > 0]
    if len(stir_fit) > 5:
        shape_s, _, scale_s = stats.lognorm.fit(
            stir_fit,
            floc=0
        )
        x_s = np.linspace(1e-4, 100, 1000)
        ax.plot(
            x_s,
            stats.lognorm.pdf(
                x_s,
                shape_s,
                loc=0,
                scale=scale_s,
            ),
            color=color_line_s,
            lw=2.3,
            label="Transverse lognormal distribution",
        )

    long_fit = t_long[t_long > 0]
    if len(long_fit) > 5:
        shape_l, _, scale_l = stats.lognorm.fit(
            long_fit,
            floc=0
        )
        x_l = np.linspace(1e-4, 100, 1000)
        ax.plot(
            x_l,
            stats.lognorm.pdf(
                x_l,
                shape_l,
                loc=0,
                scale=scale_l,
            ),
            color=color_line_l,
            lw=2.3,
            label="Longitudinal lognormal distribution",
        )

    set_axis_labels(
        ax,
        "Initial corrosion time (years)",
        "Probability density",
    )
    ax.set_xlim(0, 30)

    y_max = ax.get_ylim()[1]
    rounded_ymax = (
        np.ceil(y_max * 10.0) / 10.0
        if y_max > 0
        else 0.1
    )
    ax.set_ylim(0, rounded_ymax)
    ax.set_yticks(
        np.linspace(0, rounded_ymax, 5)
    )

    ax.legend(
        frameon=False,
        loc="upper right",
    )
    ax.grid(False)
    apply_academic_style(ax)
    fig.tight_layout(pad=0.35)
    return fig


def make_corrosion_plot(
    years,
    corr_long,
    corr_stir,
):
    fig, ax = plt.subplots(
        figsize=(6, 3.3),
        dpi=220
    )

    med_l = np.median(corr_long, axis=0)
    p16_l = np.percentile(corr_long, 16, axis=0)
    p84_l = np.percentile(corr_long, 84, axis=0)

    med_s = np.median(corr_stir, axis=0)
    p16_s = np.percentile(corr_stir, 16, axis=0)
    p84_s = np.percentile(corr_stir, 84, axis=0)

    ax.plot(
        years,
        med_s,
        color="#3B82F6",
        lw=2.2,
        label="Transverse (median)",
    )
    ax.fill_between(
        years,
        p16_s,
        p84_s,
        color="#BFDBFE",
        alpha=0.62,
        label="Transverse (16%-84%)",
    )

    ax.plot(
        years,
        med_l,
        color="#E45756",
        lw=2.2,
        label="Longitudinal (median)",
    )
    ax.fill_between(
        years,
        p16_l,
        p84_l,
        color="#FBC4C4",
        alpha=0.55,
        label="Longitudinal (16%-84%)",
    )

    set_axis_labels(
        ax,
        "Service time (years)",
        "Corrosion level",
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(
        0,
        max(
            0.1,
            np.ceil(
                max(
                    np.max(p84_l),
                    np.max(p84_s),
                )
                * 10
            )
            / 10,
        ),
    )
    ax.legend(frameon=False, loc="upper left")
    ax.grid(False)
    apply_academic_style(ax)
    fig.tight_layout(pad=0.35)
    return fig


def make_scour_plot(
    years,
    scour_depths,
):
    fig, ax = plt.subplots(figsize=(6, 3.3), dpi=220)

    med = np.median(scour_depths, axis=0)
    p16 = np.percentile(scour_depths, 16, axis=0)
    p84 = np.percentile(scour_depths, 84, axis=0)

    ax.plot(
        years,
        med,
        color="#1F7A8C",
        lw=2.4,
        label="Scour depth (median)",
        zorder=3,
    )
    ax.fill_between(
        years,
        p16,
        p84,
        color="#B8E0E6",
        alpha=0.62,
        label="Scour depth (16%-84% quantiles)",
        zorder=2,
    )

    set_axis_labels(ax, "Service time (years)", "Scour depth (m)")
    ax.set_xlim(0, 100)

    max_scour = float(np.max(p84))
    y_top = max(1, int(np.ceil(max_scour)))
    ax.set_ylim(0, y_top)
    step = 1 if y_top <= 8 else max(1, int(np.ceil(y_top / 6.0)))
    ax.set_yticks(np.arange(0, y_top + step, step))

    ax.legend(frameon=False, loc="upper left")
    ax.grid(False)
    apply_academic_style(ax)
    fig.tight_layout(pad=0.35)
    return fig


def make_failure_probability_plot(
    years,
    annual_probs,
    label_names,
):
    fig, ax = plt.subplots(
        figsize=(6, 3.3),
        dpi=220
    )

    # User-specified RGB colors /255:
    # FFF [70,192,115], CFF [60,117,189], CSF [224,47,98]
    colors = {
        "FFF": "#46C073",
        "PFF": "#3C75BD",
        "CFF": "#3C75BD",
        "PSF": "#E02F62",
        "CSF": "#E02F62",
    }

    for idx, name in enumerate(label_names):
        canonical = canonical_label(name)
        ax.plot(
            years,
            annual_probs[:, idx],
            color=colors.get(
                str(name),
                colors.get(canonical, "#555555"),
            ),
            lw=2.4,
            label=display_label(name),
        )

    set_axis_labels(
        ax,
        "Service time (years)",
        "Probability",
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(
        np.arange(0.0, 1.01, 0.2)
    )

    ax.legend(
        frameon=False,
        loc="upper right",
        ncol=3,
    )
    ax.grid(False)
    apply_academic_style(ax)
    fig.tight_layout(pad=0.35)
    return fig


def make_transition_histogram(
    hist_df,
    group_name,
):
    g = hist_df[
        hist_df["Target_Group"] == group_name
    ].copy()

    # Softer publication-style palette.
    if group_name == "To_FFF":
        color = "#6F9FC8"
        title = (
            "Transition to FFF\n"
            "(CFF→FFF + CSF→FFF)"
        )
    else:
        color = "#D78996"
        title = (
            "Transition to CSF\n"
            "(CFF→CSF + FFF→CSF)"
        )

    n_group = (
        int(g["N_group"].iloc[0])
        if not g.empty
        else 0
    )

    fig, ax = plt.subplots(
        figsize=(6, 2.85),
        dpi=220
    )

    if not g.empty:
        ax.bar(
            g["Bin_Center"],
            g["Count"],
            width=TRANSITION_HIST_BIN_WIDTH * 0.84,
            color=color,
            edgecolor="#FFFFFF",
            linewidth=0.8,
            alpha=0.95,
        )

    set_axis_labels(
        ax,
        "Transition time (years)",
        "Frequency",
    )
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))

    max_count = (
        int(g["Count"].max())
        if not g.empty
        else 0
    )

    def _nice_integer_step(value, target_intervals=4):
        if value <= 0:
            return 1
        raw = value / float(target_intervals)
        magnitude = 10 ** int(np.floor(np.log10(max(raw, 1e-12))))
        normalized = raw / magnitude
        if normalized <= 1:
            nice = 1
        elif normalized <= 2:
            nice = 2
        elif normalized <= 5:
            nice = 5
        else:
            nice = 10
        return max(1, int(nice * magnitude))

    y_step = _nice_integer_step(max_count, target_intervals=4)
    y_top = max(
        y_step,
        int(np.ceil(max_count / y_step) * y_step)
    )
    ax.set_ylim(0, y_top)
    ax.set_yticks(np.arange(0, y_top + y_step, y_step))

    ax.set_title(
        f"{title}  (n={n_group})",
        fontsize=10,
        fontproperties=GLOBAL_FONT_PROP,
        pad=5,
    )

    ax.grid(
        axis="y",
        alpha=0.16,
        linestyle="--",
        linewidth=0.5,
    )

    # Add a complete box around both transition histograms.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#4A4A4A")

    apply_academic_style(ax)
    fig.tight_layout(pad=0.35)
    return fig


# ============================================================
# 10. Main lifecycle page
# ============================================================
def render_lifecycle_app(assets=None):
    """
    Render lifecycle assessment page.

    Parameters
    ----------
    assets : dict or None
        Optional model assets passed from app.py. If None, this page loads
        model_assets_numpy.pkl internally. This keeps compatibility with both:
            render_lifecycle_app()
        and:
            render_lifecycle_app(assets=assets)
    """
    inject_css()

    if assets is None:
        assets = load_numpy_assets()

    st.markdown(
        """
        <h1 style='text-align:center;color:#333;
        font-family:"Times New Roman",serif;
        font-weight:bold;margin-bottom:0px;'>
        Lifecycle probabilistic seismic failure mode assessment of coastal bridge bents
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    col_left, spacer, col_right = st.columns(
        [6.8, 0.2, 3.0]
    )

    with col_left:
        struct_opts = [
            "Normal",
            "Lognormal",
            "Uniform",
            "DiscreteUniform",
            "Deterministic",
        ]

        part1_config = [
            ("N", "N", "Number of pile rows along loading direction", "2~4",
             2.0, 4.0, 3.0, "DiscreteUniform", 0.0, 1.0, "%.0f", struct_opts),
            ("Dp", "D<sub>p</sub> (m)", "Pile diameter", "0.6~1.8",
             0.6, 1.8, 1.2, "Normal", 0.10, 0.1, "%.2f", struct_opts),
            ("rho_pl", "ρ<sub>p,l</sub>", "Pile longitudinal reinforcement ratio", "0.005~0.015",
             0.005, 0.015, 0.010, "Normal", 0.27, 0.001, "%.3f", struct_opts),
            ("alpha", "α", "Column axial load ratio", "0.05~0.25",
             0.05, 0.25, 0.15, "Normal", 0.12, 0.01, "%.2f", struct_opts),
            ("S_Dp", "S/D<sub>p</sub>", "Pile spacing-to-diameter ratio", "2.5~3.5",
             2.5, 3.5, 3.0, "Normal", 0.15, 0.1, "%.2f", struct_opts),
            ("Dr", "D<sub>r</sub>", "Relative density of sand", "0.35~0.75",
             0.35, 0.75, 0.55, "Uniform", 0.0, 0.05, "%.2f", struct_opts),
            ("Hp_Dc", "H<sub>p</sub>/D<sub>c</sub>", "Column aspect ratio", "1~5",
             1.0, 5.0, 3.0, "Normal", 0.26, 0.1, "%.2f", struct_opts),
            ("Dc_Dp", "D<sub>c</sub>/D<sub>p</sub>", "Column-to-pile diameter ratio", "1.5~3.0",
             1.5, 3.0, 2.0, "Normal", 0.10, 0.1, "%.2f", struct_opts),
            ("rho_cl", "ρ<sub>c,l</sub>", "Column longitudinal reinforcement ratio", "0.005~0.015",
             0.005, 0.015, 0.010, "Normal", 0.27, 0.001, "%.3f", struct_opts),
            ("rho_ps", "ρ<sub>p,s</sub>", "Pile transverse reinforcement ratio", "0.003~0.013",
             0.003, 0.013, 0.008, "Normal", 0.42, 0.001, "%.3f", struct_opts),
            ("fyl", "f<sub>yl</sub> (MPa)", "Longitudinal reinforcement yield strength", "300~500",
             300.0, 500.0, 400.0, "Lognormal", 0.106, 10.0, "%.0f", struct_opts),
            ("fc", "f<sub>c</sub> (MPa)", "Concrete compressive strength", "20~60",
             20.0, 60.0, 40.0, "Lognormal", 0.20, 1.0, "%.1f", struct_opts),
            ("rho_cs", "ρ<sub>c,s</sub>", "Column transverse reinforcement ratio", "0.003~0.013",
             0.003, 0.013, 0.008, "Normal", 0.42, 0.001, "%.3f", struct_opts),
            ("t", "t<sub>c</sub (m)", "Column cover concrete thickness", "0.04~0.08",
             0.04, 0.08, 0.05, "Normal", 0.20, 0.01, "%.2f", struct_opts),
            ("d_l", "d<sub>l</sub> (m)", "Column longitudinal reinforcement diameter", "0.018~0.032",
             0.018, 0.032, 0.025, "Normal", 0.10, 0.001, "%.3f", struct_opts),
            ("fyt", "f<sub>ys</sub> (MPa)", "Transverse reinforcement yield strength", "250~450",
             250.0, 450.0, 350.0, "Lognormal", 0.106, 10.0, "%.0f", struct_opts),
            ("d_t", "d<sub>s</sub> (m)", "Transverse reinforcement diameter", "0.01~0.02",
             0.01, 0.02, 0.016, "Normal", 0.10, 0.001, "%.3f", struct_opts),
        ]

        user_struct = render_param_section(
            "1. Structure/Soil-related parameters",
            part1_config,
            use_std=False,
        )

        st.markdown(
            "<div class='section-header'>2. Corrosion-related parameters</div>",
            unsafe_allow_html=True
        )

        col_f1, col_f2 = st.columns([1.45, 1.05])
        with col_f1:
            st.latex(
                r"t_{corr}=X_1\left[\frac{d_c^2}{4k_ek_tk_cD_0(t_0)^n}"
                r"\left[\mathrm{erf}^{-1}\left(1-\frac{C_{cr}}{C_0}\right)\right]^{-2}"
                r"\right]^{\frac{1}{1-n}}"
            )
        with col_f2:
            st.latex(r"C_0=A_{cs}(w/c)+\varepsilon_{cs}")
            st.latex(
                r"i_{corr,0}=\frac{37.8\,\lambda_{corr}(1-w_b)^{-1.64}}{d_c}"
            )

        col_z1, col_z2 = st.columns([1.5, 2])
        with col_z1:
            st.markdown(
                "<div style='text-align:right;font-size:16px;"
                "font-weight:bold;padding-top:5px;'>"
                "Select Environmental Zone:</div>",
                unsafe_allow_html=True,
            )
        with col_z2:
            zone = st.selectbox(
                "Zone",
                [
                    "Submerged",
                    "Tidal and Splash",
                    "Atmospheric",
                ],
                index=1,  # transition code default
                label_visibility="collapsed",
            )

        if zone == "Submerged":
            acs_def = (10.348, 0.714)
            ecs_def = (0.000, 0.580)
            ccr_def = (1.600, 0.200)
        elif zone == "Tidal and Splash":
            acs_def = (7.758, 1.360)
            ecs_def = (0.000, 1.105)
            ccr_def = (0.900, 0.150)
        else:
            acs_def = (6.440, 0.894)
            ecs_def = (0.000, 0.753)
            ccr_def = (0.900, 0.150)

        corr_opts = [
            "Normal",
            "Lognormal",
            "Beta",
            "Gumbel",
            "Deterministic",
        ]

        part2_config = [
            ("Acs", "A<sub>cs</sub>", "Regression variable for corrosion initiation", ">0",
             1e-6, None, acs_def[0], "Normal", acs_def[1], 0.1, "%.3f", corr_opts),
            ("ecs", "ε<sub>cs</sub>", "Regression error term for corrosion initiation", "-",
             None, None, ecs_def[0], "Normal", ecs_def[1], 0.1, "%.3f", corr_opts),
            ("Ccr", "C<sub>cr</sub>", "Critical chloride concentration", ">0",
             1e-6, None, ccr_def[0], "Normal", ccr_def[1], 0.1, "%.3f", corr_opts),
            ("D0", "D<sub>0</sub>", "Reference chloride diffusion coefficient", ">0",
             1e-6, None, 473.0, "Normal", 43.20, 1.0, "%.1f", corr_opts),
            ("kc", "k<sub>c</sub>", "Curing factor for chloride diffusion", ">0",
             1e-6, None, 0.800, "Normal", 0.100, 0.1, "%.3f", corr_opts),
            ("kt", "k<sub>t</sub>", "Test method factor for chloride diffusion", ">0",
             1e-6, None, 0.850, "Normal", 0.024, 0.01, "%.3f", corr_opts),
            ("ke", "k<sub>e</sub>", "Environmental factor for chloride diffusion", ">0",
             1e-6, None, 1.000, "Normal", 0.300, 0.1, "%.3f", corr_opts),
            ("n_val", "n", "Age factor for chloride diffusion", "0~1",
             None, None, 0.250, "Beta", 0.050, 0.01, "%.3f", corr_opts),
            ("X1", "X<sub>1</sub>", "Model uncertainty factor", ">0",
             None, None, 1.000, "Lognormal", 0.050, 0.01, "%.3f", corr_opts),
            ("lam_corr", "λ<sub>corr</sub>", "Corrosion rate adjustment coefficient", "-",
             None, None, 2.000, "Deterministic", 0.000, 0.1, "%.2f", corr_opts),
            ("R", "R", "Pitting corrosion factor", ">0",
             1e-6, None, 5.560, "Gumbel", 1.160, 0.1, "%.3f", corr_opts),
        ]

        user_corr = render_param_section(
            "",
            part2_config,
            use_std=True,
        )

        scour_opts = [
            "Normal",
            "Deterministic",
        ]
        part3_config = [
            ("SD_val", "SD (m)",
             "SD<sub>mean</sub>(t) = λ<sub>SD</sub>B{p[1-exp(-qt)] + r[1-exp(-st)]}",
             "0~8", 0.0, 8.0, None, "Normal", 0.27, 0.5, "%.3f", ["Normal"]),
            ("lambda_SD", "λ<sub>SD</sub>",
             "Scour depth adjustment coefficient", "-",
             None, None, 2.000, "Deterministic", 0.0, 0.1, "%.2f", scour_opts),
            ("B_val", "B (m)", "Base width of the pile foundation", "-",
             None, None, 2.260, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
            ("p_val", "p", "Empirical scour parameter p", "-",
             None, None, 1.093, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
            ("q_val", "q", "Empirical scour parameter q", "-",
             None, None, 0.021, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
            ("r_val", "r", "Empirical scour parameter r", "-",
             None, None, 0.269, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
            ("s_val", "s", "Empirical scour parameter s", "-",
             None, None, 2.135, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
        ]

        user_scour = render_param_section(
            "3. Scour-related parameters",
            part3_config,
            use_std=False,
        )

    with col_right:
        st.markdown(
            """
            <div style='text-align:right;color:#555;line-height:1.5;
            font-family:"Times New Roman",serif;font-size:15px;margin-top:-8px;'>
            Created by Jingcheng Wang, Associate Professor. Fuzhou University<br>
            Contact: jingchengwang@fzu.edu.cn
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Given SD / corrosion data prediction",
            type="secondary",
            use_container_width=True,
        ):
            st.session_state.page_mode = "direct"
            st.rerun()

        predict_clicked = st.button(
            "Simulate Lifecycle Probabilities and Transitions (LHS)",
            type="primary",
            use_container_width=True,
        )
        download_placeholder = st.empty()

        plot_init = st.empty()
        plot_corr = st.empty()
        plot_scour = st.empty()
        plot_failure = st.empty()
        crossover_placeholder = st.empty()

        # Two transition histograms are deliberately placed below
        # the failure-mode probability curve.
        plot_to_fff = st.empty()
        plot_to_csf = st.empty()
        transition_summary_placeholder = st.empty()
        structure_placeholder = st.empty()

        if predict_clicked:
            if assets is None:
                st.error(
                    "⚠️ model_assets_numpy.pkl was not found. "
                    "Place it in the same directory as app.py."
                )
            else:
                with st.spinner(
                    "Running synchronized LHS, deterioration paths, "
                    "annual prediction and transition-time extraction..."
                ):
                    all_inputs = (
                        user_struct
                        + user_corr
                        + user_scour
                    )

                    sd_input = next(
                        p for p in all_inputs
                        if p["id"] == "SD_val"
                    )
                    sd_cov = float(sd_input["raw_disp"])

                    result = run_lifecycle_prediction(
                        assets=assets,
                        all_inputs=all_inputs,
                        sd_cov=sd_cov,
                        n_samples=N_SAMPLES,
                        seed=RANDOM_SEED,
                    )

                    with plot_init.container():
                        st.markdown(
                            "<div class='plot-title'>"
                            "Distribution of corrosion initiation time"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.pyplot(
                            make_initiation_plot(
                                result["T_init_long"],
                                result["T_init_stir"],
                            ),
                            clear_figure=True,
                        )

                    with plot_corr.container():
                        st.markdown(
                            "<div class='plot-title'>"
                            "Time-dependent corrosion level"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.pyplot(
                            make_corrosion_plot(
                                YEARS_FULL,
                                result["corr_long"],
                                result["corr_stir"],
                            ),
                            clear_figure=True,
                        )

                    with plot_scour.container():
                        st.markdown(
                            "<div class='plot-title'>"
                            "Time-dependent scour depth"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.pyplot(
                            make_scour_plot(
                                YEARS_FULL,
                                result["scour_depths"],
                            ),
                            clear_figure=True,
                        )

                    with plot_failure.container():
                        st.markdown(
                            "<div class='plot-title'>"
                            "Time-dependent failure mode probabilities"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.pyplot(
                            make_failure_probability_plot(
                                YEARS_FULL,
                                result["annual_probs"],
                                result["label_names"],
                            ),
                            clear_figure=True,
                        )

                    # Keep the annual-probability curve crossover years
                    # directly below the failure-mode probability plot.
                    label_names = list(result["label_names"])
                    display_names = [
                        display_label(x)
                        for x in label_names
                    ]
                    curve_by_display = {
                        display_names[i]:
                            result["annual_probs"][:, i]
                        for i in range(len(label_names))
                    }

                    crossover_items = []
                    pairs_to_check = [
                        ("FFF", "CFF"),
                        ("FFF", "CSF"),
                        ("CFF", "CSF"),
                    ]

                    for label_a, label_b in pairs_to_check:
                        if (
                            label_a in curve_by_display
                            and label_b in curve_by_display
                        ):
                            found = find_all_crossovers(
                                YEARS_FULL,
                                curve_by_display[label_a],
                                curve_by_display[label_b],
                                label_a,
                                label_b,
                            )
                            crossover_items.extend(found)

                    crossover_items = sorted(
                        crossover_items,
                        key=lambda x: x[0]
                    )

                    with crossover_placeholder.container():
                        if crossover_items:
                            list_items = "".join(
                                [
                                    (
                                        "<li style='margin-bottom:2px;'>"
                                        f"{t:.2f} ({desc})"
                                        "</li>"
                                    )
                                    for t, desc
                                    in crossover_items
                                ]
                            )
                            st.markdown(
                                f"""
                                <div style='font-size:14px;color:#555;
                                line-height:1.4;margin:0 0 8px 12px;'>
                                <b>Failure-mode probability curve crossover
                                time (years):</b>
                                <ul style='margin-top:4px;
                                padding-left:20px;'>{list_items}</ul>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                """
                                <div style='font-size:14px;color:#555;
                                line-height:1.4;margin:0 0 8px 12px;'>
                                <b>Failure-mode probability curve crossover
                                time (years):</b> None
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with plot_to_fff.container():
                        st.markdown(
                            "<div class='plot-title'>"
                            "Failure-mode transition-time distribution"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.pyplot(
                            make_transition_histogram(
                                result["hist_df"],
                                "To_FFF",
                            ),
                            clear_figure=True,
                        )

                    with plot_to_csf.container():
                        st.pyplot(
                            make_transition_histogram(
                                result["hist_df"],
                                "To_CSF",
                            ),
                            clear_figure=True,
                        )

                    tdf = result["transition_df"]
                    observed_ratio = float(
                        tdf["Transition_Flag"].mean()
                    )
                    no_transition_ratio = (
                        1.0 - observed_ratio
                    )

                    with transition_summary_placeholder.container():
                        st.markdown(
                            f"""
                            <div style='font-size:14px;color:#555;
                            line-height:1.5;margin-top:2px;'>
                            <b>Observed transition by 100 years:</b>
                            {observed_ratio:.1%}<br>
                            <b>No reliable transition by 100 years:</b>
                            {no_transition_ratio:.1%}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with structure_placeholder.container():
                        st.markdown(
                            "<div class='plot-title' style='margin-top:8px;'>"
                            "Structure Schematic"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        structure_path = (
                            Path(__file__).parent
                            / "GUI_structure.png"
                        )
                        if structure_path.exists():
                            st.image(
                                str(structure_path),
                                use_container_width=True,
                            )
                        else:
                            st.markdown(
                                """
                                <div style='border:1px dashed #bbb;
                                padding:24px;text-align:center;color:#888;'>
                                GUI_structure.png not found
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # ------------------------------------------------
                    # Excel output
                    # ------------------------------------------------
                    annual_df = pd.DataFrame({
                        "Year": YEARS_FULL
                    })
                    for idx, name in enumerate(
                        result["label_names"]
                    ):
                        annual_df[
                            display_label(name)
                        ] = result["annual_probs"][:, idx]

                    annual_df[
                        "Median_Scour_Depth"
                    ] = np.median(
                        result["scour_depths"],
                        axis=0
                    )
                    annual_df[
                        "Median_Xt"
                    ] = np.median(
                        result["corr_stir"],
                        axis=0
                    )
                    annual_df[
                        "Median_Xl"
                    ] = np.median(
                        result["corr_long"],
                        axis=0
                    )

                    output = io.BytesIO()
                    with pd.ExcelWriter(
                        output,
                        engine="openpyxl"
                    ) as writer:
                        annual_df.to_excel(
                            writer,
                            sheet_name="Annual_Probabilities",
                            index=False,
                        )
                        result["transition_df"].to_excel(
                            writer,
                            sheet_name="Transition_Table",
                            index=False,
                        )
                        result["hist_df"].to_excel(
                            writer,
                            sheet_name="Transition_Histograms",
                            index=False,
                        )

                    st.session_state.lifecycle_excel_data = (
                        output.getvalue()
                    )

        if (
            "lifecycle_excel_data"
            in st.session_state
        ):
            with download_placeholder.container():
                st.download_button(
                    label="Download Results (Excel)",
                    data=st.session_state.lifecycle_excel_data,
                    file_name="Lifecycle_Assessment_Results.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    type="primary",
                    use_container_width=True,
                )



# ============================================================
# 11. Standalone entry
# ============================================================
def main():
    st.set_page_config(
        page_title=(
            "Lifecycle probabilistic seismic failure mode "
            "assessment of coastal bridge bents"
        ),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_lifecycle_app()


if __name__ == "__main__":
    main()
