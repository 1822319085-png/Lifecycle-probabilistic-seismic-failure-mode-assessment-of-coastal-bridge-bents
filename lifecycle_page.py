# -*- coding: utf-8 -*-
"""
app_v8.py
Time-varying probability assessment GUI
基于 LHS 与蒙特卡洛模拟的滨海高桩承台桥墩时变概率评估
(纯 NumPy + 极致紧凑排版 + 完美跨平台学术风兼容 + 300DPI高清图表 + 冲刷深度动态均值完美抽样)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import io
import os
from pathlib import Path
import joblib
from scipy import stats, special
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")


def render_lifecycle_app(assets=None):
    """Render lifecycle time-varying probability assessment page.

    This module is designed to be imported by app.py. Therefore st.set_page_config()
    must be called only in app.py, not here. Pass the shared model assets from app.py
    when available; otherwise this function will load model_assets_numpy.pkl as a fallback.
    """

    # ================== 1. 网页全局配置与超级 CSS 美化 ==================

    # 终极前端样式注入：强行全面接管云端浏览器的网页文本字体样式
    st.markdown("""
        <style>
        /* 全局强制所有基础文本、标签、按钮使用新罗马字体 */
        html, body, [data-testid="stAppViewContainer"], .stText, .stMarkdown, p, span, label, button {
            font-family: 'Times New Roman', serif !important;
        }
    
        .block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; max-width: 98% !important; }
        hr { margin-top: 5px; margin-bottom: 10px; border-top: 1px solid #ddd; }
    
        /* 强制所有数字输入框（Number Input）文本居中、大小一致且使用新罗马字体 */
        div[data-baseweb="input"] input { 
            text-align: center !important; 
            font-family: 'Times New Roman', serif !important; 
            font-size: 16px !important;
        }  
    
        /* 强制所有下拉菜单（Selectbox）的显示框文字使用新罗马字体 */
        div[data-baseweb="select"] div { 
            font-family: 'Times New Roman', serif !important; 
            font-size: 16px !important; 
        }
    
        /* 强制下拉菜单展开后的候选项列表使用新罗马字体 */
        ul[data-baseweb="menu"] li, [role="listbox"] li { 
            font-family: 'Times New Roman', serif !important; 
            font-size: 16px !important;
        }
    
        .section-header { color: #800020; font-size: 20px; font-weight: bold; margin-bottom: 5px; font-family: 'Times New Roman', serif;}
        .col-header { text-align: center; color: #333; font-size: 16px; font-weight: bold; font-family: 'Times New Roman', serif; }
        .plot-container { padding: 0px 0px; margin-bottom: 0px; }
        </style>
    """, unsafe_allow_html=True)

    # ====== 全局可视化美化设置 ======
    # 说明：Streamlit Cloud 是 Linux 环境，通常没有 Times New Roman。
    # st.markdown/CSS 只能控制网页文字，不能控制 st.pyplot 生成图片内部的字体。
    # 若要在云端 Matplotlib 图片中真正使用 Times New Roman，
    # 请把你有合法授权的字体文件放进 GitHub 仓库，例如：fonts/times.ttf 或 times.ttf。

    def setup_matplotlib_font():
        """跨平台加载 Times New Roman，并返回 Matplotlib 可直接使用的 FontProperties。"""
        candidate_files = [
            Path(__file__).parent / "fonts" / "times.ttf",
            Path(__file__).parent / "fonts" / "Times New Roman.ttf",
            Path(__file__).parent / "fonts" / "times new roman.ttf",
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

        # 本地 Windows/macOS 可能已经安装 Times New Roman，先尝试系统字体。
        available_font_names = {f.name for f in font_manager.fontManager.ttflist}
        if "Times New Roman" in available_font_names:
            prop = font_manager.FontProperties(family="Times New Roman")
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.serif"] = ["Times New Roman"]
            return "Times New Roman", prop

        # 云端无 Times New Roman 且仓库未提供字体时，使用 Matplotlib 自带的近似衬线字体，保证不乱码。
        # 注意：这不是 Times New Roman，只是兜底显示。
        fallback = "STIXGeneral"
        prop = font_manager.FontProperties(family=fallback)
        plt.rcParams["font.family"] = fallback
        plt.rcParams["font.serif"] = [fallback, "DejaVu Serif"]
        return fallback, prop


    GLOBAL_FONT_NAME, GLOBAL_FONT_PROP = setup_matplotlib_font()

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    # 全局大标题
    st.markdown("<h1 style='text-align: center; color: #333; font-family: \"Times New Roman\", serif; font-weight: bold; margin-bottom: 0px;'>Lifecycle probabilistic seismic failure mode assessment of coastal bridge bents</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ================== 2. 模型与缩放器加载 (纯 NumPy) ==================
    @st.cache_resource
    def load_numpy_assets():
        assets_path = 'model_assets_numpy.pkl'
        if not os.path.exists(assets_path):
            return None
        return joblib.load(assets_path)

    if assets is None:
        assets = load_numpy_assets()

    # ================== 3. 辅助计算与抽样函数 ==================
    def get_samples(u_array, dist_type, mean, std, p_min, p_max):
        _min = -np.inf if p_min is None else p_min
        _max = np.inf if p_max is None else p_max

        if dist_type == "Deterministic" or std == 0:
            s = np.full_like(u_array, mean)
        elif dist_type == "Normal":
            if _min == -np.inf and _max == np.inf:
                s = stats.norm.ppf(u_array, loc=mean, scale=std)
            else:
                a = (_min - mean) / std if std > 0 else 0
                b = (_max - mean) / std if std > 0 else 0
                s = stats.truncnorm.ppf(u_array, a, b, loc=mean, scale=std)
        elif dist_type == "Lognormal":
            sigma2 = np.log(1 + (std/mean)**2)
            mu = np.log(mean) - sigma2 / 2
            s = stats.lognorm.ppf(u_array, s=np.sqrt(sigma2), scale=np.exp(mu))
            if p_min is not None or p_max is not None:
                s = np.clip(s, _min, _max)
        elif dist_type == "Uniform":
            if p_min is not None and p_max is not None:
                lower, upper = mean - np.sqrt(3) * std, mean + np.sqrt(3) * std
                lower, upper = max(lower, _min), min(upper, _max)
                if lower >= upper: lower = upper = (lower + upper) / 2.0
                s = stats.uniform.ppf(u_array, loc=lower, scale=upper-lower)
            else:
                lower, upper = mean - np.sqrt(3) * std, mean + np.sqrt(3) * std
                s = stats.uniform.ppf(u_array, loc=lower, scale=upper-lower)
        elif dist_type == "Beta":
            var = std**2
            a = mean * (mean * (1 - mean) / var - 1)
            b = (1 - mean) * (mean * (1 - mean) / var - 1)
            s = stats.beta.ppf(u_array, a, b)
        elif dist_type == "Gumbel":
            scale = std * np.sqrt(6) / np.pi
            loc = mean - 0.5772 * scale
            s = stats.gumbel_r.ppf(u_array, loc=loc, scale=scale)
        else:
            s = np.full_like(u_array, mean)
        return s

    def find_all_crossovers(years, prob_a, prob_b, label_a, label_b):
        """查找两条概率曲线的所有交点"""
        crossovers = []
        diff = prob_a - prob_b
        for i in range(len(years)-1):
            if diff[i] == 0:
                if i > 0 and diff[i-1] != 0:
                    slope_a = prob_a[i+1] - prob_a[i]
                    slope_b = prob_b[i+1] - prob_b[i]
                    if diff[i-1] > 0 and slope_a < slope_b:
                        crossovers.append((round(years[i], 2), f"{label_a} to {label_b}"))
                    elif diff[i-1] < 0 and slope_a > slope_b:
                        crossovers.append((round(years[i], 2), f"{label_b} to {label_a}"))
            elif diff[i] * diff[i+1] < 0:
                t = years[i] - diff[i] * (years[i+1] - years[i]) / (diff[i+1] - diff[i])
                slope_a = prob_a[i+1] - prob_a[i]
                slope_b = prob_b[i+1] - prob_b[i]
                if slope_a < slope_b:
                    crossovers.append((round(t, 2), f"{label_a} to {label_b}"))
                else:
                    crossovers.append((round(t, 2), f"{label_b} to {label_a}"))
        return crossovers

    def apply_academic_style(ax_obj):
        """统一设置 Matplotlib 图内所有文字字体。
        关键点：不能只改 rcParams，也要给当前 Axes 现有对象逐个指定 FontProperties。
        """
        ax_obj.xaxis.label.set_fontproperties(GLOBAL_FONT_PROP)
        ax_obj.yaxis.label.set_fontproperties(GLOBAL_FONT_PROP)
        ax_obj.title.set_fontproperties(GLOBAL_FONT_PROP)

        ax_obj.xaxis.label.set_fontsize(12)
        ax_obj.yaxis.label.set_fontsize(12)
        ax_obj.title.set_fontsize(13)

        for label in (ax_obj.get_xticklabels() + ax_obj.get_yticklabels()):
            label.set_fontproperties(GLOBAL_FONT_PROP)
            label.set_fontsize(11)

        legend = ax_obj.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontproperties(GLOBAL_FONT_PROP)
                text.set_fontsize(10)

        ax_obj.tick_params(axis='both', direction='in', top=True, right=True, labelsize=11, width=1.0, length=4.0)


    def set_axis_labels(ax_obj, xlabel, ylabel):
        ax_obj.set_xlabel(xlabel, fontsize=12, fontproperties=GLOBAL_FONT_PROP)
        ax_obj.set_ylabel(ylabel, fontsize=12, fontproperties=GLOBAL_FONT_PROP)

    # ================== 4. 核心界面布局 ==================
    col_left, spacer, col_right = st.columns([6.8, 0.2, 3.0])

    with col_left:
        def render_param_section(title, params_config, use_std=False):
            if title:
                st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
            cols = st.columns([1.2, 2.8, 1.6, 1.1, 1.1, 1.2])
            headers = ["Parameter", "Description", "Distribution", "Mean", "St. dev." if use_std else "COV", "Range"]
            for i, h in enumerate(headers):
                cols[i].markdown(f"<div class='col-header'>{h}</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin-top: 5px; margin-bottom: 5px;'>", unsafe_allow_html=True)
        
            user_vals = []
            for p_id, html_name, desc, rng_str, p_min, p_max, p_mean, p_dist, p_disp, p_step, p_fmt, dist_opts in params_config:
                c1, c2, c3, c4, c5, c6 = st.columns([1.2, 2.8, 1.6, 1.1, 1.1, 1.2])
                c1.markdown(f"<div style='text-align: center; font-weight: bold; padding-top: 8px; font-family: \"Times New Roman\", serif;'>{html_name}</div>", unsafe_allow_html=True)
                c2.markdown(f"<div style='text-align: center; color: #444; font-size: 14px; padding-top: 8px; font-family: \"Times New Roman\", serif;'>{desc}</div>", unsafe_allow_html=True)
            
                with c3:
                    dist_val = st.selectbox(label=f"{p_id}_dist", options=dist_opts, index=dist_opts.index(p_dist), label_visibility="collapsed")
                with c4:
                    # --- 判断如果 p_mean 为 None，则禁用该输入框 ---
                    if p_mean is None:
                        mean_val = st.number_input(label=f"{p_id}_mean", value=0.0, format=p_fmt, disabled=True, label_visibility="collapsed")
                    else:
                        kwargs = {}
                        if p_min is not None: kwargs['min_value'] = float(p_min)
                        if p_max is not None: kwargs['max_value'] = float(p_max)
                        mean_val = st.number_input(label=f"{p_id}_mean", value=float(p_mean), step=float(p_step), format=p_fmt, label_visibility="collapsed", **kwargs)
                    # -----------------------------------------------------
                with c5:
                    disp_disabled = (dist_val == "Deterministic")
                    disp_val = st.number_input(label=f"{p_id}_disp", min_value=0.0, value=0.0 if disp_disabled else float(p_disp), step=0.05, format="%.3f", disabled=disp_disabled, label_visibility="collapsed")
                c6.markdown(f"<div style='text-align: center; color: #666; font-size: 15px; padding-top: 8px; font-family: \"Times New Roman\", serif;'>{rng_str}</div>", unsafe_allow_html=True)
            
                std_val = disp_val if use_std else (mean_val * disp_val)
                # 新增 raw_disp 字段，无论 mean 是否为 0 都能记录原始界面输入的离散度数值
                user_vals.append({"id": p_id, "mean": mean_val, "std": std_val, "raw_disp": disp_val, "dist": dist_val, "min": p_min, "max": p_max})
        
            st.write("")
            return user_vals

        # ---------------- 1. 结构参数 ----------------
        struct_opts = ["Normal", "Lognormal", "Uniform", "Deterministic"]
        part1_config = [
            ("N", "N", "Number of pile rows along loading direction", "2~4", 2.0, 4.0, 3.0, "Deterministic", 0.0, 1.0, "%.0f", struct_opts),
            ("Dp", "D<sub>p</sub> (m)", "Pile diameter", "0.6~1.8", 0.6, 1.8, 1.2, "Normal", 0.10, 0.1, "%.2f", struct_opts),
            ("rho_pl", "ρ<sub>pile,l</sub>", "Pile longitudinal reinforcement ratio", "0.005~0.015", 0.005, 0.015, 0.010, "Normal", 0.27, 0.001, "%.3f", struct_opts),
            ("alpha", "α", "Column axial load ratio", "0.05~0.25", 0.05, 0.25, 0.15, "Normal", 0.12, 0.01, "%.2f", struct_opts),
            ("S_Dp", "S (D<sub>p</sub>)", "Pile spacing-to-diameter ratio", "2.5~3.5", 2.5, 3.5, 3.0, "Normal", 0.15, 0.1, "%.2f", struct_opts),
            ("Dr", "D<sub>r</sub>", "Sand relative density", "0.35~0.75", 0.35, 0.75, 0.55, "Uniform", 0.21, 0.05, "%.2f", struct_opts),
            ("Hp_Dc", "H<sub>p</sub>/D<sub>c</sub>", "Column aspect ratio", "1~5", 1.0, 5.0, 3.0, "Normal", 0.26, 0.1, "%.2f", struct_opts),
            ("Dc_Dp", "D<sub>c</sub> (D<sub>p</sub>)", "Pier-to-pile diameter ratio", "1.5~3.0", 1.5, 3.0, 2.0, "Normal", 0.10, 0.1, "%.2f", struct_opts),
            ("rho_cl", "ρ<sub>column,l</sub>", "Pier longitudinal reinforcement ratio", "0.005~0.015", 0.005, 0.015, 0.010, "Normal", 0.27, 0.001, "%.3f", struct_opts),
            ("rho_ps", "ρ<sub>pile,s</sub>", "Pile transverse reinforcement ratio", "0.003~0.013", 0.003, 0.013, 0.008, "Normal", 0.42, 0.001, "%.3f", struct_opts),
            ("fyl", "f<sub>yl</sub> (MPa)", "Longitudinal reinforcement yield strength", "300~500", 300.0, 500.0, 400.0, "Lognormal", 0.106, 10.0, "%.0f", struct_opts),
            ("fc", "f<sub>c</sub> (MPa)", "Concrete compressive strength", "20~60", 20.0, 60.0, 40.0, "Lognormal", 0.20, 1.0, "%.1f", struct_opts),
            ("rho_cs", "ρ<sub>column,s</sub>", "Pier transverse reinforcement ratio", "0.003~0.013", 0.003, 0.013, 0.008, "Normal", 0.42, 0.001, "%.3f", struct_opts),
            ("t", "t (m)", "Pier cover concrete thickness", "0.04~0.08", 0.04, 0.08, 0.06, "Normal", 0.20, 0.01, "%.2f", struct_opts),
            ("d_l", "d<sub>l</sub> (m)", "Pier longitudinal reinforcement diameter", "0.018~0.032", 0.018, 0.032, 0.025, "Normal", 0.10, 0.001, "%.3f", struct_opts),
            ("fyt", "f<sub>yt</sub> (MPa)", "Transverse reinforcement yield strength", "250~450", 250.0, 450.0, 350.0, "Lognormal", 0.106, 10.0, "%.0f", struct_opts),
            ("d_t", "d<sub>s</sub> (m)", "Transverse reinforcement diameter", "0.01~0.02", 0.01, 0.02, 0.016, "Normal", 0.10, 0.001, "%.3f", struct_opts)
        ]
        user_struct = render_param_section("1. Structure/Soil-related parameters", part1_config, use_std=False)

        # ---------------- 2. 锈蚀参数 ----------------
        corr_opts = ["Normal", "Lognormal", "Beta", "Gumbel", "Uniform", "Deterministic"]
    
        st.markdown("<div class='section-header'>2. Corrosion-related parameters</div>", unsafe_allow_html=True)
    
        st.markdown("<div style='margin-top: 13px; margin-bottom: 5px;'>", unsafe_allow_html=True)
        col_f1, col_f2 = st.columns([1.4, 1.1])
        with col_f1:
            st.latex(r"t_{corr} = X_1 \left[ \frac{d_c^2}{4k_ek_tk_cD_0(t_0)^n} \left[ \text{erf}^{-1} \left( 1 - \frac{C_{cr}}{C_0} \right) \right]^{-2} \right]^{\frac{1}{1-n}}")
        with col_f2:
            st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
            st.latex(r"C_0 = A_{cs}(w/c) + \varepsilon_{cs}")
        st.markdown("</div>", unsafe_allow_html=True)

        col_z1, col_z2 = st.columns([1.5, 2])
        with col_z1:
            st.markdown("<div style='text-align: right; font-family: \"Times New Roman\", serif; font-size: 16px; font-weight: bold; padding-top: 5px; color: #333;'>Select Environmental Zone:</div>", unsafe_allow_html=True)
        with col_z2:
            zone = st.selectbox("Zone", ["Submerged", "Tidal and Splash", "Atmospheric"], label_visibility="collapsed")
    
        if zone == "Submerged":
            acs_def, ecs_def, ccr_def = (10.348, 0.714), (0.000, 0.580), (1.600, 0.200)
        elif zone == "Tidal and Splash":
            acs_def, ecs_def, ccr_def = (7.758, 1.360), (0.000, 1.105), (0.900, 0.150)
        else: 
            acs_def, ecs_def, ccr_def = (6.440, 0.894), (0.000, 0.753), (0.900, 0.150)

        part2_config = [
            ("Acs", "A<sub>cs</sub>", "Regression variable for corrosion initiation", "-", None, None, acs_def[0], "Normal", acs_def[1], 0.1, "%.3f", corr_opts),
            ("ecs", "ε<sub>cs</sub>", "Regression error term for corrosion initiation", "-", None, None, ecs_def[0], "Normal", ecs_def[1], 0.1, "%.3f", corr_opts),
            ("Ccr", "C<sub>cr</sub>", "Critical chloride concentration", "-", None, None, ccr_def[0], "Normal", ccr_def[1], 0.1, "%.3f", corr_opts),
            ("D0", "D<sub>0</sub>", "Reference chloride diffusion coefficient", "-", None, None, 473.0, "Normal", 43.20, 1.0, "%.1f", corr_opts),
            ("kc", "k<sub>c</sub>", "Curing factor for chloride diffusion", "-", None, None, 0.800, "Normal", 0.100, 0.1, "%.3f", corr_opts),
            ("kt", "k<sub>t</sub>", "Test method factor for chloride diffusion", "-", None, None, 0.850, "Normal", 0.024, 0.01, "%.3f", corr_opts),
            ("ke", "k<sub>e</sub>", "Environmental factor for chloride diffusion", "-", None, None, 1.000, "Normal", 0.300, 0.1, "%.3f", corr_opts),
            ("n_val", "n", "Age factor for chloride diffusion", "-", None, None, 0.250, "Beta", 0.050, 0.01, "%.3f", corr_opts),
            ("X1", "X<sub>1</sub>", "Model uncertainty factor", "-", None, None, 1.000, "Lognormal", 0.050, 0.01, "%.3f", corr_opts),
            ("lam", "λ", "Corrosion rate adjustment coefficient", "-", None, None, 1.000, "Deterministic", 0.000, 0.1, "%.2f", corr_opts),
            ("R", "R", "Pitting corrosion factor", "-", None, None, 5.560, "Gumbel", 1.160, 0.1, "%.3f", corr_opts)
        ]
        user_corr = render_param_section("", part2_config, use_std=True)

        # ---------------- 3. 冲刷参数 ----------------
        scour_opts = ["Normal", "Lognormal", "Uniform", "Deterministic"]
        part3_config = [
            ("SD_val", "SD (m)", "SD<sub>mean</sub> / B = p[1 - exp(-qt)] + r[1 - exp(-st)]", "0~8", 0.0, 8.0, None, "Normal", 0.27, 0.5, "%.3f", scour_opts),
            ("B_val", "B (m)", "Base width of the pile foundation", "-", None, None, 2.260, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
            ("p_val", "p", "Empirical scour parameter p", "-", None, None, 1.093, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
            ("q_val", "q", "Empirical scour parameter q", "-", None, None, 0.021, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
            ("r_val", "r", "Empirical scour parameter r", "-", None, None, 0.269, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
            ("s_val", "s", "Empirical scour parameter s", "-", None, None, 2.135, "Deterministic", 0.0, 0.01, "%.3f", scour_opts)
        ]
        # 将 use_std=False 传入，强制第三部分表头显示为 COV
        user_scour = render_param_section("3. Scour-related parameters", part3_config, use_std=False)

    # ----------------- 右侧：控制与图表区 -----------------
    with col_right:
        st.markdown("""
            <div style='text-align: right; color: #555; line-height: 1.5; font-family: "Times New Roman", serif; font-size: 15px; margin-top: -8px;'>
                Created by Jingcheng Wang, Associate Professor. Fuzhou University<br>Contact: jingchengwang@fzu.edu.cn
            </div>
        """, unsafe_allow_html=True)
    
        predict_clicked = st.button("Simulate Lifecycle Probabilities (LHS)", type="primary", use_container_width=True)
        direct_clicked = st.button("Direct Prediction Using Existing Scour and Corrosion Data", key="btn_direct_prediction", use_container_width=True)
        if direct_clicked:
            st.session_state.page_mode = "direct"
            st.rerun()

        download_placeholder = st.empty()

        plot_placeholders = [st.empty() for _ in range(4)]
        crossover_placeholder = st.empty()
    
        if predict_clicked:
            if assets is None:
                st.error("⚠️ 未检测到 `model_assets_numpy.pkl`。请确保权重文件与脚本处于同一目录。")
            else:
                with st.spinner('Running Latin Hypercube Sampling and Physics Engines...'):
                    weights = assets['weights']
                    scaler = assets['scaler']
                    label_names = assets['le'].classes_ if 'le' in assets else ['FFF', 'PFF', 'PSF']

                    N_SAMPLES = 2000
                    N_YEARS = 101
                
                    all_inputs = user_struct + user_corr + user_scour
                    NUM_DIM = len(all_inputs)

                    sampler = qmc.LatinHypercube(d=NUM_DIM, seed=42)
                    U = sampler.random(n=N_SAMPLES)
                
                    samples_dict = {}
                    for idx, p in enumerate(all_inputs):
                        samples_dict[p['id']] = get_samples(U[:, idx], p['dist'], p['mean'], p['std'], p['min'], p['max'])

                    t_samples = samples_dict['t']
                    dl_samples = samples_dict['d_l']
                    ds_samples = samples_dict['d_t']

                    Acs = samples_dict['Acs']
                    ecs = samples_dict['ecs']
                    Ccr = samples_dict['Ccr']
                    w_b = 0.5
                    C0 = Acs * w_b + ecs
                
                    def compute_T_init(cover_mm, D0, k_e, k_t, k_c, n, C_cr, C0_arr, X1):
                        valid_mask = (C0_arr > C_cr) & (C0_arr > 0) & (C_cr > 0)
                        safe_C0 = np.where(valid_mask, C0_arr, 2.0)
                        safe_Ccr = np.where(valid_mask, C_cr, 1.0)
                        ratio = np.clip(safe_Ccr / safe_C0, 1e-12, 1 - 1e-12) 
                        inv_erf = special.erfinv(1 - ratio)
                        denom = 4 * k_e * k_t * k_c * D0 * (1.0**n) * (inv_erf**2)
                        valid_denom = denom > 0
                        safe_denom = np.where(valid_denom, denom, 1.0)
                        base_val = np.maximum(cover_mm**2 / safe_denom, 1e-12) 
                        T = X1 * (base_val) ** (1 / np.maximum(1 - n, 1e-6))
                        return np.where(valid_mask & valid_denom, T, np.inf)

                    tc_mm = t_samples * 1000.0
                    cover_stir_mm = np.maximum(tc_mm - ds_samples * 1000.0, 15.0)

                    T_init_long = compute_T_init(tc_mm, samples_dict['D0'], samples_dict['ke'], samples_dict['kt'], samples_dict['kc'], samples_dict['n_val'], Ccr, C0, samples_dict['X1'])
                    T_init_stir = compute_T_init(cover_stir_mm, samples_dict['D0'], samples_dict['ke'], samples_dict['kt'], samples_dict['kc'], samples_dict['n_val'], Ccr, C0, samples_dict['X1'])

                    def pitting_corrosion_matrix(d_rein_mm, T_init_arr, cover_mm, years_arr, R_arr, lambda_arr):
                        i_corr0 = 1.0 * (37.8 * lambda_arr) * (1 - w_b) ** (-1.64) / cover_mm
                        A0 = np.pi * d_rein_mm**2 / 4.0
                        n_s = len(d_rein_mm)
                        n_y = len(years_arr)
                    
                        corr_rate = np.zeros((n_s, n_y))
                        for y_idx, yr in enumerate(years_arr):
                            t_p = np.where(np.isinf(T_init_arr), 0.0, np.maximum(0.0, yr - T_init_arr))
                            diam_loss_uniform = 2 * (0.0116 * 0.85 * i_corr0 * (t_p ** 0.71) / 0.71)
                            d_rem_uniform = np.maximum(0, d_rein_mm - diam_loss_uniform)
                            Au = np.pi * d_rem_uniform**2 / 4.0
                        
                            p_val = R_arr * (diam_loss_uniform / 2.0)
                            A_rem = np.copy(A0)
                            mask_p = p_val > 0
                        
                            p_v = np.minimum(p_val[mask_p], d_rein_mm[mask_p])
                            dr = d_rein_mm[mask_p]
                            inner_val = np.maximum(0, 1 - (p_v/dr)**2)
                            a = 2 * p_v * np.sqrt(inner_val)
                        
                            theta1 = 2 * np.arcsin(np.clip(a / dr, -1.0, 1.0))
                            A1 = 0.5 * (theta1 * (0.5*dr)**2 - a * (0.5*dr - p_v**2/dr))
                        
                            theta2 = np.where(p_v > 1e-12, 2 * np.arcsin(np.clip(a / (2*p_v), -1.0, 1.0)), 0.0)
                            A2 = 0.5 * (theta2 * p_v**2 - a * p_v**2 / dr)
                        
                            ADP = np.where(p_v <= dr / np.sqrt(2), A0[mask_p] - A1 - A2, A1 - A2)
                            Ap = (1 - a/(2*dr)) * (Au[mask_p] - A0[mask_p]) + ADP
                        
                            A_rem[mask_p] = np.clip(Ap, 0.0, A0[mask_p])
                            corr_rate[:, y_idx] = np.clip(1 - A_rem / A0, 0.0, 1.0)
                        return corr_rate

                    years_arr = np.arange(0, N_YEARS)
                    corr_long = pitting_corrosion_matrix(dl_samples*1000.0, T_init_long, tc_mm, years_arr, samples_dict['R'], samples_dict['lam'])
                    corr_stir = pitting_corrosion_matrix(ds_samples*1000.0, T_init_stir, cover_stir_mm, years_arr, samples_dict['R'], samples_dict['lam'])

                    B_arr = samples_dict['B_val']
                    p_arr, q_arr, r_arr, s_arr = samples_dict['p_val'], samples_dict['q_val'], samples_dict['r_val'], samples_dict['s_val']
                
                    # ================== 冲刷深度动态均值与分布抽样完全重构 ==================
                    sd_idx = [p['id'] for p in all_inputs].index('SD_val')
                    U_SD = U[:, sd_idx]  # 获取独立生成的 LHS 均匀抽样矩阵 (用于保证 SD_val 自身的抽样正交性)
                
                    sd_input = next(p for p in all_inputs if p['id'] == 'SD_val')
                    # 直接提取界面输入的原始值(0.27)作为 COV，避免因 mean 禁用(值为 0)导致 std 被清零
                    sd_std_val = sd_input['raw_disp']  
                
                    sd_min = 0.0 if sd_input['min'] is None else sd_input['min']
                    sd_max = 8.0 if sd_input['max'] is None else sd_input['max']
                    sd_dist_type = sd_input['dist']
                
                    scour_depths = np.zeros((N_SAMPLES, N_YEARS))
                
                    for y_idx, yr in enumerate(years_arr):
                        # 1. 结合时变公式，计算当年的均值
                        term1 = p_arr * (1 - np.exp(-q_arr * yr))
                        term2 = r_arr * (1 - np.exp(-s_arr * yr))
                        sd_mean = B_arr * (term1 + term2)
                    
                        # 2. 动态扩展标准差：均值越大，标准差也同比例放大 (即 COV 恒定机制)
                        dynamic_std = sd_mean * sd_std_val
                    
                        # 3. 动态抽样 (使用数组级 truncnorm，完美对应每一条样本)
                        if sd_dist_type == "Deterministic" or sd_std_val == 0:
                            sd_samples = sd_mean
                        elif sd_dist_type == "Normal":
                            safe_std = np.maximum(dynamic_std, 1e-6)
                            a = (sd_min - sd_mean) / safe_std
                            b = (sd_max - sd_mean) / safe_std
                            sd_samples = stats.truncnorm.ppf(U_SD, a, b, loc=sd_mean, scale=safe_std)
                        elif sd_dist_type == "Lognormal":
                            safe_std = np.maximum(dynamic_std, 1e-6)
                            safe_mean = np.maximum(sd_mean, 1e-6)
                            sigma2 = np.log(1 + (safe_std/safe_mean)**2)
                            mu = np.log(safe_mean) - sigma2 / 2
                            sd_samples = stats.lognorm.ppf(U_SD, s=np.sqrt(sigma2), scale=np.exp(mu))
                            sd_samples = np.clip(sd_samples, sd_min, sd_max)
                        elif sd_dist_type == "Uniform":
                            safe_std = np.maximum(dynamic_std, 1e-6)
                            lower = np.maximum(sd_mean - np.sqrt(3) * safe_std, sd_min)
                            upper = np.minimum(sd_mean + np.sqrt(3) * safe_std, sd_max)
                            invalid = lower >= upper
                            lower[invalid] = sd_mean[invalid]
                            upper[invalid] = sd_mean[invalid] + 1e-6
                            sd_samples = stats.uniform.ppf(U_SD, loc=lower, scale=upper-lower)
                        else:
                            sd_samples = sd_mean
                        
                        scour_depths[:, y_idx] = np.clip(sd_samples, sd_min, sd_max)
                    # ====================================================================

                    X_fixed = np.zeros((N_SAMPLES, 20)) 
                    mapping = [
                        ('N', 0), ('Dp', 1), ('rho_pl', 2), ('alpha', 3), ('S_Dp', 4), ('Dr', 5),
                        ('Hp_Dc', 7), ('Dc_Dp', 8), ('rho_cl', 9), ('rho_ps', 10), ('fyl', 11), 
                        ('fc', 12), ('rho_cs', 13), ('t', 16), ('d_l', 17), ('fyt', 18), ('d_t', 19)
                    ]
                    for key, col_idx in mapping:
                        X_fixed[:, col_idx] = samples_dict[key]

                    annual_probs = {name: np.zeros(N_YEARS) for name in label_names}
                
                    for year in range(N_YEARS):
                        X_fixed[:, 6]  = scour_depths[:, year]     
                        X_fixed[:, 14] = corr_stir[:, year]        
                        X_fixed[:, 15] = corr_long[:, year]        
                    
                        a_layer = scaler.transform(X_fixed)
                        for w, b in weights[:-1]:
                            z = np.dot(a_layer, w) + b
                            a_layer = np.maximum(0, z)
                        w_out, b_out = weights[-1]
                        z_out = np.dot(a_layer, w_out) + b_out
                    
                        predictions = np.argmax(z_out, axis=1)
                        for idx, name in enumerate(label_names):
                            annual_probs[name][year] = np.sum(predictions == idx) / N_SAMPLES

                    # ================== 5. 图表渲染 ==================
                    color_hist_s = '#CBE5F5'  
                    color_line_s = '#0000FF'  
                    color_hist_l = '#FADBDC'  
                    color_line_l = '#FF0000'  
                    color_scour  = '#00796B'

                    # ---------------- 图 1: Distribution of corrosion initiation time ----------------
                    with plot_placeholders[0].container():
                        st.markdown("<div class='plot-container'><div style='text-align: center; font-family: \"Times New Roman\", serif; font-weight: bold; font-size: 17px; margin-bottom: 2px;'>Distribution of corrosion initiation time</div>", unsafe_allow_html=True)
                        fig1, ax1 = plt.subplots(figsize=(6, 3.5), dpi=300)
                        t_long_valid = T_init_long[T_init_long <= 100]
                        t_stir_valid = T_init_stir[T_init_stir <= 100]
                    
                        ax1.hist(t_stir_valid, bins=80, rwidth=1.0, density=True, alpha=0.8, color=color_hist_s, edgecolor='gray', linewidth=0.5, label='Transverse frequency')
                        ax1.hist(t_long_valid, bins=80, rwidth=1.0, density=True, alpha=0.6, color=color_hist_l, edgecolor='gray', linewidth=0.5, label='Longitudinal frequency')
                    
                        if len(t_stir_valid[t_stir_valid > 0]) > 5:
                            s, l, sc = stats.lognorm.fit(t_stir_valid[t_stir_valid > 0], floc=0)
                            x = np.linspace(0, 100, 1000)
                            ax1.plot(x, stats.lognorm.pdf(x, s, loc=0, scale=sc), color=color_line_s, lw=2.5, label='Transverse lognormal distribution')
                    
                        if len(t_long_valid[t_long_valid > 0]) > 5:
                            s, l, sc = stats.lognorm.fit(t_long_valid[t_long_valid > 0], floc=0)
                            x = np.linspace(0, 100, 1000)
                            ax1.plot(x, stats.lognorm.pdf(x, s, loc=0, scale=sc), color=color_line_l, lw=2.5, label='Longitudinal lognormal distribution')

                        set_axis_labels(ax1, 'Initial corrosion time (years)', 'Probability density')
                        ax1.set_xlim(0, 30)
                    
                        y_max1 = ax1.get_ylim()[1]
                        rounded_ymax1 = np.ceil(y_max1 * 10) / 10 if y_max1 > 0 else 0.1
                        ax1.set_ylim(0, rounded_ymax1)
                        ax1.set_yticks(np.linspace(0, rounded_ymax1, 5))
                    
                        ax1.legend(frameon=False, loc='upper right', prop=font_manager.FontProperties(fname=GLOBAL_FONT_PROP.get_file(), size=10) if GLOBAL_FONT_PROP.get_file() else font_manager.FontProperties(family=GLOBAL_FONT_NAME, size=10))
                        apply_academic_style(ax1)
                        plt.tight_layout(pad=0.3)
                        st.pyplot(fig1, clear_figure=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # ---------------- 图 2: Time-dependent corrosion rate ----------------
                    with plot_placeholders[1].container():
                        st.markdown("<div class='plot-container'><div style='text-align: center; font-family: \"Times New Roman\", serif; font-weight: bold; font-size: 17px; margin-bottom: 2px;'>Time-dependent corrosion level</div>", unsafe_allow_html=True)
                        fig2, ax2 = plt.subplots(figsize=(6, 3.5), dpi=300)
                        med_l, p16_l, p84_l = np.median(corr_long, axis=0), np.percentile(corr_long, 16, axis=0), np.percentile(corr_long, 84, axis=0)
                        med_s, p16_s, p84_s = np.median(corr_stir, axis=0), np.percentile(corr_stir, 16, axis=0), np.percentile(corr_stir, 84, axis=0)
                    
                        ax2.plot(years_arr, med_s, color=color_line_s, lw=2.5, label='Transverse (median)', zorder=3)
                        ax2.fill_between(years_arr, p16_s, p84_s, color=color_hist_s, alpha=0.6, label='Transverse (16%-84% quantiles)', zorder=2)
                    
                        ax2.plot(years_arr, med_l, color=color_line_l, lw=2.5, label='Longitudinal (median)', zorder=3)
                        ax2.fill_between(years_arr, p16_l, p84_l, color=color_hist_l, alpha=0.6, label='Longitudinal (16%-84% quantiles)', zorder=2)
                    
                        set_axis_labels(ax2, 'Service time (years)', 'Corrosion level')
                        ax2.set_xlim(0, 100)
                    
                        max_corr = np.max([np.max(p84_l), np.max(p84_s)])
                        rounded_ymax2 = np.ceil(max_corr * 10) / 10 if max_corr > 0 else 0.1
                        ax2.set_ylim(0, rounded_ymax2)
                        ax2.set_yticks(np.linspace(0, rounded_ymax2, 5))
                    
                        ax2.legend(frameon=False, loc='upper left', prop=font_manager.FontProperties(fname=GLOBAL_FONT_PROP.get_file(), size=10) if GLOBAL_FONT_PROP.get_file() else font_manager.FontProperties(family=GLOBAL_FONT_NAME, size=10))
                        apply_academic_style(ax2)
                        plt.tight_layout(pad=0.3)
                        st.pyplot(fig2, clear_figure=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # ---------------- 图 3: Scour depth evolution ----------------
                    with plot_placeholders[2].container():
                        st.markdown("<div class='plot-container'><div style='text-align: center; font-family: \"Times New Roman\", serif; font-weight: bold; font-size: 17px; margin-bottom: 2px;'>Time-dependent scour depth</div>", unsafe_allow_html=True)
                        fig3, ax3 = plt.subplots(figsize=(6, 3.5), dpi=300)
                    
                        # 使用已经包含完整不确定性的 scour_depths
                        med_sd = np.median(scour_depths, axis=0)
                        p16_sd = np.percentile(scour_depths, 16, axis=0)
                        p84_sd = np.percentile(scour_depths, 84, axis=0)
                    
                        ax3.plot(years_arr, med_sd, color=color_scour, lw=2.5, label='Scour depth (median)', zorder=3)
                        ax3.fill_between(years_arr, p16_sd, p84_sd, color='#B2DFDB', alpha=0.6, label='Scour depth (16%-84% quantiles)', zorder=2)
                    
                        set_axis_labels(ax3, 'Service time (years)', 'Scour depth (m)')
                        ax3.set_xlim(0, 100)
                    
                        max_scour = np.max(p84_sd)
                        rounded_ymax3 = np.ceil(max_scour) if max_scour > 0 else 1.0
                        ax3.set_ylim(0, rounded_ymax3)
                        ax3.set_yticks(np.linspace(0, rounded_ymax3, 5))
                    
                        ax3.legend(frameon=False, loc='upper left', prop=font_manager.FontProperties(fname=GLOBAL_FONT_PROP.get_file(), size=10) if GLOBAL_FONT_PROP.get_file() else font_manager.FontProperties(family=GLOBAL_FONT_NAME, size=10))
                        apply_academic_style(ax3)
                        plt.tight_layout(pad=0.3)
                        st.pyplot(fig3, clear_figure=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # ---------------- 图 4: Time-dependent Failure Mode Probabilities ----------------
                    with plot_placeholders[3].container():
                        st.markdown("<div class='plot-container'><div style='text-align: center; font-family: \"Times New Roman\", serif; font-weight: bold; font-size: 17px; margin-bottom: 2px;'>Time-dependent failure mode probabilities</div>", unsafe_allow_html=True)
                        fig4, ax4 = plt.subplots(figsize=(6, 3.5), dpi=300)
                    
                        color_map_prob = {'FFF': color_scour, 'PFF': color_line_s, 'PSF': color_line_l}
                        for name in label_names:
                            ax4.plot(years_arr, annual_probs[name], color=color_map_prob.get(name, '#333'), lw=2.5, label=name)
                    
                        set_axis_labels(ax4, 'Service time (years)', 'Probability')
                        ax4.set_xlim(0, 100)
                    
                        max_prob = np.max([np.max(annual_probs[name]) for name in label_names])
                        rounded_ymax4 = np.ceil(max_prob * 10) / 10 if max_prob > 0 else 0.1
                        ax4.set_ylim(0, 1.0)
                        ax4.set_yticks(np.linspace(0, 1.0, 5))
                    
                        ax4.legend(frameon=False, loc='upper right', prop=font_manager.FontProperties(fname=GLOBAL_FONT_PROP.get_file(), size=10) if GLOBAL_FONT_PROP.get_file() else font_manager.FontProperties(family=GLOBAL_FONT_NAME, size=10))
                        apply_academic_style(ax4)
                        plt.tight_layout(pad=0.3)
                        st.pyplot(fig4, clear_figure=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # ---------------- 多交点全面计算输出 ----------------
                    crossovers_found = []
                    pairs_to_check = [('FFF', 'PFF'), ('FFF', 'PSF'), ('PFF', 'PSF')]
                    for label_a, label_b in pairs_to_check:
                        if label_a in label_names and label_b in label_names:
                            cross_list = find_all_crossovers(years_arr, annual_probs[label_a], annual_probs[label_b], label_a, label_b)
                            for t, desc in cross_list:
                                crossovers_found.append({"time": float(t), "text": f"{t} ({desc})"})

                    crossovers_found = sorted(crossovers_found, key=lambda x: x["time"])

                    if crossovers_found:
                        list_items = "".join([f"<li style='margin-bottom: 2px;'>{item['text']}</li>" for item in crossovers_found])
                        crossover_html = f"""
                        <div style='font-family: "Times New Roman", serif; font-size: 15px; color: #444; margin: 0px 0px 10px 20px;'>
                            <b>Time to transfer of seismic failure mode (years):</b>
                            <ul style='margin-top: 5px; padding-left: 20px;'>
                                {list_items}
                            </ul>
                        </div>
                        """
                    else:
                        crossover_html = """
                        <div style='font-family: "Times New Roman", serif; font-size: 15px; color: #444; margin: 0px 0px 10px 20px;'>
                            <b>Time to transfer of seismic failure mode (years):</b> None
                        </div>
                        """
                
                    with crossover_placeholder.container():
                        st.markdown(crossover_html, unsafe_allow_html=True)

                    # Excel 导出
                    prob_df = pd.DataFrame({'Year': years_arr})
                    for name in label_names:
                        prob_df[name] = annual_probs[name]
                    prob_df['Median_Scour_Depth'] = med_sd
                
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        prob_df.to_excel(writer, sheet_name='Annual_Probabilities', index=False)
                
                    st.session_state.excel_data = output.getvalue()

        if 'excel_data' in st.session_state:
            with download_placeholder.container():
                st.download_button(
                    label="Download Results (Excel)",
                    data=st.session_state.excel_data,
                    file_name="Assessment_Results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )

        st.write("---")
        st.markdown("<div style='text-align: center; font-family: \"Times New Roman\", serif; font-weight: bold; font-size: 17px; margin-bottom: 5px;'>Structure Schematic</div>", unsafe_allow_html=True)
        try:
            st.image("structure.png", use_container_width=True)
        except:
            st.markdown("""
            <div style='border: 1px dashed #ccc; padding: 20px; text-align: center; color: #999; font-family: \"Times New Roman\", serif;'>
                [Structure Schematic Placeholder]<br>(structure.png not found)
            </div>
            """, unsafe_allow_html=True)
