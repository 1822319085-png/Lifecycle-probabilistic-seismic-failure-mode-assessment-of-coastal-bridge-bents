# -*- coding: utf-8 -*-
"""
app_direct_modified.py

已有冲刷深度与钢筋腐蚀率数据的三类地震破坏模式概率预测 GUI
- 直接输入冲刷深度 SD、箍筋腐蚀率 Xt、纵筋腐蚀率 Xl 及其分布
- 箍筋腐蚀率 Xt 与纵筋腐蚀率 Xl 均按机器学习训练范围 0~1.0 设置
- 新增箍筋直径 d_t 作为第 20 个输入参数
- 纯 NumPy 前向传播 + Monte Carlo 抽样 + Excel 导出

说明：
1. 单独运行本文件时，会自动调用 st.set_page_config。
2. 与生命周期 GUI 合并时，建议只导入 render_direct_prediction_app()，
   并在主 app.py 中统一调用一次 st.set_page_config()。
"""

import io
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


# ================== 1. 模型加载 ==================
@st.cache_resource
def load_numpy_model(model_path: str = "model_assets_numpy.pkl"):
    """加载纯 NumPy 模型权重、缩放器和标签编码器。"""
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


# ================== 2. CSS 样式 ==================
def inject_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 95% !important;
        }
        div[data-testid="column"] {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        hr {
            margin-top: 5px;
            margin-bottom: 15px;
        }
        div[data-baseweb="input"] input {
            text-align: center !important;
            font-family: 'Times New Roman', serif !important;
            font-size: 18px !important;
        }
        div[data-baseweb="select"] div {
            font-family: 'Times New Roman', serif !important;
            font-size: 18px !important;
        }
        /* 强制下拉菜单展开后的候选项列表使用新罗马字体 */
        ul[data-baseweb="menu"] li, [role="listbox"] li { 
            font-family: 'Times New Roman', serif !important; 
            font-size: 18px !important; /* 👈 修改这里改变下拉菜单展开后的大小 */
        }
        .param-header {
            text-align: center;
            color: #800020;
            font-size: 20px;
            font-weight: bold;
            font-family: Arial, sans-serif;
        }
        .param-symbol {
            text-align: center;
            color: #4a235a;
            font-weight: bold;
            font-family: Arial, sans-serif;
            padding-top: 8px;
        }
        .param-desc {
            text-align: center;
            color: #444444;
            font-size: 16px;
            padding-top: 8px;
            padding-left: 10px;
            font-family: Arial, sans-serif;
        }
        .param-range {
            text-align: center;
            color: #666666;
            font-size: 16px;
            padding-top: 8px;
            font-family: Arial, sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ================== 3. 抽样函数 ==================
def sample_parameter(mean, cov, dist, p_min, p_max, n_samples):
    """
    根据均值、COV、分布类型和上下限进行 Monte Carlo 抽样。
    注意：这里采用截断/裁剪到训练范围，避免输入超出机器学习模型训练域。
    """
    mean = float(mean)
    cov = float(cov)
    p_min = float(p_min)
    p_max = float(p_max)

    if dist == "Deterministic" or cov == 0:
        samples = np.full(n_samples, mean, dtype=float)

    else:
        std = abs(mean * cov)

        if std <= 1e-12:
            samples = np.full(n_samples, mean, dtype=float)

        elif dist == "Normal":
            samples = np.random.normal(mean, std, n_samples)

        elif dist == "Lognormal":
            # 对正值变量使用对数正态；若 mean <= 0，则退化为确定值再裁剪
            if mean <= 0:
                samples = np.full(n_samples, mean, dtype=float)
            else:
                sigma2 = np.log(1.0 + (std / mean) ** 2)
                mu = np.log(mean) - sigma2 / 2.0
                samples = np.random.lognormal(mu, np.sqrt(sigma2), n_samples)

        elif dist == "Uniform":
            lower = mean - np.sqrt(3.0) * std
            upper = mean + np.sqrt(3.0) * std
            if lower >= upper:
                samples = np.full(n_samples, mean, dtype=float)
            else:
                samples = np.random.uniform(lower, upper, n_samples)

        else:
            samples = np.full(n_samples, mean, dtype=float)

    return np.clip(samples, p_min, p_max)


# ================== 4. 纯 NumPy 前向传播 ==================
def predict_with_numpy_network(samples, assets):
    """使用保存的 scaler 和 NumPy 权重进行前向传播预测。"""
    weights = assets["weights"]
    scaler = assets["scaler"]

    a = scaler.transform(samples)
    for w, b in weights[:-1]:
        z = np.dot(a, w) + b
        a = np.maximum(0.0, z)

    w_out, b_out = weights[-1]
    z_out = np.dot(a, w_out) + b_out
    predictions = np.argmax(z_out, axis=1)
    return predictions


# ================== 5. 进度条绘图 ==================
def draw_progress_bar(label, percentage, color_hex):
    percentage = max(0.0, min(100.0, float(percentage)))
    text_color = "white" if percentage > 50 else "#333333"
    html = f"""
    <div style="margin-bottom: 15px;">
        <div style="font-weight: bold; font-family: Arial, sans-serif; font-size: 14px; margin-bottom: 5px;">{label}</div>
        <div style="width: 100%; background-color: #f0f0f0; border-radius: 4px; border: 1px solid #ccc; position: relative; height: 28px;">
            <div style="width: {percentage}%; background-color: {color_hex}; height: 100%; border-radius: 3px; transition: width 0.6s ease-in-out;"></div>
            <div style="position: absolute; width: 100%; text-align: center; top: 0; left: 0; line-height: 28px; font-weight: bold; font-family: Arial, sans-serif; font-size: 13px; color: {text_color};">{percentage:.2f}%</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ================== 6. 直接预测 GUI 主函数 ==================
def render_direct_prediction_app(assets=None, show_back_button=False):
    """
    直接输入已有冲刷与腐蚀数据的破坏模式概率预测界面。

    Parameters
    ----------
    assets : dict or None
        已加载的模型资源。如果为 None，则自动加载 model_assets_numpy.pkl。
    show_back_button : bool
        与生命周期 GUI 合并时可设为 True，在右侧显示返回按钮。
    """
    inject_css()

    if assets is None:
        assets = load_numpy_model()

    # 初始化状态，避免不同 GUI 合并时 session_state 键冲突
    if "direct_mc_probs" not in st.session_state:
        st.session_state.direct_mc_probs = [0.0, 0.0, 0.0]
    if "direct_excel_data" not in st.session_state:
        st.session_state.direct_excel_data = None

    st.markdown(
        "<h3 style='text-align: center; color: #333; font-family: Arial, sans-serif; margin-bottom: 10px;'>"
        "Seismic Failure Mode Probability Assessment with Given Scour and Corrosion Data"
        "</h3>",
        unsafe_allow_html=True,
    )

    col_left, spacer, col_right = st.columns([6.8, 0.2, 3.0])

    # ----------------- 左侧：20 个参数输入 -----------------
    with col_left:
        st.markdown("<h4 style='color: #333;'>Structure/Soil-related parameters</h4>", unsafe_allow_html=True)

        cols = st.columns([1.0, 2.5, 1.6, 1.2, 1.2, 1.5])
        headers = ["Parameter", "Description", "Distribution", "Mean", "COV", "Range"]
        for i, header_name in enumerate(headers):
            cols[i].markdown(f"<div class='param-header'>{header_name}</div>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # 参数顺序必须与 20 维训练模型一致：
        # 0 N, 1 Dp, 2 rho_pl, 3 alpha, 4 S_Dp, 5 Dr, 6 SD,
        # 7 Hp_Dc, 8 Dc_Dp, 9 rho_cl, 10 rho_ps, 11 fyl, 12 fc,
        # 13 rho_cs, 14 Xt, 15 Xl, 16 t, 17 d_l, 18 fyt, 19 d_t
        params_config = [
            ("N", "N", "Number of pile rows along the loading direction", "2~4", 2.0, 4.0, 3.0, "Deterministic", 0.00, 1.0, "%.0f"),
            ("Dp", "D<sub>p</sub> (m)", "Pile diameter", "0.6~1.8", 0.6, 1.8, 1.2, "Normal", 0.10, 0.1, "%.2f"),
            ("rho_pl", "ρ<sub>pile,l</sub>", "Pile longitudinal reinforcement ratio", "0.005~0.015", 0.005, 0.015, 0.010, "Normal", 0.27, 0.001, "%.3f"),
            ("alpha", "α", "Column axial load ratio", "0.05~0.25", 0.05, 0.25, 0.15, "Normal", 0.12, 0.01, "%.2f"),
            ("S_Dp", "S (D<sub>p</sub>)", "Pile spacing-to-diameter ratio", "2.5~3.5", 2.5, 3.5, 3.0, "Normal", 0.15, 0.1, "%.2f"),
            ("Dr", "D<sub>r</sub>", "Sand relative density", "0.35~0.75", 0.35, 0.75, 0.55, "Uniform", 0.21, 0.05, "%.2f"),
            ("SD", "SD (m)", "Scour depth", "0~8", 0.0, 8.0, 4.0, "Normal", 0.27, 0.5, "%.2f"),
            ("Hp_Dc", "H<sub>p</sub>/D<sub>c</sub>", "Column aspect ratio", "1~5", 1.0, 5.0, 3.0, "Normal", 0.26, 0.1, "%.2f"),
            ("Dc_Dp", "D<sub>c</sub> (D<sub>p</sub>)", "Pier-to-pile diameter ratio", "1.5~3.0", 1.5, 3.0, 2.0, "Normal", 0.10, 0.1, "%.2f"),
            ("rho_cl", "ρ<sub>column,l</sub>", "Pier longitudinal reinforcement ratio", "0.005~0.015", 0.005, 0.015, 0.010, "Normal", 0.27, 0.001, "%.3f"),
            ("rho_ps", "ρ<sub>pile,s</sub>", "Pile transverse reinforcement ratio", "0.003~0.013", 0.003, 0.013, 0.008, "Normal", 0.42, 0.001, "%.3f"),
            ("fyl", "f<sub>yl</sub> (MPa)", "Longitudinal rebar yield strength", "300~500", 300.0, 500.0, 400.0, "Lognormal", 0.106, 10.0, "%.0f"),
            ("fc", "f<sub>c</sub> (MPa)", "Concrete compressive strength", "20~60", 20.0, 60.0, 40.0, "Lognormal", 0.20, 1.0, "%.1f"),
            ("rho_cs", "ρ<sub>column,s</sub>", "Pier transverse reinforcement ratio", "0.003~0.013", 0.003, 0.013, 0.008, "Normal", 0.42, 0.001, "%.3f"),

            # ===== 腐蚀参数：按机器学习训练范围 0~1.0 设置 =====
            # Xt 不再表示 Xt/Xl，而是箍筋/横向钢筋腐蚀率本身
            ("Xt", "X<sub>t</sub>", "Corrosion level of transverse reinforcement", "0~1.00", 0.0, 1.0, 0.30, "Normal", 0.29, 0.01, "%.2f"),
            ("Xl", "X<sub>l</sub>", "Corrosion level of longitudinal reinforcement", "0~1.00", 0.0, 1.0, 0.15, "Normal", 0.20, 0.01, "%.2f"),

            ("t", "t (m)", "Pier cover concrete thickness", "0.04~0.08", 0.04, 0.08, 0.06, "Normal", 0.20, 0.01, "%.2f"),
            ("d_l", "d<sub>l</sub> (m)", "Pier longitudinal reinforcement diameter", "0.018~0.032", 0.018, 0.032, 0.025, "Normal", 0.10, 0.001, "%.3f"),
            ("fyt", "f<sub>yt</sub> (MPa)", "Transverse rebar yield strength", "250~450", 250.0, 450.0, 350.0, "Lognormal", 0.106, 10.0, "%.0f"),
            ("d_t", "d<sub>t</sub> (m)", "Transverse reinforcement diameter", "0.010~0.020", 0.010, 0.020, 0.016, "Normal", 0.10, 0.001, "%.3f"),
        ]

        dist_options = ["Normal", "Lognormal", "Uniform", "Deterministic"]
        user_inputs = []

        for p_id, html_name, desc, rng, p_min, p_max, p_mean, p_dist, p_cov, p_step, p_format in params_config:
            c1, c2, c3, c4, c5, c6 = st.columns([1.0, 2.5, 1.6, 1.2, 1.2, 1.5])

            c1.markdown(f"<div class='param-symbol' style='font-size: 16px;'>{html_name}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='param-desc' style='font-size: 16px;'>{desc}</div>", unsafe_allow_html=True)

            with c3:
                dist_val = st.selectbox(
                    label=f"direct_{p_id}_dist",
                    options=dist_options,
                    index=dist_options.index(p_dist),
                    label_visibility="collapsed",
                )

            with c4:
                mean_val = st.number_input(
                    label=f"direct_{p_id}_mean",
                    min_value=float(p_min),
                    max_value=float(p_max),
                    value=float(p_mean),
                    step=float(p_step),
                    format=p_format,
                    label_visibility="collapsed",
                )

            with c5:
                cov_disabled = dist_val == "Deterministic"
                cov_val = st.number_input(
                    label=f"direct_{p_id}_cov",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.0 if cov_disabled else float(p_cov),
                    step=0.05,
                    format="%.2f",
                    disabled=cov_disabled,
                    label_visibility="collapsed",
                )

            c6.markdown(f"<div class='param-range'>{rng}</div>", unsafe_allow_html=True)

            user_inputs.append(
                {
                    "id": p_id,
                    "name": html_name,
                    "desc": desc,
                    "mean": mean_val,
                    "cov": cov_val,
                    "dist": dist_val,
                    "min": p_min,
                    "max": p_max,
                }
            )

    # ----------------- 右侧：控制与输出 -----------------
    with col_right:
        st.markdown(
            """
            <div style='text-align: right; color: #555555; line-height: 1.5; font-family: Arial, sans-serif; font-size: 14px; margin-top: 25px;'>
                Created by Jingcheng Wang, Associate Professor. Fuzhou University<br>
                Contact: Jingchengwang@fzu.edu.cn
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        if show_back_button:
            back_clicked = st.button(
                "Return to Lifecycle Time-Varying Probability Assessment",
                key="btn_return_lifecycle",
                use_container_width=True,
            )
            if back_clicked:
                st.session_state.page_mode = "lifecycle"
                st.rerun()

        predict_clicked = st.button(
            "Calculate (Monte Carlo Simulation)",
            type="primary",
            use_container_width=True,
        )

        download_placeholder = st.empty()
        st.write("")

        st.markdown(
            "<h5 style='color: #333; font-family: Arial, sans-serif; font-weight: bold;'>"
            "Failure Mode Probabilities"
            "</h5>",
            unsafe_allow_html=True,
        )

        if predict_clicked:
            if assets is None:
                st.error("⚠️ 未检测到 model_assets_numpy.pkl。请将纯权重文件放在同一目录下。")
            else:
                weights = assets["weights"]
                scaler = assets["scaler"]
                label_names = assets["le"].classes_ if "le" in assets else np.array(["FFF", "PFF", "PSF"])

                n_features_required = getattr(scaler, "n_features_in_", len(user_inputs))
                n_features_current = len(user_inputs)

                if n_features_required != n_features_current:
                    st.error(
                        f"⚠️ 模型输入维度不匹配：当前 GUI 提供 {n_features_current} 个参数，"
                        f"但 scaler 需要 {n_features_required} 个参数。"
                        "请确认使用的是加入箍筋直径 d_t 后重新训练/导出的 20 维模型。"
                    )
                else:
                    N_SAMPLES = 5000
                    samples = np.zeros((N_SAMPLES, n_features_current), dtype=float)

                    for i, p in enumerate(user_inputs):
                        samples[:, i] = sample_parameter(
                            mean=p["mean"],
                            cov=p["cov"],
                            dist=p["dist"],
                            p_min=p["min"],
                            p_max=p["max"],
                            n_samples=N_SAMPLES,
                        )

                    predictions = predict_with_numpy_network(samples, assets)

                    probs_dict = {}
                    for idx, name in enumerate(label_names):
                        probs_dict[name] = np.sum(predictions == idx) / N_SAMPLES

                    st.session_state.direct_mc_probs = [
                        probs_dict.get("PFF", 0.0),
                        probs_dict.get("FFF", 0.0),
                        probs_dict.get("PSF", 0.0),
                    ]

                    excel_columns = [
                        "N",
                        "Dp (m)",
                        "rho_pile,l",
                        "alpha",
                        "S (Dp)",
                        "Dr",
                        "SD (m)",
                        "Hp/Dc",
                        "Dc (Dp)",
                        "rho_column,l",
                        "rho_pile,s",
                        "fyl (MPa)",
                        "fc (MPa)",
                        "rho_column,s",
                        "Xt",
                        "Xl",
                        "t (m)",
                        "dl (m)",
                        "fyt (MPa)",
                        "dt (m)",
                    ]

                    df_samples = pd.DataFrame(samples, columns=excel_columns)
                    df_samples["Failure_Mode"] = [label_names[i] for i in predictions]

                    df_stats = pd.DataFrame(
                        {
                            "Failure_Mode": label_names,
                            "Count": [int(np.sum(predictions == idx)) for idx, _ in enumerate(label_names)],
                            "Probability": [
                                f"{probs_dict.get(name, 0.0) * 100:.2f}%" for name in label_names
                            ],
                        }
                    )

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        df_samples.to_excel(writer, sheet_name="Samples", index=False)
                        df_stats.to_excel(writer, sheet_name="Statistics", index=False)

                    st.session_state.direct_excel_data = output.getvalue()

        if st.session_state.direct_excel_data is not None:
            with download_placeholder.container():
                st.download_button(
                    label="Save results",
                    data=st.session_state.direct_excel_data,
                    file_name="Direct_Prediction_Results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True,
                )

        probs = st.session_state.direct_mc_probs
        draw_progress_bar("PFF (Pier Flexure Failure)", probs[0] * 100, "#0078d4")
        draw_progress_bar("FFF (Foundation Flexure Failure)", probs[1] * 100, "#008000")
        draw_progress_bar("PSF (Pier Shear Failure)", probs[2] * 100, "#d83b01")

        st.write("---")
        st.markdown(
            "<h5 style='color: #333; font-family: Arial, sans-serif; font-weight: bold; margin-bottom: 10px;'>"
            "Parameter Schematic"
            "</h5>",
            unsafe_allow_html=True,
        )
        try:
            st.image("structure.png", use_container_width=True)
        except Exception:
            st.markdown(
                """
                <div style='border: 1px dashed #ccc; padding: 40px; text-align: center; color: #999; font-family: Arial, sans-serif;'>
                    Structure Image<br>(structure.png not found)
                </div>
                """,
                unsafe_allow_html=True,
            )


# ================== 7. 单独运行入口 ==================
def main():
    st.set_page_config(
        page_title="Seismic Failure Mode Probability Assessment",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_direct_prediction_app(assets=None, show_back_button=False)


if __name__ == "__main__":
    main()
