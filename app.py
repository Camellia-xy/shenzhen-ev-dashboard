from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import (
    cluster_zones,
    evaluate_demand_models,
    occupancy_summary,
    quality_report,
    typical_day,
    zone_summary,
)
from src.data_loader import Dataset, load_dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"


st.set_page_config(
    page_title="深圳充电需求洞察平台",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {--ink:#17332d; --muted:#617a73; --teal:#0f766e; --line:#dbe7e3; --paper:#ffffff;}
      .block-container {padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1440px;}
      [data-testid="stSidebar"] {background:#edf4f1; border-right:1px solid var(--line);}
      [data-testid="stSidebar"] h1 {color:var(--ink); font-size:1.45rem;}
      [data-testid="stMetric"] {background:var(--paper); border:1px solid var(--line); padding:16px 18px; border-radius:16px; box-shadow:0 8px 24px rgba(21,72,61,.06);}
      [data-testid="stMetricLabel"] {color:var(--muted); font-weight:600;}
      [data-testid="stMetricValue"] {color:var(--ink);}
      h1, h2, h3 {letter-spacing:-0.025em; color:var(--ink);}
      .hero {background:linear-gradient(125deg,#0b3d36 0%,#0f766e 62%,#34a28f 100%); border-radius:22px; padding:28px 32px; margin:4px 0 20px; box-shadow:0 16px 36px rgba(8,71,62,.18);}
      .hero h1 {color:#fff !important; margin:.35rem 0 .45rem; font-size:2.2rem;}
      .hero .eyebrow {color:#b9f3e7; font-weight:750; letter-spacing:.12em; font-size:.72rem;}
      .hero .subtitle {color:#e5fff9; font-size:1rem; max-width:760px;}
      .hero .proof {display:inline-block; margin-top:14px; color:#d4fff5; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22); padding:6px 10px; border-radius:999px; font-size:.78rem;}
      .section-note {color:var(--muted); margin-top:-.4rem; margin-bottom:1rem;}
      .stTabs [data-baseweb="tab-list"] {gap:8px; background:#e8f0ed; padding:6px; border-radius:14px;}
      .stTabs [data-baseweb="tab"] {height:40px; border-radius:10px; padding:0 15px; color:#49625c;}
      .stTabs [aria-selected="true"] {background:#fff !important; color:#0f766e !important; box-shadow:0 2px 10px rgba(15,118,110,.1);}
      .stTabs [data-baseweb="tab-highlight"] {display:none;}
      [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {background:#fff; border:1px solid var(--line); border-radius:16px; padding:8px;}
      .stDownloadButton button {border-radius:10px; border-color:#b8d1ca; color:#0f766e;}
      #MainMenu, [data-testid="stDeployButton"] {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="正在读取官方数据…")
def get_data() -> Dataset:
    return load_dataset(DATA_DIR)


@st.cache_data(show_spinner="正在训练 K-Means 聚类模型…")
def get_clusters(_dataset: Dataset, n_clusters: int, start: pd.Timestamp, end: pd.Timestamp):
    occupancy = _dataset.occupancy.loc[( _dataset.occupancy.index >= start) & (_dataset.occupancy.index < end)]
    price = _dataset.price.loc[( _dataset.price.index >= start) & (_dataset.price.index < end)]
    return cluster_zones(
        occupancy,
        price,
        _dataset.information,
        n_clusters=n_clusters,
    )


@st.cache_data(show_spinner="正在训练需求预测模型…")
def get_forecast(_summary: pd.DataFrame):
    return evaluate_demand_models(_summary)


try:
    data = get_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.info("请按 README 的说明下载 ST-EVCDP 官方 CSV 文件后刷新页面。")
    st.stop()

full_summary = occupancy_summary(data.occupancy, data.information)

with st.sidebar:
    st.title("⚡ EV Insight")
    st.caption("深圳公共充电需求分析")
    st.divider()
    date_min = full_summary.index.min().date()
    date_max = full_summary.index.max().date()
    date_range = st.date_input(
        "分析日期",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    metric_label = st.selectbox(
        "空间指标",
        ["平均占用率", "平均忙碌桩数", "峰值忙碌桩数", "平均价格"],
    )
    aggregation_label = st.selectbox("趋势聚合粒度", ["1小时", "30分钟", "1天"], index=0)
    zone_scope = st.multiselect("区域属性", ["CBD", "非CBD"], default=["CBD", "非CBD"])
    pricing_scope = st.multiselect("定价方式", ["动态定价", "固定定价"], default=["动态定价", "固定定价"])
    st.divider()
    st.success("官方数据已载入")
    st.caption("ST-EVCDP · 4个原始CSV · 页面指标实时计算")

if isinstance(date_range, tuple) and len(date_range) == 2:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.DateOffset(days=1)
    filtered_occupancy = data.occupancy[(data.occupancy.index >= start) & (data.occupancy.index < end)]
    filtered_price = data.price[(data.price.index >= start) & (data.price.index < end)]
else:
    start = pd.Timestamp(date_min)
    end = pd.Timestamp(date_max) + pd.DateOffset(days=1)
    filtered_occupancy = data.occupancy
    filtered_price = data.price

filtered = occupancy_summary(filtered_occupancy, data.information)
zones = zone_summary(filtered_occupancy, filtered_price, data.information)

cbd_values = []
if "CBD" in zone_scope:
    cbd_values.append(1)
if "非CBD" in zone_scope:
    cbd_values.append(0)
pricing_values = []
if "动态定价" in pricing_scope:
    pricing_values.append(1)
if "固定定价" in pricing_scope:
    pricing_values.append(0)
visible_zones = zones[zones["CBD"].isin(cbd_values) & zones["dynamic_pricing"].isin(pricing_values)]

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">URBAN ENERGY ANALYTICS</div>
      <h1 style="margin:.3rem 0">深圳充电需求洞察平台</h1>
      <div class="subtitle">基于18,061根公共充电桩的真实状态记录，分析深圳充电需求的时间节律、空间差异与短期变化。</div>
      <div class="proof">数据来源：ST-EVCDP 官方仓库 · 2022-06-19—2022-07-18 · 5分钟粒度</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["运营总览", "时间规律", "空间分布", "分区聚类", "需求预测", "数据质量", "数据与口径"])

with tabs[0]:
    peak_time = filtered["busy_piles"].idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("公共充电桩", f"{int(data.information['count'].sum()):,}")
    c2.metric("交通分区", f"{len(data.information):,}")
    c3.metric("平均占用率", f"{filtered['utilization'].mean():.1%}")
    c4.metric("需求峰值", f"{int(filtered['busy_piles'].max()):,}", peak_time.strftime("%m-%d %H:%M"))

    aggregation = {"1小时": "1h", "30分钟": "30min", "1天": "1D"}[aggregation_label]
    hourly = filtered.resample(aggregation).mean(numeric_only=True).reset_index()
    fig = px.area(
        hourly,
        x="time",
        y="busy_piles",
        labels={"time": "时间", "busy_piles": "忙碌充电桩数"},
        color_discrete_sequence=["#16866f"],
        title=f"全市充电需求趋势 · {aggregation_label}平均值",
    )
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.download_button(
        "下载当前趋势数据（CSV）",
        data=hourly.to_csv(index=False).encode("utf-8-sig"),
        file_name="shenzhen_ev_demand_filtered.csv",
        mime="text/csv",
    )

    left, right = st.columns([1.2, 1])
    with left:
        daily = filtered.resample("1D").agg({"busy_piles": "mean", "utilization": "mean"}).reset_index()
        daily["日期类型"] = daily["time"].dt.dayofweek.map(lambda x: "周末" if x >= 5 else "工作日")
        fig = px.bar(
            daily,
            x="time",
            y="busy_piles",
            color="日期类型",
            labels={"time": "日期", "busy_piles": "日均忙碌桩数"},
            color_discrete_map={"工作日": "#294f64", "周末": "#ef9c66"},
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        top = visible_zones.nlargest(10, "utilization").sort_values("utilization")
        fig = px.bar(
            top,
            x="utilization",
            y="grid",
            orientation="h",
            labels={"grid": "交通分区", "utilization": "平均占用率"},
            color="utilization",
            color_continuous_scale=["#d9f1e9", "#167d68"],
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

with tabs[1]:
    st.subheader("典型日充电节律")
    st.markdown('<div class="section-note">曲线由当前日期范围内所有5分钟记录按小时取均值，不包含模拟或补造数据。</div>', unsafe_allow_html=True)
    profile = typical_day(filtered)
    profile["时段"] = pd.cut(
        profile["hour"],
        [-1, 5, 9, 16, 20, 23],
        labels=["深夜", "早高峰", "日间", "晚高峰", "夜间"],
    )
    fig = px.line(
        profile,
        x="hour",
        y="utilization",
        markers=True,
        labels={"hour": "小时", "utilization": "平均占用率"},
        color_discrete_sequence=["#167d68"],
    )
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig, width="stretch")
    peak_hour = int(profile.loc[profile["utilization"].idxmax(), "hour"])
    st.info(f"所选日期范围内的典型需求峰值出现在 {peak_hour:02d}:00 左右。后续将加入工作日/周末显著性检验和天气联动分析。")

with tabs[2]:
    st.subheader("交通分区空间分布")
    st.markdown(f'<div class="section-note">当前显示 {len(visible_zones)} 个分区；侧栏的日期、区域属性和定价方式筛选均已生效。</div>', unsafe_allow_html=True)
    metric_map = {
        "平均占用率": "utilization",
        "平均忙碌桩数": "avg_busy",
        "峰值忙碌桩数": "peak_busy",
        "平均价格": "avg_price",
    }
    metric = metric_map[metric_label]
    fig = px.scatter_map(
        visible_zones,
        lat="la",
        lon="lon",
        size="count",
        color=metric,
        hover_name="grid",
        hover_data={"count": True, "avg_busy": ":.1f", "utilization": ":.1%", "avg_price": ":.2f", "la": False, "lon": False},
        color_continuous_scale="Tealgrn",
        size_max=30,
        zoom=9.5,
        center={"lat": 22.57, "lon": 114.05},
        map_style="open-street-map",
        labels={metric: metric_label, "count": "充电桩数"},
    )
    fig.update_layout(height=610, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.download_button(
        "下载当前分区统计（CSV）",
        data=visible_zones.to_csv(index=False).encode("utf-8-sig"),
        file_name="shenzhen_ev_zone_statistics.csv",
        mime="text/csv",
    )

with tabs[3]:
    st.subheader("交通分区充电模式聚类")
    st.caption("K-Means 使用24小时占用率曲线、充电桩容量、平均价格、CBD和动态定价属性；全部特征在聚类前进行标准化。")
    cluster_count = st.slider("聚类数量 K", min_value=3, max_value=6, value=4)
    assignments, profiles, silhouette = get_clusters(data, cluster_count, start, end)

    c1, c2, c3 = st.columns(3)
    c1.metric("聚类数量", cluster_count)
    c2.metric("轮廓系数", f"{silhouette:.3f}", help="越接近1，类内越紧凑、类间越分离")
    c3.metric("已覆盖分区", f"{len(assignments)}/{len(data.information)}")

    left, right = st.columns([1.15, 1])
    with left:
        fig = px.line(
            profiles,
            x="hour",
            y="utilization",
            color="cluster",
            markers=True,
            labels={"hour": "小时", "utilization": "平均占用率", "cluster": "聚类"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_xaxes(dtick=2)
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
    with right:
        cluster_stats = (
            assignments.groupby("cluster")
            .agg(
                分区数量=("grid", "count"),
                平均占用率=("avg_utilization", "mean"),
                平均容量=("count", "mean"),
                平均价格=("avg_price", "mean"),
                典型峰值小时=("peak_hour", "median"),
            )
            .reset_index()
            .rename(columns={"cluster": "聚类"})
        )
        cluster_stats["平均占用率"] = cluster_stats["平均占用率"].map(lambda value: f"{value:.1%}")
        cluster_stats["平均容量"] = cluster_stats["平均容量"].round(1)
        cluster_stats["平均价格"] = cluster_stats["平均价格"].round(2)
        cluster_stats["典型峰值小时"] = cluster_stats["典型峰值小时"].round().astype(int).map(lambda value: f"{value:02d}:00")
        st.dataframe(cluster_stats, width="stretch", hide_index=True)
        st.info("轮廓系数用于比较不同 K 值；最终聚类数还应结合曲线差异和业务可解释性确定。")

    with st.expander("查看聚类数量 K 的选择依据", expanded=True):
        score_frame = pd.DataFrame(
            [
                {"K": candidate, "轮廓系数": get_clusters(data, candidate, start, end)[2]}
                for candidate in range(3, 7)
            ]
        )
        best_k = int(score_frame.loc[score_frame["轮廓系数"].idxmax(), "K"])
        fig = px.bar(
            score_frame,
            x="K",
            y="轮廓系数",
            text_auto=".3f",
            color="轮廓系数",
            color_continuous_scale=["#d9f1e9", "#167d68"],
        )
        fig.update_xaxes(dtick=1)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.success(f"在 K=3–6 的候选范围内，K={best_k} 的轮廓系数最高，建议作为报告中的主方案。")

    fig = px.scatter_map(
        assignments,
        lat="la",
        lon="lon",
        size="count",
        color="cluster",
        hover_name="grid",
        hover_data={"count": True, "avg_utilization": ":.1%", "peak_hour": True, "la": False, "lon": False},
        size_max=28,
        zoom=9.5,
        center={"lat": 22.57, "lon": 114.05},
        map_style="open-street-map",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={"cluster": "聚类", "count": "充电桩数", "avg_utilization": "平均占用率", "peak_hour": "峰值小时"},
    )
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, width="stretch")
    st.download_button(
        "下载分区聚类结果（CSV）",
        data=assignments.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"shenzhen_ev_clusters_k{cluster_count}.csv",
        mime="text/csv",
    )

with tabs[4]:
    st.subheader("全市短期充电需求预测")
    st.caption("目标为下一小时的全市平均忙碌充电桩数。训练与测试严格按时间先后切分，避免未来信息泄漏。")
    forecast_metrics, predictions, importance, split_time = get_forecast(full_summary)
    baseline = forecast_metrics.loc[forecast_metrics["model"] == "24小时季节性基线"].iloc[0]
    forest = forecast_metrics.loc[forecast_metrics["model"] == "随机森林"].iloc[0]
    improvement = (baseline["MAE"] - forest["MAE"]) / baseline["MAE"] if baseline["MAE"] else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("测试集起点", split_time.strftime("%m-%d %H:%M"))
    c2.metric("随机森林 MAE", f"{forest['MAE']:.1f}", f"较基线 {improvement:+.1%}")
    c3.metric("随机森林 RMSE", f"{forest['RMSE']:.1f}")
    c4.metric("随机森林 R²", f"{forest['R2']:.3f}")

    chart_data = predictions.tail(24 * 7).rename(
        columns={
            "time": "时间",
            "actual": "实际值",
            "seasonal_naive": "24小时季节性基线",
            "random_forest": "随机森林",
        }
    )
    chart_long = chart_data.melt(
        id_vars="时间", var_name="序列", value_name="忙碌充电桩数"
    )
    fig = px.line(
        chart_long,
        x="时间",
        y="忙碌充电桩数",
        color="序列",
        color_discrete_map={"实际值": "#172f3f", "24小时季节性基线": "#ef9c66", "随机森林": "#16866f"},
    )
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1, 1])
    with left:
        display_metrics = forecast_metrics.copy()
        display_metrics[["MAE", "RMSE", "R2"]] = display_metrics[["MAE", "RMSE", "R2"]].round(3)
        display_metrics = display_metrics.rename(columns={"model": "模型"})
        st.subheader("回测指标")
        st.dataframe(display_metrics, width="stretch", hide_index=True)
        st.caption("MAE/RMSE 越低越好；R² 越接近1越好。季节性基线使用前一天同一小时的需求作为预测。")
    with right:
        top_features = importance.head(10).sort_values("importance")
        fig = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "重要性", "feature": "预测特征"},
            color="importance",
            color_continuous_scale=["#d9f1e9", "#167d68"],
        )
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=35, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    d1, d2 = st.columns(2)
    d1.download_button("下载预测明细（CSV）", predictions.to_csv(index=False).encode("utf-8-sig"), "shenzhen_ev_forecast.csv", "text/csv")
    d2.download_button("下载模型指标（CSV）", forecast_metrics.to_csv(index=False).encode("utf-8-sig"), "shenzhen_ev_model_metrics.csv", "text/csv")

    if forest["MAE"] < baseline["MAE"]:
        st.success(f"随机森林的 MAE 相比24小时季节性基线降低 {improvement:.1%}，说明历史滞后和日历特征提供了额外预测信息。")
    else:
        st.warning("随机森林尚未优于季节性基线，说明当前需求的日周期较强；后续应加入天气、价格和节假日特征。")

with tabs[5]:
    st.subheader("当前筛选范围的数据质量")
    st.markdown('<div class="section-note">质量检查与日期筛选同步；异常记录只标记，不会静默修改原始值。</div>', unsafe_allow_html=True)
    report = quality_report(filtered_occupancy, filtered_price, data.information)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("时间点", f"{report['timestamps']:,}")
    c2.metric("占用缺失值", f"{report['occupancy_missing']:,}")
    c3.metric("超容量记录", f"{report['above_capacity']:,}")
    c4.metric("重复时间戳", f"{report['duplicate_timestamps']:,}")
    st.subheader("检查结论")
    checks = pd.DataFrame(
        [
            ["时间轴完整性", report["duplicate_timestamps"] == 0, "检查重复时间戳"],
            ["占用值非负", report["negative_occupancy"] == 0, "负值应标记为异常"],
            ["占用不超过容量", report["above_capacity"] == 0, "忙碌桩数应不高于分区容量"],
            ["关键数据无缺失", report["occupancy_missing"] == 0, "缺失值需插补或剔除"],
        ],
        columns=["规则", "通过", "说明"],
    )
    st.dataframe(checks, width="stretch", hide_index=True)

with tabs[6]:
    st.subheader("数据来源与计算口径")
    st.markdown('<div class="section-note">本页列出的行列数直接读取当前本地文件，便于核验界面中的每一个数字。</div>', unsafe_allow_html=True)
    dictionary = pd.DataFrame(
        [
            ["information.csv", len(data.information), len(data.information.columns), "分区容量、坐标、CBD和定价属性"],
            ["occupancy.csv", len(data.occupancy), len(data.occupancy.columns) + 1, "每5分钟各分区忙碌充电桩数"],
            ["price.csv", len(data.price), len(data.price.columns) + 1, "每5分钟各分区平均充电价格"],
            ["time.csv", len(data.timestamps), 6, "年、月、日、时、分、秒"],
        ],
        columns=["本地文件", "数据行数", "数据列数", "用途"],
    )
    st.dataframe(dictionary, width="stretch", hide_index=True)
    st.markdown("""
    **核心口径**

    - 平均占用率 = 全市忙碌充电桩数 ÷ 18,061根公共充电桩。
    - 空间统计、质量检查与聚类使用侧栏选定日期范围；预测模型固定使用完整30天序列进行时间回测。
    - 地图圆点大小代表分区充电桩容量，颜色代表侧栏选择的空间指标。
    - 聚类输入为24小时占用率曲线、容量、价格、CBD与动态定价属性，输入特征先标准化。
    - 预测测试集严格晚于训练集；随机森林不会读取未来时刻目标值。
    """)
    st.link_button("打开 ST-EVCDP 官方数据仓库", "https://github.com/IntelligentSystemsLab/ST-EVCDP")
