from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def capacity_series(information: pd.DataFrame) -> pd.Series:
    return information.set_index("grid")["count"].astype(float)


def occupancy_summary(
    occupancy: pd.DataFrame, information: pd.DataFrame
) -> pd.DataFrame:
    """Return one row per timestamp with demand and utilization metrics."""
    capacity = capacity_series(information).reindex(occupancy.columns)
    total_capacity = capacity.sum()
    result = pd.DataFrame(index=occupancy.index)
    result["busy_piles"] = occupancy.sum(axis=1, min_count=1)
    result["utilization"] = result["busy_piles"] / total_capacity
    result["available_piles"] = total_capacity - result["busy_piles"]
    return result


def zone_summary(
    occupancy: pd.DataFrame,
    price: pd.DataFrame,
    information: pd.DataFrame,
) -> pd.DataFrame:
    """Build spatial features for every traffic zone."""
    result = information.set_index("grid").copy()
    result["avg_busy"] = occupancy.mean().reindex(result.index)
    result["peak_busy"] = occupancy.max().reindex(result.index)
    result["avg_price"] = price.mean().reindex(result.index)
    result["utilization"] = result["avg_busy"] / result["count"].replace(0, np.nan)
    result["utilization"] = result["utilization"].clip(lower=0)
    return result.reset_index()


def typical_day(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a timestamp-indexed summary into a typical 24-hour profile."""
    grouped = summary.groupby(summary.index.hour).mean(numeric_only=True)
    grouped.index.name = "hour"
    return grouped.reset_index()


def build_multivariate_frame(
    summary: pd.DataFrame,
    price: pd.DataFrame,
    supplemental: pd.DataFrame,
) -> pd.DataFrame:
    """Align official demand/price with team-provided weather and energy metrics."""
    official = summary[["busy_piles", "utilization"]].copy()
    official["avg_price"] = price.mean(axis=1)
    return official.join(supplemental, how="inner").sort_index()


def hour_adjusted_correlations(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Correlate variables after removing their average 24-hour pattern.

    Weather is shared by all 247 zones. Working at one row per timestamp avoids
    pseudo-replication, while hour demeaning reduces mechanical time-of-day
    confounding. The result remains descriptive rather than causal.
    """
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    residual = numeric - numeric.groupby(frame.index.hour).transform("mean")
    return residual.corr()


def zone_cluster_features(
    occupancy: pd.DataFrame,
    price: pd.DataFrame,
    information: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build standardized clustering features and raw hourly utilization curves.

    Hourly curves describe demand shape. Static features provide interpretable
    context without allowing large-capacity zones to dominate the distance.
    """
    capacity = capacity_series(information).reindex(occupancy.columns)
    utilization = occupancy.div(capacity, axis="columns").clip(0, 1)
    hourly = utilization.groupby(utilization.index.hour).mean().T
    hourly.columns = [f"h{hour:02d}" for hour in hourly.columns]

    info = information.set_index("grid").reindex(hourly.index)
    features = hourly.copy()
    features["log_capacity"] = np.log1p(info["count"].astype(float))
    features["avg_price"] = price.mean().reindex(hourly.index)
    for column in ("CBD", "dynamic_pricing"):
        features[column] = pd.to_numeric(info[column], errors="coerce")

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0)
    return features, hourly


def cluster_zones(
    occupancy: pd.DataFrame,
    price: pd.DataFrame,
    information: pd.DataFrame,
    n_clusters: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Cluster traffic zones and return assignments, profiles and quality score."""
    features, hourly = zone_cluster_features(occupancy, price, information)
    if not 2 <= n_clusters < len(features):
        raise ValueError("聚类数必须不小于2且小于交通分区数量")

    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = model.fit_predict(scaled)
    score = float(silhouette_score(scaled, labels))

    assignments = information.set_index("grid").reindex(features.index).copy()
    assignments["cluster_id"] = labels + 1
    assignments["cluster"] = assignments["cluster_id"].map(lambda value: f"群组 {value}")
    assignments["avg_utilization"] = hourly.mean(axis=1)
    assignments["peak_hour"] = hourly.idxmax(axis=1).str.removeprefix("h").astype(int)
    assignments["avg_price"] = price.mean().reindex(features.index)
    assignments.index.name = "grid"
    assignments = assignments.reset_index()

    profiles = hourly.assign(cluster=assignments.set_index("grid")["cluster"])
    profiles = profiles.groupby("cluster").mean().T
    profiles.index = profiles.index.str.removeprefix("h").astype(int)
    profiles.index.name = "hour"
    profiles = profiles.reset_index().melt(
        id_vars="hour", var_name="cluster", value_name="utilization"
    )
    return assignments, profiles, score


def evaluate_cluster_candidates(
    occupancy: pd.DataFrame,
    price: pd.DataFrame,
    information: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = 6,
    stability_runs: int = 6,
    sample_fraction: float = 0.8,
) -> pd.DataFrame:
    """Compare K values by separation, balance and resampling stability.

    Stability is the mean adjusted Rand index between the full-data solution
    and solutions fitted on repeated subsamples, then predicted for all zones.
    """
    features, _ = zone_cluster_features(occupancy, price, information)
    if not 2 <= min_clusters <= max_clusters < len(features):
        raise ValueError("候选聚类数必须不小于2且小于交通分区数量")
    if stability_runs < 2:
        raise ValueError("稳定性评估至少需要2次重采样")
    if not 0.5 <= sample_fraction < 1:
        raise ValueError("重采样比例必须在0.5到1之间")

    feature_values = features.to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(feature_values)
    rows = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        reference = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        reference_labels = reference.fit_predict(scaled)
        rng = np.random.default_rng(4200 + n_clusters)
        sample_size = min(
            len(features),
            max(n_clusters * 3, int(round(len(features) * sample_fraction))),
        )
        stability_scores = []
        for run in range(stability_runs):
            sample_indices = rng.choice(len(features), size=sample_size, replace=False)
            sample_scaler = StandardScaler().fit(feature_values[sample_indices])
            sampled_model = KMeans(
                n_clusters=n_clusters,
                random_state=1000 + run,
                n_init=10,
            )
            sampled_model.fit(sample_scaler.transform(feature_values[sample_indices]))
            stability_scores.append(
                adjusted_rand_score(
                    reference_labels,
                    sampled_model.predict(sample_scaler.transform(feature_values)),
                )
            )
        cluster_counts = np.bincount(reference_labels, minlength=n_clusters)
        rows.append(
            {
                "K": n_clusters,
                "silhouette": float(silhouette_score(scaled, reference_labels)),
                "stability_ari": float(np.mean(stability_scores)),
                "stability_std": float(np.std(stability_scores, ddof=1)),
                "min_cluster_share": float(cluster_counts.min() / len(features)),
            }
        )
    return pd.DataFrame(rows)


def cluster_profile_summary(
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Build human-readable, evidence-based summaries for zone clusters."""
    summary = (
        assignments.groupby("cluster")
        .agg(
            zone_count=("grid", "count"),
            avg_utilization=("avg_utilization", "mean"),
            avg_capacity=("count", "mean"),
            avg_price=("avg_price", "mean"),
            cbd_share=("CBD", "mean"),
            dynamic_pricing_share=("dynamic_pricing", "mean"),
            peak_hour=("peak_hour", "median"),
        )
        .reset_index()
    )
    profile_wide = profiles.pivot(index="cluster", columns="hour", values="utilization")
    periods = {
        "overnight_utilization": list(range(0, 6)),
        "morning_utilization": list(range(6, 11)),
        "daytime_utilization": list(range(11, 17)),
        "evening_utilization": list(range(17, 22)),
        "late_utilization": list(range(22, 24)),
    }
    for column, hours in periods.items():
        available_hours = [hour for hour in hours if hour in profile_wide.columns]
        summary[column] = summary["cluster"].map(
            profile_wide[available_hours].mean(axis=1) if available_hours else pd.Series(dtype=float)
        )

    utilization_rank = summary["avg_utilization"].rank(method="average", pct=True)
    levels = np.select(
        [utilization_rank <= 1 / 3, utilization_rank > 2 / 3],
        ["低占用", "高占用"],
        default="中占用",
    )

    def peak_period(hour: float) -> str:
        value = int(round(hour)) % 24
        if value <= 5:
            return "凌晨"
        if value <= 10:
            return "早间"
        if value <= 16:
            return "日间"
        if value <= 21:
            return "晚间"
        return "夜间"

    summary["profile_name"] = [
        f"{level}·{peak_period(hour)}峰值（{cluster}）"
        for level, hour, cluster in zip(levels, summary["peak_hour"], summary["cluster"])
    ]
    return summary


def demand_forecast_features(
    summary: pd.DataFrame,
    exogenous: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create leakage-safe hourly features for one-hour-ahead demand prediction."""
    hourly = summary["busy_piles"].resample("1h").mean().to_frame("target")
    hourly.index.name = "time"
    hours = hourly.index.hour
    weekdays = hourly.index.dayofweek
    hourly["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    hourly["weekday_sin"] = np.sin(2 * np.pi * weekdays / 7)
    hourly["weekday_cos"] = np.cos(2 * np.pi * weekdays / 7)
    hourly["is_weekend"] = (weekdays >= 5).astype(int)
    hourly["trend"] = np.arange(len(hourly), dtype=float)
    for lag in (1, 2, 3, 24, 48, 168):
        hourly[f"lag_{lag}"] = hourly["target"].shift(lag)
    hourly["rolling_24_mean"] = hourly["target"].shift(1).rolling(24).mean()
    hourly["rolling_24_std"] = hourly["target"].shift(1).rolling(24).std()
    if exogenous is not None:
        aggregation = {
            "temperature": "mean",
            "humidity": "mean",
            "wind_speed": "mean",
            "has_rain": "max",
            "avg_price": "mean",
        }
        available = {key: value for key, value in aggregation.items() if key in exogenous.columns}
        if available:
            hourly = hourly.join(exogenous.resample("1h").agg(available), how="left")
    return hourly.dropna()


def _make_forest(n_estimators: int = 300) -> RandomForestRegressor:
    """Create the shared, deterministic forecast estimator."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def _forecast_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    external_columns = [
        column
        for column in ("temperature", "humidity", "wind_speed", "has_rain", "avg_price")
        if column in frame.columns
    ]
    feature_columns = [
        column for column in frame.columns if column != "target" and column not in external_columns
    ]
    return feature_columns, external_columns


def _metric_row(
    name: str,
    actual: pd.Series,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float | str]:
    return {
        "model": name,
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def evaluate_demand_models(
    summary: pd.DataFrame,
    test_ratio: float = 0.2,
    exogenous: pd.DataFrame | None = None,
    n_estimators: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Backtest seasonal and RF models using one shared chronological split."""
    if not 0.1 <= test_ratio <= 0.4:
        raise ValueError("测试集比例必须在 0.1 到 0.4 之间")
    frame = demand_forecast_features(summary, exogenous=exogenous)
    split = int(len(frame) * (1 - test_ratio))
    if split < 50 or len(frame) - split < 10:
        raise ValueError("时间序列过短，无法进行可靠的训练测试切分")

    feature_columns, external_columns = _forecast_feature_columns(frame)
    train, test = frame.iloc[:split], frame.iloc[split:]
    model = _make_forest(n_estimators)
    model.fit(train[feature_columns], train["target"])

    prediction_frame = pd.DataFrame(index=test.index)
    prediction_frame["actual"] = test["target"]
    prediction_frame["seasonal_naive"] = test["lag_24"]
    prediction_frame["random_forest"] = model.predict(test[feature_columns])

    importance_model = model
    importance_columns = feature_columns
    if external_columns:
        enhanced_columns = feature_columns + external_columns
        enhanced = _make_forest(n_estimators)
        enhanced.fit(train[enhanced_columns], train["target"])
        prediction_frame["weather_price_forest"] = enhanced.predict(test[enhanced_columns])
        importance_model = enhanced
        importance_columns = enhanced_columns

    rows = []
    model_columns = [
        ("24小时季节性基线", "seasonal_naive"),
        ("随机森林", "random_forest"),
    ]
    if "weather_price_forest" in prediction_frame:
        model_columns.append(("随机森林（天气+价格）", "weather_price_forest"))
    for name, column in model_columns:
        actual = prediction_frame["actual"]
        predicted = prediction_frame[column]
        rows.append(_metric_row(name, actual, predicted))
    metrics = pd.DataFrame(rows)
    importance = pd.DataFrame(
        {"feature": importance_columns, "importance": importance_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return metrics, prediction_frame.reset_index(), importance, test.index.min()


def evaluate_demand_models_rolling(
    summary: pd.DataFrame,
    exogenous: pd.DataFrame | None = None,
    test_window: int = 72,
    max_splits: int = 4,
    n_estimators: int = 100,
) -> pd.DataFrame:
    """Evaluate forecast models over chronological expanding-window splits."""
    if test_window < 24:
        raise ValueError("每个回测窗口至少需要24小时")
    if max_splits < 2:
        raise ValueError("滚动回测至少需要2折")

    frame = demand_forecast_features(summary, exogenous=exogenous)
    minimum_train = 24 * 7
    initial_train = max(minimum_train, len(frame) - test_window * max_splits)
    available_splits = (len(frame) - initial_train) // test_window
    if available_splits < 2:
        raise ValueError("时间序列过短，无法进行至少2折滚动回测")
    split_count = min(max_splits, available_splits)
    initial_train = len(frame) - test_window * split_count

    feature_columns, external_columns = _forecast_feature_columns(frame)
    rows: list[dict[str, float | str | int | pd.Timestamp]] = []
    for fold_index in range(split_count):
        test_start_position = initial_train + fold_index * test_window
        test_end_position = test_start_position + test_window
        train = frame.iloc[:test_start_position]
        test = frame.iloc[test_start_position:test_end_position]

        base_model = _make_forest(n_estimators)
        base_model.fit(train[feature_columns], train["target"])
        predictions: list[tuple[str, pd.Series | np.ndarray]] = [
            ("24小时季节性基线", test["lag_24"]),
            ("随机森林", base_model.predict(test[feature_columns])),
        ]
        if external_columns:
            enhanced_columns = feature_columns + external_columns
            enhanced_model = _make_forest(n_estimators)
            enhanced_model.fit(train[enhanced_columns], train["target"])
            predictions.append(
                ("随机森林（天气+价格）", enhanced_model.predict(test[enhanced_columns]))
            )

        for model_name, predicted in predictions:
            row = _metric_row(model_name, test["target"], predicted)
            row.update(
                {
                    "fold": fold_index + 1,
                    "train_end": train.index.max(),
                    "test_start": test.index.min(),
                    "test_end": test.index.max(),
                    "test_hours": len(test),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def quality_report(
    occupancy: pd.DataFrame,
    price: pd.DataFrame,
    information: pd.DataFrame,
) -> dict[str, float | int]:
    capacity = capacity_series(information).reindex(occupancy.columns)
    above_capacity = occupancy.gt(capacity, axis="columns").sum().sum()
    return {
        "occupancy_missing": int(occupancy.isna().sum().sum()),
        "price_missing": int(price.isna().sum().sum()),
        "negative_occupancy": int(occupancy.lt(0).sum().sum()),
        "above_capacity": int(above_capacity),
        "duplicate_timestamps": int(occupancy.index.duplicated().sum()),
        "zones": int(occupancy.shape[1]),
        "timestamps": int(occupancy.shape[0]),
    }
