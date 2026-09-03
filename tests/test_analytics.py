import unittest

import pandas as pd

from src.analytics import (
    build_multivariate_frame,
    cluster_profile_summary,
    cluster_zones,
    demand_forecast_features,
    evaluate_demand_models,
    evaluate_demand_models_rolling,
    evaluate_cluster_candidates,
    hour_adjusted_correlations,
    occupancy_summary,
    quality_report,
    typical_day,
    zone_cluster_features,
    zone_summary,
)


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2022-06-19", periods=4, freq="6h")
        self.occupancy = pd.DataFrame({"101": [1, 2, 3, 4], "102": [0, 2, 2, 4]}, index=index)
        self.price = pd.DataFrame({"101": [1.0] * 4, "102": [0.8] * 4}, index=index)
        self.info = pd.DataFrame(
            {
                "grid": ["101", "102"],
                "count": [4, 4],
                "lon": [114.0, 114.1],
                "la": [22.5, 22.6],
            }
        )

    def test_occupancy_summary(self):
        result = occupancy_summary(self.occupancy, self.info)
        self.assertEqual(result.iloc[-1]["busy_piles"], 8)
        self.assertEqual(result.iloc[-1]["utilization"], 1)

    def test_zone_summary(self):
        result = zone_summary(self.occupancy, self.price, self.info).set_index("grid")
        self.assertAlmostEqual(result.loc["101", "avg_busy"], 2.5)
        self.assertAlmostEqual(result.loc["102", "avg_price"], 0.8)

    def test_typical_day(self):
        result = typical_day(occupancy_summary(self.occupancy, self.info))
        self.assertEqual(len(result), 4)

    def test_quality_report_detects_valid_fixture(self):
        result = quality_report(self.occupancy, self.price, self.info)
        self.assertEqual(result["above_capacity"], 0)
        self.assertEqual(result["occupancy_missing"], 0)

    def test_cluster_features_are_one_row_per_zone(self):
        features, hourly = zone_cluster_features(self.occupancy, self.price, self.info.assign(CBD=0, dynamic_pricing=0))
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(hourly.shape, (2, 4))

    def test_cluster_zones_separates_opposite_profiles(self):
        index = pd.date_range("2022-06-19", periods=48, freq="1h")
        day = [8 if 8 <= hour <= 17 else 1 for hour in index.hour]
        night = [1 if 8 <= hour <= 17 else 8 for hour in index.hour]
        occupancy = pd.DataFrame(
            {**{str(i): day for i in range(4)}, **{str(i): night for i in range(4, 8)}},
            index=index,
        )
        price = pd.DataFrame(1.0, index=index, columns=occupancy.columns)
        info = pd.DataFrame(
            {
                "grid": occupancy.columns,
                "count": [10] * 8,
                "lon": [114.0] * 8,
                "la": [22.5] * 8,
                "CBD": [0] * 8,
                "dynamic_pricing": [0] * 8,
            }
        )
        assignments, profiles, score = cluster_zones(occupancy, price, info, n_clusters=2)
        self.assertEqual(assignments["cluster"].nunique(), 2)
        self.assertEqual(len(profiles), 48)
        self.assertGreater(score, 0.9)

        summary = cluster_profile_summary(assignments, profiles)
        self.assertEqual(len(summary), 2)
        self.assertTrue(summary["profile_name"].str.contains("占用").all())
        self.assertTrue(summary["profile_name"].str.contains("峰值").all())

        candidates = evaluate_cluster_candidates(
            occupancy,
            price,
            info,
            min_clusters=2,
            max_clusters=2,
            stability_runs=3,
        )
        self.assertEqual(candidates["K"].tolist(), [2])
        self.assertTrue(candidates["stability_ari"].between(-1, 1).all())
        self.assertTrue(candidates["min_cluster_share"].between(0, 1).all())

    def test_forecast_features_do_not_use_current_target(self):
        index = pd.date_range("2022-01-01", periods=240, freq="1h")
        summary = pd.DataFrame({"busy_piles": range(240)}, index=index)
        features = demand_forecast_features(summary)
        first = features.iloc[0]
        self.assertEqual(first["lag_1"], first["target"] - 1)
        self.assertEqual(first["lag_24"], first["target"] - 24)

    def test_forecast_models_return_metrics_and_predictions(self):
        index = pd.date_range("2022-01-01", periods=24 * 40, freq="1h")
        signal = pd.Series(index.hour, index=index).map(lambda hour: 100 + 20 * (8 <= hour <= 20))
        summary = pd.DataFrame({"busy_piles": signal.astype(float)}, index=index)
        metrics, predictions, importance, split_time = evaluate_demand_models(summary)
        self.assertEqual(set(metrics["model"]), {"24小时季节性基线", "随机森林"})
        self.assertGreater(len(predictions), 100)
        self.assertAlmostEqual(metrics.loc[metrics["model"] == "24小时季节性基线", "MAE"].iloc[0], 0)
        self.assertEqual(predictions["time"].min(), split_time)
        self.assertGreater(importance["importance"].sum(), 0.99)

    def test_multivariate_frame_aligns_official_and_supplemental_data(self):
        summary = occupancy_summary(self.occupancy, self.info)
        supplemental = pd.DataFrame(
            {"temperature": [28, 29, 30, 31], "avg_duration": [1, 1, 2, 2]},
            index=self.occupancy.index,
        )
        result = build_multivariate_frame(summary, self.price, supplemental)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result.iloc[0]["avg_price"], 0.9)
        self.assertIn("temperature", result.columns)

    def test_hour_adjusted_correlations_return_square_matrix(self):
        index = pd.date_range("2022-01-01", periods=24 * 5, freq="1h")
        frame = pd.DataFrame(
            {
                "demand": index.hour + pd.Series(range(len(index)), index=index) / 100,
                "temperature": index.hour * 2 + pd.Series(range(len(index)), index=index) / 50,
            },
            index=index,
        )
        result = hour_adjusted_correlations(frame, ["demand", "temperature"])
        self.assertEqual(result.shape, (2, 2))
        self.assertAlmostEqual(result.loc["demand", "temperature"], 1.0)

    def test_weather_enhanced_forecast_is_returned_when_features_exist(self):
        index = pd.date_range("2022-01-01", periods=24 * 40, freq="1h")
        signal = pd.Series(index.hour, index=index).map(lambda hour: 100 + 20 * (8 <= hour <= 20))
        summary = pd.DataFrame({"busy_piles": signal.astype(float)}, index=index)
        exogenous = pd.DataFrame(
            {
                "temperature": 25 + index.hour / 10,
                "humidity": 80 - index.hour / 10,
                "wind_speed": 2.0,
                "has_rain": (index.dayofweek == 0).astype(int),
                "avg_price": 1.0,
            },
            index=index,
        )
        metrics, predictions, importance, _ = evaluate_demand_models(
            summary, exogenous=exogenous
        )
        self.assertIn("随机森林（天气+价格）", set(metrics["model"]))
        self.assertIn("weather_price_forest", predictions.columns)
        self.assertIn("temperature", set(importance["feature"]))

    def test_rolling_forecast_uses_chronological_non_overlapping_windows(self):
        index = pd.date_range("2022-01-01", periods=24 * 40, freq="1h")
        signal = pd.Series(index.hour, index=index).map(lambda hour: 100 + 20 * (8 <= hour <= 20))
        summary = pd.DataFrame({"busy_piles": signal.astype(float)}, index=index)
        result = evaluate_demand_models_rolling(
            summary,
            test_window=48,
            max_splits=3,
            n_estimators=20,
        )
        self.assertEqual(result["fold"].nunique(), 3)
        self.assertEqual(set(result["model"]), {"24小时季节性基线", "随机森林"})
        fold_windows = result.drop_duplicates("fold").sort_values("fold")
        self.assertTrue((fold_windows["train_end"] < fold_windows["test_start"]).all())
        current_starts = fold_windows["test_start"].iloc[1:].reset_index(drop=True)
        previous_ends = fold_windows["test_end"].iloc[:-1].reset_index(drop=True)
        self.assertTrue((current_starts > previous_ends).all())
        self.assertTrue((result["test_hours"] == 48).all())


if __name__ == "__main__":
    unittest.main()
