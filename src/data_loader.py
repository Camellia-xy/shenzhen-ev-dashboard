from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    information: pd.DataFrame
    occupancy: pd.DataFrame
    price: pd.DataFrame
    timestamps: pd.DatetimeIndex


REQUIRED_FILES = ("information.csv", "occupancy.csv", "price.csv", "time.csv")
SUPPLEMENTAL_COLUMNS = (
    "temperature",
    "station_pressure",
    "sea_level_pressure",
    "humidity",
    "wind_speed",
    "dew_point",
    "visibility",
    "cloud_cover",
    "has_rain",
    "avg_duration",
    "avg_volume",
)


def validate_data_dir(data_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"缺少官方数据文件：{names}")


def _load_timestamps(path: Path) -> pd.DatetimeIndex:
    raw = pd.read_csv(path)
    renamed = raw.rename(columns={"year": "year", "month": "month", "day": "day"})
    timestamps = pd.to_datetime(renamed[["year", "month", "day", "hour", "minute", "second"]])
    return pd.DatetimeIndex(timestamps, name="time")


def _load_matrix(path: Path, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path.name} 缺少 timestamp 列")
    frame = frame.drop(columns="timestamp")
    frame.columns = frame.columns.astype(str)
    if len(frame) != len(timestamps):
        raise ValueError(
            f"{path.name} 有 {len(frame)} 行，但 time.csv 有 {len(timestamps)} 行"
        )
    frame.index = timestamps
    return frame.apply(pd.to_numeric, errors="coerce")


def load_dataset(data_dir: str | Path) -> Dataset:
    data_dir = Path(data_dir)
    validate_data_dir(data_dir)

    information = pd.read_csv(data_dir / "information.csv")
    information["grid"] = information["grid"].astype(str)
    timestamps = _load_timestamps(data_dir / "time.csv")
    occupancy = _load_matrix(data_dir / "occupancy.csv", timestamps)
    price = _load_matrix(data_dir / "price.csv", timestamps)

    shared = [column for column in occupancy.columns if column in set(information["grid"])]
    if not shared:
        raise ValueError("information.csv 与 occupancy.csv 没有共同交通分区")

    return Dataset(
        information=information[information["grid"].isin(shared)].copy(),
        occupancy=occupancy[shared],
        price=price[shared],
        timestamps=timestamps,
    )


def load_supplemental_metrics(path: str | Path) -> pd.DataFrame:
    """Load the compact team-provided weather, duration and volume dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"缺少小组补充数据文件：{path.name}")

    frame = pd.read_csv(path, parse_dates=["datetime"])
    missing = [column for column in SUPPLEMENTAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少字段：{', '.join(missing)}")
    if frame["datetime"].duplicated().any():
        raise ValueError(f"{path.name} 存在重复时间戳")

    frame = frame.set_index("datetime").sort_index()
    frame.index.name = "time"
    return frame[list(SUPPLEMENTAL_COLUMNS)].apply(pd.to_numeric, errors="coerce")
