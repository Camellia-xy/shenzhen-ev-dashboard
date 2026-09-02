# 深圳充电需求数据分析及可视化

课程第二题的可运行起始版本。项目基于 ST-EVCDP 官方数据，展示深圳公共充电桩的描述性统计、时间规律、空间分布和数据质量，并为聚类与预测模块预留了清晰的扩展路径。

## 当前功能

- 自动读取并校验 `information.csv`、`time.csv`、`occupancy.csv`、`price.csv`
- 统计充电桩数量、交通分区、平均占用率、需求峰值
- 展示全市需求趋势、日均需求、典型 24 小时曲线
- 在深圳地图上展示分区容量、占用率、需求峰值与平均价格
- 检查缺失值、负数、超容量记录和重复时间戳
- 支持按日期筛选，并提供聚类与预测的迭代路线
- 使用 K-Means 对交通分区进行需求模式聚类，展示轮廓系数、典型日曲线、聚类画像与空间分布
- 使用按时间切分的回测比较24小时季节性基线与随机森林，并展示 MAE、RMSE、R²、预测曲线和特征重要性

## 数据说明

官方数据源：<https://github.com/IntelligentSystemsLab/ST-EVCDP>

当前 `data/raw/` 应包含：

```text
information.csv
occupancy.csv
price.csv
time.csv
```

ST-EVCDP 覆盖 18,061 根公共充电桩、247 个交通分区和 30 天数据，最小采样间隔为 5 分钟。仓库数据仅用于课程学习与研究，使用时请保留官方论文引用。

## 本地运行

```powershell
cd shenzhen-ev-dashboard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

浏览器打开终端显示的本地地址（通常是 `http://localhost:8501`）。

## Streamlit Community Cloud 部署

1. 登录 <https://share.streamlit.io/> 并连接 GitHub。
2. 选择仓库 `Camellia-xy/shenzhen-ev-dashboard`、分支 `main` 和入口文件 `app.py`。
3. Python 版本选择 `3.12`，然后点击 Deploy。
4. 部署完成后在 Sharing 中设置访问范围并邀请组员。

## 测试

```powershell
python -m unittest discover -s tests -v
```

## 下一步

1. 加入 `duration.csv`、`volume.csv` 和天气数据，完成多变量分析。
2. 加入天气、价格和节假日特征，进一步提升需求预测效果。
3. 增加分析结论导出、项目报告和答辩 PPT。
