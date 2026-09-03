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
- 综合小组整理的天气、充电时长和充电电量数据，展示去除小时规律后的相关性与雨天/非雨天对比
- 在同一测试集上比较季节性基线、基础随机森林和“天气+价格”增强随机森林

## 数据说明

官方数据源：<https://github.com/IntelligentSystemsLab/ST-EVCDP>

当前 `data/raw/` 应包含：

```text
information.csv
occupancy.csv
price.csv
time.csv
```

小组补充数据整理为 `data/processed/supplemental_5min.csv.gz`，包含 8,640 个5分钟时间点的天气变量、全市平均充电时长和平均充电电量。主指标继续读取官方原始数据；清洗报告中的159个超容量值只在质量页披露，不会静默覆盖原始数据。

分析口径说明：天气在同一时刻对247个交通分区完全相同，因此相关性不能把长表的2,134,080行当作独立天气观测。平台先聚合为每个时间点一行，再剔除各变量的典型小时规律。不同目标的 RMSE 具有不同单位，不能直接用柱高判断哪个目标“预测得更好”。

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

1. 补充天气数据的原始来源、观测站和插值方法说明。
2. 用更长时间跨度和真实天气预报验证天气变量的样本外价值。
3. 增加分析结论导出、项目报告和答辩 PPT。
