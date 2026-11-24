"""零重跑诊断：检查同股异日安慰剂的高收益是否来自 ci < si（已知后续趋势的前视偏差）。

2026-11-24 的运行结论（重要）：
  安慰剂抽样**没有**前视偏差 —— control.py / s4_arm.py 都是 `rng.sample(days, k)`
  的均匀抽样，不含任何未来信息。
  "信号前"远好于"信号后"的真正原因是**持仓窗口重叠**：安慰剂里有一批交易的
  窗口含住了那次信号的行情，买的是同一波行情的更早位置（+7.4%），干净桶只有
  +0.7%。所以按"窗口含不含信号行情"拆桶，而不是按抽样方向筛。

  永久版工具（口径已固化为三条对照并排输出）：`python run_placebo_audit.py`
  完整复核见 `docs/安慰剂口径修正与验收结论_20261124.md`
"""
from pathlib import Path
import pandas as pd

ROOT = Path("results")


def analyze(name: str, sig_path: Path, pbo_path: Path, regime_map: dict) -> None:
    sig = pd.read_csv(sig_path)
    pbo = pd.read_csv(pbo_path)
    for df in (sig, pbo):
        df["anchor_day"] = df["anchor_day"].astype(str)
        df["year"] = df["anchor_day"].str[:4]
        df["regime"] = df["anchor_day"].map(regime_map).fillna("UNKNOWN")

    bounds = (
        sig.groupby(["symbol", "year"])["anchor_day"]
        .agg(first_sig="min", last_sig="max", sig_cnt="nunique")
        .reset_index()
    )
    m = pbo.merge(bounds, on=["symbol", "year"], how="inner")

    def bucket(row):
        if row["anchor_day"] < row["first_sig"]:
            return "1_STRICT_BEFORE (信号前)"
        if row["anchor_day"] > row["last_sig"]:
            return "3_STRICT_AFTER  (信号后)"
        return "2_BETWEEN       (多信号间)"

    m["pos"] = m.apply(bucket, axis=1)

    print(f"\n================ {name} ================")
    print(f"真实信号均值: {sig['net_ret'].mean()*100:+.3f}% (n={len(sig)})")
    print(f"安慰剂总均值: {m['net_ret'].mean()*100:+.3f}% (n={len(m)})")
    print("\n--- 按相对信号日先后拆分 ---")
    g = m.groupby("pos")["net_ret"].agg(
        笔数="size",
        均值pct=lambda s: round(s.mean() * 100, 3),
        中位pct=lambda s: round(s.median() * 100, 3),
        胜率pct=lambda s: round((s > 0).mean() * 100, 1),
    )
    print(g.to_string())

    print("\n--- 按 市场环境 x 相对先后 交叉拆分 ---")
    g2 = m.groupby(["regime", "pos"])["net_ret"].agg(
        笔数="size",
        均值pct=lambda s: round(s.mean() * 100, 3),
        胜率pct=lambda s: round((s > 0).mean() * 100, 1),
    )
    print(g2.to_string())


def main() -> None:
    reg_path = ROOT / "regime" / "regime_series_hs300.csv"
    rmap = {}
    if reg_path.exists():
        rf = pd.read_csv(reg_path)
        rmap = dict(zip(rf["date"].astype(str).str[:10], rf["regime"]))

    analyze("S4 信号 vs 同股异日", ROOT / "s4_arm/s4_arm.csv", ROOT / "s4_arm/s4_placebo.csv", rmap)
    analyze("R1 真实 vs 同股异日", ROOT / "control/T0_真实信号.csv", ROOT / "control/C2_同股异日.csv", rmap)


if __name__ == "__main__":
    main()
