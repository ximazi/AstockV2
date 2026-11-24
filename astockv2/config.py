"""集中路径与常量配置。

所有外部路径集中在此，便于换环境。所有成本/滑点口径与 v1 研究
(results/research/mainwave_path) 保持一致，避免两套账不可比。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---- 外部只读依赖（v1 工作区）------------------------------------
V1_ROOT = Path(r"D:/AICode/Astock")
V1_DATA = V1_ROOT / "data"
V1_RESULTS = V1_ROOT / "results" / "research" / "mainwave_path"

# ---- V2 本地输出 ---------------------------------------------------
V2_ROOT = Path(__file__).resolve().parent.parent
V2_RESULTS = V2_ROOT / "results"


@dataclass(frozen=True)
class CostModel:
    """双边总成本（基点）与单边滑点。

    v1 研究口径：双边总成本 30bp 为基准，40bp 为压力检验。
    """

    total_bp: float = 30.0
    stress_total_bp: float = 40.0
    slippage_bp: float = 3.0  # 单边滑点，已含在 total_bp 口径内（记账用）

    @property
    def half_bp(self) -> float:
        return self.total_bp / 2.0


COST = CostModel()

# ---- 交易规则常量 ---------------------------------------------------
LOT_SIZE = 100                # A 股一手 = 100 股
MAX_SEATS = 2                 # 股票席位上限
SEAT_BUDGET = 50_000.0        # 每席预算上限（元）
MAX_THEME_EXPOSURE = 0.50     # 同题材合计敞口上限
DRAWDOWN_PAUSE = 0.10         # 回撤暂停线
GREEN_ONLY = True             # 市场门只认 GREEN

# ---- 诊断常量 -------------------------------------------------------
FWD_BARS = 20                 # 入场后观察的交易日数（与 v1 "20 日内" 口径一致）
MFE_MATERIAL = 0.05           # 5%：算“价格确实涨过”的门槛
MFE_WEAK = 0.02               # 2%：低于此算无跟随
