# -*- coding: utf-8 -*-
"""
A股主线题材+ETF投资系统 - 配置文件
"""

# ==================== Tushare配置 ====================
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")  # 令牌放在仓库根目录的 .env，不写在代码里

# ==================== 账户配置 ====================
INITIAL_CAPITAL = 100000  # 初始资金（元），默认10万
PORTFOLIO_FILE = "portfolio_data.json"  # 仓位数据文件

# ==================== 主线评分权重 ====================
THEME_WEIGHTS = {
    'fund_flow': 0.30,       # 资金流向强度 30%
    'market_strength': 0.25,  # 市场相对强度 25%
    'policy_support': 0.20,   # 政策支持度 20%
    'consensus': 0.15,        # 机构共识度 15%
    'prosperity': 0.10,       # 景气度趋势 10%
}

# ==================== ETF评分权重（业绩为王 - 旧版，已弃用） ====================
ETF_WEIGHTS = {
    'performance': 0.50,      # 业绩表现 50% (核心指标)
    'liquidity': 0.18,        # 流动性 18%
    'aum': 0.12,              # 规模 12%
    'risk_adjusted': 0.10,    # 风险调整收益 10%
    'expense': 0.06,          # 费率 6%
    'premium': 0.04,          # 折溢价 4%
}

# ==================== ETF评分权重 V2（信号驱动） ====================
# 新逻辑：DKX金叉+MACD多头是核心，业绩只作辅助参考
ETF_WEIGHTS_V2 = {
    'signal': 0.50,           # 信号强度 50% (DKX+MACD 双确认)
    'theme_heat': 0.20,       # 主线热度 20% (该ETF所属题材的资金/相对强度)
    'liquidity': 0.10,        # 流动性 10%
    'aum': 0.10,              # 规模 10%
    'performance': 0.10,      # 业绩参考 10% (仅同强度时排序)
}

# ==================== DKX+MACD 信号引擎配置 ====================
SIGNAL_ENGINE = {
    # DKX 标准参数：DKX = (1*C[19天前] + 2*C[18天前] + ... + 20*C[今]) / 210
    'dkx_period': 20,         # DKX 计算用最近 20 天
    'madkx_period': 10,       # MADKX 平滑周期

    # 金叉确认窗口：缩窄到 2 天，避免追在金叉尾巴上
    'cross_window': 2,
    'min_data_days': 60,

    # 信号强度阈值（v2 收紧后）
    'buy_score_min': 70,      # 综合分 ≥70 才入选 BUY（原 60，太松）
    'sell_score_max': 35,     # 综合分 ≤35 触发 SELL（原 30，对持仓更敏感）

    # —— 新增硬约束：对入场右侧确认更严格 ——
    # 1) MACD 仅认 BULL_ATTACK（红柱放大+DIF上扬），其余多头状态最多给 HOLD
    'macd_buy_whitelist': ['BULL_ATTACK'],

    # 2) DKX 必须连续上行
    'require_dkx_rising': True,
    'dkx_rising_days': 2,     # 当前与前一日 DKX 都需高于再前一日

    # 3) 金叉日量能必须放大（量比 ≥ 1.2），否则 BUY → HOLD
    'require_volume_confirm': True,
    'volume_confirm_ratio': 1.2,
}

# ==================== 持续多头加车模式（中等档） ====================
# 用于覆盖"老金叉但仍在持续走强"的标的
# 所有硬约束必须满足 + 综合分≥min_score 才出 BUY 信号
CONTINUATION = {
    'enabled': True,

    # 入选硬约束
    'max_bias_20': 10.0,          # 当前价与 MA20 偏离 ≤ 10%（拒末段追高）
    'max_drawdown_20': 10.0,      # 近 20 日最大回撤 ≤ 10%
    'min_volume_health': 0.85,    # 近 5 日均量 / 20 日均量 ≥ 0.85
    'max_consecutive_up': 5,      # 近期连涨天数 ≤ 5
    'min_score': 75,              # 持续性综合分门槛（满分 100）

    # 评分子项权重（合计 100）
    'weight_trend_health': 35,    # 趋势健康度：BULL_ATTACK 持续天数
    'weight_position_safety': 25, # 位置安全度：BIAS 越低越好
    'weight_rhythm': 20,          # 节奏合理性：连涨不过 3 天
    'weight_volume_match': 20,    # 量能配合：5/20 量比
}

# ==================== ETF最低门槛 ====================
ETF_MIN_REQUIREMENTS = {
    'aum': 2,                 # 最低规模 2亿元
    'daily_turnover': 50,     # 最低日均成交额 5000万
    'listing_days': 180,      # 最低上市天数 6个月
}

# ==================== 业绩评估周期 ====================
# 用于ETF业绩评分
PERFORMANCE_PERIODS = {
    'short': 20,              # 短期 20日
    'medium': 60,             # 中期 60日 (主要参考)
    'long': 120,              # 长期 120日
}

# 多周期权重
PERIOD_WEIGHTS = {
    'short': 0.2,
    'medium': 0.5,
    'long': 0.3,
}

# ==================== 主线数据分析周期（稳健型）====================
# 固定使用多周期分析，不可调整
THEME_ANALYSIS_PERIODS = {
    'return_short': 20,       # 短期收益 20日
    'return_medium': 60,      # 中期收益 60日
    'etf_flow': 20,           # ETF资金流向 20日（约一个月）
    'northbound': 20,         # 北向资金 20日
    'volatility': 60,         # 波动率 60日
}

# 收益率综合权重
RETURN_WEIGHTS = {
    'short': 0.4,             # 短期权重 40%
    'medium': 0.6,            # 中期权重 60%
}

# ==================== 候选主线题材池 ====================
THEME_POOL = {
    'AI算力': {
        'keywords': ['算力', '光模块', 'CPO', 'AI服务器', '英伟达'],
        'etf_keywords': ['通信', '5G', '芯片', '人工智能'],
    },
    '半导体': {
        'keywords': ['半导体', '芯片', '设备', '材料', '先进制程'],
        'etf_keywords': ['半导体', '芯片', '科创芯片', '集成电路'],
    },
    '人形机器人': {
        'keywords': ['机器人', '减速器', '传感器', '特斯拉'],
        'etf_keywords': ['机器人', '智能制造'],
    },
    '自主可控': {
        'keywords': ['信创', '国产替代', '操作系统', '数据库'],
        'etf_keywords': ['信创', '软件', '计算机'],
    },
    '新能源': {
        'keywords': ['锂电', '储能', '光伏', '新能源车'],
        'etf_keywords': ['新能源', '锂电池', '光伏', '储能'],
    },
    '红利资产': {
        'keywords': ['高股息', '央企', '银行', '公用事业'],
        'etf_keywords': ['红利', '银行', '央企', '国企'],
    },
    '消费复苏': {
        'keywords': ['消费', '白酒', '旅游', '医药'],
        'etf_keywords': ['消费', '食品饮料', '医药', '旅游'],
    },
    '低空经济': {
        'keywords': ['无人机', 'eVTOL', '通航', '低空'],
        'etf_keywords': ['军工', '国防', '航空'],
    },
}

# ==================== 风控参数 ====================
RISK_CONTROL = {
    'max_single_theme_weight': 0.50,   # 单一主线最大仓位 50%
    'max_single_etf_weight': 0.40,     # 单一ETF最大仓位 40%
    'theme_score_threshold': 50,        # 主线评分低于此需减仓
    'stop_loss': -0.15,                 # 止损线 -15%
    'trailing_stop': 0.10,              # 移动止盈回撤 10%
}

# ==================== 出场规则（V7：数据驱动） ====================
# 11K 入场样本回测验证：TIME_20 + SL_10 + TP_15 三合一最优
#   平均 +4.5% / 胜率 64-66% / 风险调整收益 0.40+
# 关键发现：所有"反应式 SELL"（破MA/MACD死叉/健康度）都是反向信号，砍在底部
EXIT_RULES = {
    'time_limit_days': 20,       # 持有满 20 个交易日卖（最优时长）
    'stop_loss_pct': -10.0,      # 浮亏 ≤ -10% 卖（防灾难，不防回调）
    'take_profit_pct': 15.0,     # 浮盈 ≥ +15% 卖（锁定大涨）
    # 三个条件先到先卖，不依赖任何技术信号
}

# ==================== 持仓健康度评分（V5：绝对盈亏主导，0-100） ====================
HOLDING_HEALTH = {
    'base_score': 50,

    # A. 绝对盈亏（主导，±60）—— 赚钱才是硬道理
    'pnl_buckets': [
        (5.0, 30),    # 大赚 +5%以上
        (2.0, 20),    # 赚 +2~5%
        (0.0, 5),     # 微赚 0~2%
        (-2.0, -20),  # 微亏 -2~0%（亏就是亏，明显扣）
        (-5.0, -40),  # 亏 -5~-2%
        (-999, -60),  # 大亏（叠加硬止损必卖）
    ],

    # B. 高点回撤（±25）—— 趋势是否还在
    'drawdown_buckets': [
        (-2.0, 20),   # 接近新高
        (-5.0, 10),   # 浅回撤
        (-8.0, -10),  # 中回撤
        (-12.0, -20), # 深回撤
        (-999, -25),  # 大回撤
    ],

    # C. 价格 vs MA20
    'above_ma20_bonus': 5,
    'below_ma20_penalty': -10,

    # D. 信号状态（±25）—— 辅助证据
    'signal_weights': {
        'dkx_gold': 5,
        'dkx_dead': -10,
        'BULL_ATTACK': 10,
        'BULL_PULLBACK': 0,
        'BULL_SHOCK': 0,
        'BULL_WEAKENING': -10,
        'BEAR_EXHAUST': -8,
        'BEAR_RESIST': -15,
        'BEAR_SHOCK': -18,
        'BEAR_ATTACK': -25,
    },

    # E. 时间×盈亏惩罚 —— 扛着不盈利就该走
    'time_penalty': {
        'days_loss': 5,        # 持有 ≥5 天仍浮亏
        'penalty_loss': -20,
        'days_weak': 8,        # 持有 ≥8 天浮盈仍 <2%
        'penalty_weak': -10,
    },

    # F. 量能（±5）
    'vol_buckets': [
        (1.3, 5),
        (0.8, 0),
        (0, -5),
    ],
}

# ==================== 入场规则（V9：强势回调 PB，walk-forward 验证最优） ====================
# 1349只池/250日样本外: PB 总收益+116% 夏普3.13 回撤-11.9%, 全面优于突破追强(HOLD)和抄底反转(TURN)
# 核心: 相对强势(RS>0,趋势在) + 暂时回调(缩回MA20) + 企稳(站回MA10收阳) + 不追高(BIAS≤12)
PB_ENTRY = {
    'enabled': True,
    'min_rs_20d': 2.0,            # 近20日相对沪深300超额 ≥ 2%（确实相对强）
    'max_bias_20': 12.0,          # BIAS_20 ≤ 12%（不追高）
    'dip_tolerance': 1.03,        # 近4日内缩回到 MA20×1.03 以内（回踩到位）
    'ma20_rising_lookback': 5,    # MA20 较 5 日前上行（中期趋势在）
}

# ==================== 旧入场规则（V7：4信号突破，已被 PB 取代，保留备查） ====================
# 20万次基础信号回测，仅保留胜率 ≥63% + 10日均涨 ≥2% 的：
#   D5 BULL_ATTACK 首日 + 放量(≥1.5)   样本1286, 10日+2.34%, 胜率64%
#   D2 创20日新高 + 放量(≥1.5)         样本2072, 10日+2.51%, 胜率63%
#   A5 红柱递增 3 日                   样本5940, 10日+2.36%, 胜率65%
#   C3 连阳 4 根                       样本2970, 10日+2.98%, 胜率64%
# 撤销 cross_confirm (A1 金叉单独胜率仅 54%) 和 pullback_retest (C8/D6 仅 54-55%)
ENTRY_RULES = {
    'enabled_modes': ['D5', 'D2', 'A5', 'C3'],

    # D5: BULL_ATTACK 首日 + 放量
    'd5_min_vol_ratio_1d': 1.5,

    # D2: 创 N 日新高 + 放量
    'd2_new_high_window': 20,
    'd2_min_vol_ratio_1d': 1.5,

    # A5: 红柱连续递增 N 天（默认 3 天）
    'a5_consecutive_days': 3,

    # C3: 连阳 N 根（默认 4 根）
    'c3_consecutive_up': 4,
}

# ==================== 趋势跟随入场（trend_follow，数据驱动） ====================
# 解决"强势逼空票从不回踩、买不进"的问题
TREND_FOLLOW = {
    'enabled': True,
    'max_bias_20': 15.0,        # 默认 BIAS 上限（被波动率动态值覆盖）
    'min_vol_ratio': 1.0,       # 量能 5日/20日 ≥ 1.0（放量）
    'new_high_window': 20,      # 价格创近 N 日新高
    'trailing_stop_pct': -6.0,  # 趋势跟随标的用更紧的移动止盈（追高需快跑）
}

# ==================== 波动率差异化（弹性） ====================
# 6164 样本分组回测结论（用触发点当时 60 日波动率，与实盘 calc_annualized_vol 一致）：
#   极高波动(≥32%): BIAS>15% 后胜率从64%崩到46%，超买更危险 → 上限 15%
#   高波动(23-32%): BIAS 0-15% 胜率61-73%，20% 才转负 → 上限 20%
#   中低波动(<23%): 拉不出高乖离，15% 够用
# 反直觉但数据明确：高弹性≠放宽阈值。真正的弹性差异在收益 → 评分加溢价
VOLATILITY_BIAS_LIMITS = [
    (32.0, 15.0),   # 60日年化波动率 ≥32% → BIAS 上限 15%
    (23.0, 20.0),   # 23-32% → 20%
    (0.0, 15.0),    # <23% → 15%
]

# 弹性溢价：高波动赛道同等信号下预期收益高，评分加分（直接加到综合分）
ELASTICITY_PREMIUM = {
    'enabled': True,
    'buckets': [
        (32.0, 8.0),    # 极高波动 +8
        (23.0, 5.0),    # 高波动 +5
        (17.0, 2.0),    # 中波动 +2
        (0.0, 0.0),     # 低波动 +0
    ],
}

# ==================== 大盘熔断 ====================
MARKET_CIRCUIT_BREAKER = {
    'enabled': True,
    'index_code': '000300.SH',    # 沪深300
    'max_daily_drop_pct': -1.5,   # 当日跌幅 ≤ -1.5% → 推迟建仓 1 天
}

# ==================== 板块过滤 ====================
# A. 指数去重：用 benchmark 提取真实底层指数，同指数不同壳只留 1 只
#    （根治"日经225华夏+日经225易方达"这种买双份的问题）
# B. 周期类黑名单关闭（标签 = 风险 是教条，应该看实时数据）
SECTOR_FILTERS = {
    'enable_dedup': True,
    'enable_blacklist': False,
    'blacklist_keywords': [],
}

# ==================== 海外敞口上限（这是 A股系统） ====================
# V8：纯裸 K 信号不分国界，海外指数（纳指/日经/恒生）走强时会霸占候选。
# 限制海外/QDII 持仓数量，保证组合以 A股为主。
OVERSEAS_CAP = {
    'enabled': True,
    'max_positions': 1,   # 最多 1 只海外/QDII（约占总仓位 10-16%，≈ ≤20% 敞口）
}

# ==================== 全市场扫描配置 ====================
MARKET_SCAN = {
    'min_aum': 2,             # 最低规模 2 亿
    'min_turnover': 50,       # 最低日均成交额 5000万（单位：百万元）
    'min_listing_days': 90,   # 最低上市 3 个月
    'max_candidates': 30,     # 信号触发后保留 Top N 候选进入精排
    'top_n_buy': 5,           # 最终建议买入数量
    'api_sleep': 0.25,        # API 限流间隔（秒）
}
