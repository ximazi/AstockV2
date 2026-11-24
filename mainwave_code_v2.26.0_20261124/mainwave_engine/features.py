# -*- coding: utf-8 -*-
"""九原子特征引擎 (方案 §3; 阈值=第一版制度参数, 修订走 §10 纪律).

2026-08-13 按 Codex 实现审核修订(P1×6/P2×4):
  簇宽硬约束≤1×ATR / 低点序列绑定阻力测试段 / 特征4显式定位最近完整回撤 /
  挖坑只用破位前信息+当时ATR并取最近事件 / 相对收益按日期对齐(缺行业不授予) /
  机构行为补"上涨日换手温和高于回撤日" / 20cm冲击阈值0.60特例 / 150日边界 / 全列校验。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import atr, close_location, limit_pct

REQUIRED = ("open", "high", "low", "close", "preclose", "amount", "turnover", "pct_chg")


class Ctx:
    """单股特征上下文: 一次构建, 各特征共享派生序列。df=尾部≤320根。"""

    def __init__(self, symbol: str, df: pd.DataFrame):
        self.symbol = symbol
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{symbol} 缺列 {missing}")
        self.df = df
        self.n = len(df)
        self.dates = [str(x) for x in df["date"]]
        self.C = df["close"].values
        self.H = df["high"].values
        self.L = df["low"].values
        self.O = df["open"].values
        self.PC = df["preclose"].values
        self.pct = df["pct_chg"].values
        self.turn = pd.to_numeric(df["turnover"], errors="coerce").values
        self.amount = pd.to_numeric(df["amount"], errors="coerce").values
        self.lim = limit_pct(symbol)
        self.atr20 = atr(df, 20)
        self.atr10 = atr(df, 10)
        self.atr60 = atr(df, 60)
        self.cloc = close_location(df)
        c = pd.Series(self.C)
        self.ma20 = c.rolling(20).mean().values
        self.ma60 = c.rolling(60).mean().values
        self.ma120 = c.rolling(120).mean().values

    def swing_lows(self, lo: int, hi: int) -> list[int]:
        return [i for i in range(max(2, lo), min(self.n - 2, hi))
                if self.L[i] == min(self.L[i - 2:i + 3])]

    def swing_highs(self, lo: int, hi: int) -> list[int]:
        return [i for i in range(max(2, lo), min(self.n - 2, hi))
                if self.H[i] == max(self.H[i - 2:i + 3])]


def _impulse_threshold(lim: float) -> float:
    """冲击强度阈值: 10cm=0.70; 20cm=0.60(轮4裁决: +12%强收与主板涨停同功能)。"""
    return 0.60 if lim >= 20 else 0.70


# ── 特征1 板制归一化"冲击-承接" ───────────────────────────
def impulse_retention(ctx: Ctx) -> dict:
    n = ctx.n
    thr = _impulse_threshold(ctx.lim)
    best = None
    for i in range(max(10, n - 61), n - 5):
        strength = ctx.pct[i] / ctx.lim
        if strength < thr or ctx.cloc[i] < 0.60 or not (ctx.L[i] < ctx.H[i]):
            continue
        base_low = np.nanmin(ctx.L[max(0, i - 6):i])
        retained = (ctx.C[-1] >= ctx.PC[i] * (1 + 0.5 * ctx.pct[i] / 100)
                    and ctx.C[-1] > base_low)
        if retained:
            best = {"idx": i, "age": n - 1 - i, "strength": round(float(strength), 2)}
    ok = best is not None
    return {"ok": ok, "score": best["strength"] if ok else 0.0,
            "age": best["age"] if ok else None,
            "event_date": ctx.dates[best["idx"]] if ok else None,
            "detail": f"冲击日T-{best['age']} 强度{best['strength']}" if ok else "无保留良好的历史冲击"}


# ── 特征2 压力收敛 ───────────────────────────────────────
def resistance_cluster(ctx: Ctx) -> dict:
    n = ctx.n
    a = ctx.atr20[-1]
    if not np.isfinite(a) or a <= 0:
        return {"ok": False, "detail": "ATR不可用", "score": 0.0}
    peaks = ctx.swing_highs(n - 41, n - 2)
    clusters: list[list[int]] = []
    for p in sorted(peaks, key=lambda i: ctx.H[i]):
        placed = False
        for cl in clusters:
            prices = [ctx.H[x] for x in cl] + [ctx.H[p]]
            # 簇宽硬约束: 全簇 max-min ≤ 1×ATR (审核修订: 中位数漂移会放宽簇宽)
            if max(prices) - min(prices) <= a:
                cl.append(p); placed = True; break
        if not placed:
            clusters.append([p])
    valid = [cl for cl in clusters if len(cl) >= 2 and (max(cl) - min(cl)) >= 5]
    if not valid:
        return {"ok": False, "detail": "无≥2峰的窄带阻力簇", "score": 0.0}
    valid.sort(key=lambda cl: (-len(cl), float(np.std([ctx.H[x] for x in cl])), -max(cl)))
    cl = valid[0]
    res = float(np.median([ctx.H[x] for x in cl]))
    first_peak_date = ctx.dates[min(cl)]  # 结构出生日期(A型=首个簇峰日)

    # 低点序列绑定阻力测试段: 首个簇峰之后的摆动低点须持续抬高 (审核修订)
    first_peak = min(cl)
    lows = [i for i in ctx.swing_lows(first_peak, n - 2)]
    if len(lows) >= 3:
        hl = ctx.L[lows[-1]] > ctx.L[lows[-2]] >= ctx.L[lows[-3]]
    elif len(lows) == 2:
        hl = ctx.L[lows[-1]] > ctx.L[lows[-2]]
    else:
        hl = False
    contraction = (np.isfinite(ctx.atr60[-1]) and np.isfinite(ctx.atr10[-11]) and ctx.atr60[-11] > 0
                   and (ctx.atr10[-1] / ctx.atr60[-1]) < (ctx.atr10[-11] / ctx.atr60[-11]))
    near = (res - a) <= ctx.C[-1] < (res + 0.2 * a)
    ok = hl and contraction and near
    return {"ok": ok, "resistance": round(res, 2), "resistance_atr": round(float(a), 3),
            "peaks": len(cl), "higher_lows": bool(hl), "contraction": bool(contraction), "near": bool(near),
            "score": float(len(cl)), "first_peak_date": first_peak_date,
            "detail": f"阻力{res:.2f}({len(cl)}峰,簇宽≤1ATR) 抬低点={hl} 收缩={contraction} 贴近={near}"}


def structure_vs_frozen(ctx: Ctx, frozen_res: float, frozen_atr: float,
                        stype: str | None = None) -> bool:
    """在池个股的结构有效性: 冻结锚近距 + 按结构类型的轻量不变量(第6轮复审#7:
    只查价格区间会让'结构已坏但价格还在'的股维持READY)。"""
    C = ctx.C[-1]
    if not ((frozen_res - frozen_atr) <= C < (frozen_res + 0.2 * frozen_atr)):
        return False
    if stype == "A_convergence":   # 低点序列不得走坏
        lows = ctx.swing_lows(ctx.n - 41, ctx.n - 2)
        if len(lows) >= 2 and ctx.L[lows[-1]] < ctx.L[lows[-2]]:
            return False
    elif stype == "C_platform":    # 机构平台的趋势前提仍在
        if not (np.isfinite(ctx.ma60[-1]) and C >= ctx.ma60[-1]
                and ctx.n >= 81 and ctx.ma60[-1] > ctx.ma60[-21]):
            return False
    # B_pit: 坑底失效价守卫已覆盖其核心不变量
    return True


# ── 特征3 挖坑/假破位回收 ────────────────────────────────
def pit_reclaim(ctx: Ctx) -> dict:
    """审核修订: S 与 ATR 只用破位日之前已确认的信息(禁事后重画); 多事件取最近。"""
    n = ctx.n
    best = None
    for u in range(max(50, n - 21), n - 1):
        a_u = ctx.atr20[u]
        if not np.isfinite(a_u) or a_u <= 0:
            continue
        lows_before = ctx.swing_lows(u - 45, u - 2)  # ±2确认 → u-2 前的低点在 u 日已确认
        if len(lows_before) < 2:
            continue
        # "平台下沿"语义: 破位前必须存在【地板簇】(≥2低点落在1×ATR带内);
        # 任意前低不构成平台——否则下跌趋势的每次反弹都被误判为挖坑回收 (2026-08-13回放暴露)。
        floor = [i for i in lows_before
                 if abs(ctx.L[i] - min(ctx.L[j] for j in lows_before)) <= a_u]
        if len(floor) < 2 or (max(floor) - min(floor)) < 5:
            continue
        S = float(min(ctx.L[i] for i in floor))
        if not ctx.L[u] < S - 0.2 * a_u:
            continue
        pit_low = float(np.nanmin(ctx.L[u:min(n, u + 4)]))
        if pit_low < S - 1.5 * a_u:
            continue
        base_turn = np.nanmedian(ctx.turn[max(0, u - 20):u])
        for k in range(0, 4):
            j = u + k
            if j >= n:
                break
            if (ctx.C[j] > S + 0.2 * a_u and ctx.cloc[j] >= 0.70
                    and np.isfinite(base_turn) and base_turn > 0
                    and ctx.turn[j] >= 1.2 * base_turn):
                highs_before = ctx.swing_highs(u - 45, u - 2)  # 坑前平台上沿(结构B的冻结锚)
                ceiling = round(float(max(ctx.H[i] for i in highs_before)), 2) if highs_before else None
                best = {"ok": True, "score": 1.0, "age": n - 1 - j,
                        "pit_invalid": round(pit_low - 0.2 * a_u, 2),
                        "anchor": ceiling, "event_date": ctx.dates[j],
                        "start_date": ctx.dates[u],  # 结构出生日期(B型=破位日)
                        "detail": f"挖坑@T-{n-1-u} 收回@T-{n-1-j} 坑底{pit_low:.2f}"}
                break  # 该 undercut 已确认收回, 继续扫更近的事件
    if best:
        return best
    return {"ok": False, "detail": "无挖坑回收", "score": 0.0}


# ── 结构族 (v1.1, 方案§4 P0-2): 三选一, 各自冻结锚 ─────────
def inst_platform(ctx: Ctx) -> dict:
    """结构C 机构平台/旗形: 站上MA60+MA60上斜+低波, 近30日窄平台(≤4×ATR)且价处上半部。"""
    n = ctx.n
    a = ctx.atr20[-1]
    if n < 80 or not np.isfinite(a) or a <= 0 or not np.isfinite(ctx.ma60[-1]):
        return {"ok": False, "detail": "历史/ATR不足", "score": 0.0}
    C = ctx.C[-1]
    slope60 = ctx.ma60[-1] - ctx.ma60[-21] if n >= 81 else -1
    atr_pct = ctx.atr20 / ctx.C
    hist = atr_pct[-120:]
    vol_q = float(np.mean(hist[np.isfinite(hist)] <= atr_pct[-1]))
    top = float(np.nanmax(ctx.H[-30:]))
    bot = float(np.nanmin(ctx.L[-30:]))
    ok = (C >= ctx.ma60[-1] and slope60 > 0 and vol_q <= 0.40
          and (top - bot) <= 4 * a and C >= (top + bot) / 2)
    return {"ok": bool(ok), "score": round(1 - vol_q, 2), "anchor": round(top, 2),
            "invalid": round(bot - 0.3 * a, 2),
            "detail": f"平台[{bot:.2f},{top:.2f}] 波动分位{vol_q:.2f} MA60上斜={slope60 > 0}"}


def structure_family(ctx: Ctx, rc: dict, pit: dict, ip: dict) -> dict:
    """结构支柱=结构族三选一(优先级 A收敛>B挖坑>C平台), 统一近距条件 [锚-1×ATR, 锚+0.2×ATR)。
    返回冻结要素: type/anchor/atr/invalid。"""
    a = ctx.atr20[-1]
    C = ctx.C[-1]
    if not np.isfinite(a) or a <= 0:
        return {"ok": False, "type": "", "detail": "ATR不可用"}
    def near(anchor: float) -> bool:
        return (anchor - a) <= C < (anchor + 0.2 * a)
    if rc.get("ok"):  # A: rc.near 已含近距
        lows = ctx.swing_lows(ctx.n - 41, ctx.n - 2)
        inv = round(float(ctx.L[lows[-1]] - 0.3 * a), 2) if lows else round(C * 0.92, 2)
        return {"ok": True, "type": "A_convergence", "anchor": rc["resistance"],
                "atr": rc["resistance_atr"], "invalid": inv,
                "started_at": rc.get("first_peak_date"), "detail": rc["detail"]}
    if pit.get("ok") and pit.get("anchor") is not None and (pit.get("age") or 99) <= 20 and near(pit["anchor"]):
        return {"ok": True, "type": "B_pit", "anchor": pit["anchor"],
                "atr": round(float(a), 3), "invalid": pit["pit_invalid"],
                "started_at": pit.get("start_date"), "detail": pit["detail"]}
    if ip.get("ok") and near(ip["anchor"]):
        return {"ok": True, "type": "C_platform", "anchor": ip["anchor"],
                "atr": round(float(a), 3), "invalid": ip["invalid"],
                "started_at": ctx.dates[max(0, ctx.n - 30)], "detail": ip["detail"]}
    return {"ok": False, "type": "", "detail": "三结构族均未成立或未贴近锚位"}


# ── 新鲜冲击 (v1.2裁决#2: "游资来没来"看昨日涨停即可, 与"试盘成功"的承接特征分开) ──
def fresh_impulse(ctx: Ctx) -> dict:
    """近1~4日出现强冲击(≥板制阈值, 可交易): 只作【路径签名】(游资参与证据),
    不进承接特征评分(那需要5日以上消化验证), 两者机制不同不算重复计票。"""
    n = ctx.n
    thr = _impulse_threshold(ctx.lim)
    for k in range(1, 5):
        i = n - 1 - k
        if i < 1:
            break
        if ctx.pct[i] / ctx.lim >= thr and ctx.L[i] < ctx.H[i] and ctx.cloc[i] >= 0.60:
            return {"ok": True, "age": k, "event_date": ctx.dates[i],
                    "detail": f"T-{k}日强冲击(强度{ctx.pct[i]/ctx.lim:.2f})"}
    return {"ok": False, "detail": "近4日无强冲击"}


# ── 相对行业对齐工具 (审核修订: 按日期join, 缺行业不授予) ──
def align_rel(ctx: Ctx, ind_df: pd.DataFrame | None, k: int = 20) -> np.ndarray | None:
    """返回按日期对齐的 (个股-行业) 日收益差, 长度k; 对齐不足返回 None。"""
    if ind_df is None or ind_df.empty:
        return None
    ind = dict(zip(ind_df["date"].astype(str), ind_df["ret"]))
    pairs = [(ctx.pct[i], ind.get(ctx.dates[i])) for i in range(ctx.n - k, ctx.n)]
    vals = [(s - r) for s, r in pairs if r is not None and np.isfinite(s)]
    return np.array(vals) if len(vals) >= max(15, k * 3 // 4) else None


# ── 特征4 推进放量/回撤缩量非对称 ─────────────────────────
def volume_asymmetry(ctx: Ctx, rel20: np.ndarray | None) -> dict:
    n = ctx.n
    if rel20 is None:
        return {"ok": False, "score": 0.0, "detail": "行业基准缺失/对齐不足, 不授予"}
    t = ctx.turn[-len(rel20):]
    adv, ret = t[rel20 > 0], t[rel20 < 0]
    if len(adv) < 5 or len(ret) < 5:
        return {"ok": False, "score": 0.0, "detail": "推进/回撤样本不足"}
    ratio = float(np.nanmedian(adv) / max(np.nanmedian(ret), 1e-9))
    # 显式定位最近一次【完整】2-5日回撤 (审核修订: 无合格回撤不授予)
    c = ctx.C
    pull = None
    for end in range(n - 2, n - 11, -1):        # 回撤结束日(其后一日转涨=完整)
        if c[end + 1] <= c[end]:
            continue
        start = end
        while start - 1 >= 0 and c[start - 1] > c[start]:
            start -= 1
        span = end - start + 1
        if 2 <= span <= 5:
            pull = (start, end)
            break
    if pull is None:
        return {"ok": False, "score": round(ratio, 2), "detail": "近10日无完整2-5日回撤, 不授予"}
    s_, e_ = pull
    turn_down = ctx.turn[e_] < ctx.turn[s_] and np.nanmedian(ctx.turn[s_:e_ + 1]) < np.nanmedian(ctx.turn[max(0, s_ - 5):s_])
    lows_ok = True
    prior_lows = ctx.swing_lows(n - 41, s_)
    if prior_lows:
        lows_ok = min(c[s_:e_ + 1]) >= ctx.L[prior_lows[-1]]
    ok = ratio >= 1.2 and turn_down and lows_ok
    return {"ok": ok, "score": round(ratio, 2),
            "detail": f"换手比{ratio:.2f} 回撤T-{n-1-e_}~T-{n-1-s_}缩量={turn_down} 未破低={lows_ok}"}


# ── 特征5 窄幅充分换手后供给枯竭 ──────────────────────────
def _weighted_quantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[m], w[m]
    if len(v) == 0:
        return np.nan
    idx = np.argsort(v)
    v, w = v[idx], w[idx]
    cw = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cw, v))


def silent_supply(ctx: Ctx) -> dict:
    n = ctx.n
    if n < 150:  # 规格边界150 (审核P2修订)
        return {"ok": False, "score": 0.0, "detail": "历史<150日不授予"}
    cum30 = float(np.nansum(ctx.turn[-30:]))
    hist = [float(np.nansum(ctx.turn[j - 30:j])) for j in range(max(30, n - 250), n - 30)]
    if len(hist) < 60:
        return {"ok": False, "score": 0.0, "detail": "历史窗不足"}
    pr = float(np.mean([h <= cum30 for h in hist]))
    tp = (ctx.H[-30:] + ctx.L[-30:] + ctx.C[-30:]) / 3
    p20 = _weighted_quantile(tp, ctx.turn[-30:], 0.20)
    p80 = _weighted_quantile(tp, ctx.turn[-30:], 0.80)
    a = ctx.atr20[-1]
    narrow = np.isfinite(p20) and (p80 - p20) <= 3 * a
    upper = ctx.C[-1] >= (p20 + p80) / 2
    in_zone = ctx.C[-1] <= p80 + a
    recent_turn = ctx.turn[-5:]
    prior_turn = ctx.turn[-25:-5]
    drying = bool(
        np.isfinite(recent_turn).any()
        and np.isfinite(prior_turn).any()
        and np.nanmedian(recent_turn) <= 0.70 * np.nanmedian(prior_turn)
    )
    ok = (pr >= 0.65) and narrow and upper and in_zone and drying
    return {"ok": ok, "score": round(pr, 2),
            "detail": f"30日换手分位{pr:.2f} 窄带={narrow} 上半部={upper} 枯竭={drying}"}


# ── 特征7 梯队传导(最小实现, 审核修订: 原分支不可达) ───────
def relay_response(ctx: Ctx, ignition: dict | None, mkt_med: dict) -> dict:
    """ignition = {"date": 龙头点火日, "leader": sym} (由 daily_run 用前一日RS判龙头);
    候选响应 = 点火日起1-3日内, 某日跑赢市场中位 且 收位≥0.60。"""
    if not ignition:
        return {"ok": False, "score": 0.0, "detail": "板块近5日无龙头点火"}
    try:
        d0 = ctx.dates.index(ignition["date"])
    except ValueError:
        return {"ok": False, "score": 0.0, "detail": "点火日不在本股日历"}
    for j in range(d0 + 1, min(ctx.n, d0 + 4)):  # 复审#8: 点火后1~3日, 不含T+0
        med = mkt_med.get(ctx.dates[j])
        if med is not None and ctx.pct[j] > med and ctx.cloc[j] >= 0.60:
            return {"ok": True, "score": 1.0, "age": ctx.n - 1 - j, "event_date": ctx.dates[j],
                    "detail": f"龙头{ignition.get('leader','')}点火后T+{j-d0}同步收强"}
    return {"ok": False, "score": 0.0, "detail": "龙头点火后1-3日无同步收强"}


# ── 特征9 基本面锚 + 机构行为 ────────────────────────────
def fundamental_anchor(view: dict | None) -> dict:
    """规格原版(§3特征9): 最新已披露季 单季营收+利润增速双正, 且至少一项环比上一已披露季加速;
    仅利润加速而营收不跟随=弱证据(score降半)。view 来自 data.fundamental_view(点时)。"""
    if not view:
        return {"ok": False, "score": 0.0, "detail": "无已披露季度数据"}
    la, pv = view["latest"], view["prev"]
    qs, qp = la.get("q_sales_yoy"), la.get("q_profit_yoy")
    if pd.isna(qs) or pd.isna(qp):
        qs, qp = la.get("or_yoy"), la.get("netprofit_yoy")
    both_pos = pd.notna(qs) and pd.notna(qp) and qs > 0 and qp > 0
    if not both_pos:
        return {"ok": False, "score": 0.0, "detail": f"非双正(营收{qs}/利润{qp})"}
    if pv is None:
        return {"ok": False, "score": 0.0, "detail": "无上季可比, 加速无法判定"}
    acc_s = pd.notna(pv.get("q_sales_yoy")) and qs > pv["q_sales_yoy"]
    acc_p = pd.notna(pv.get("q_profit_yoy")) and qp > pv["q_profit_yoy"]
    if not (acc_s or acc_p):
        return {"ok": False, "score": 0.0, "detail": f"双正但无环比加速({qs:+.0f}%/{qp:+.0f}%)"}
    weak = acc_p and not acc_s
    return {"ok": True, "score": 0.5 if weak else 1.0,
            "detail": f"季报{int(la['end_date'])}: 营收{qs:+.0f}%{'↑' if acc_s else ''} 利润{qp:+.0f}%{'↑' if acc_p else ''}"
                      + (" (弱:仅利润加速)" if weak else "")}


def institutional_behavior(ctx: Ctx, rel20: np.ndarray | None) -> dict:
    n = ctx.n
    if n < 140:
        return {"ok": False, "score": 0.0, "detail": "历史不足"}
    if rel20 is None:
        return {"ok": False, "score": 0.0, "detail": "行业基准缺失, 不授予"}
    rel = np.cumsum(rel20)
    slope = float(np.polyfit(np.arange(len(rel)), rel, 1)[0])
    atr_pct = ctx.atr20 / ctx.C
    cur = atr_pct[-1]
    hist = atr_pct[-120:]
    vol_q = float(np.mean(hist[np.isfinite(hist)] <= cur))
    max_imp = float(np.nanmax(ctx.pct[-20:]) / ctx.lim)
    # 审核修订: 补规格"上涨日换手温和高于回撤日"(1.0~2.0温和带, 排除爆量)
    t = ctx.turn[-len(rel20):]
    adv, ret = t[rel20 > 0], t[rel20 < 0]
    gentle = False
    if len(adv) >= 5 and len(ret) >= 5:
        r = float(np.nanmedian(adv) / max(np.nanmedian(ret), 1e-9))
        gentle = 1.0 <= r <= 2.0
    ok = slope > 0 and vol_q <= 0.50 and max_imp < 0.70 and gentle
    return {"ok": ok, "score": round(slope, 3),
            "detail": f"相对斜率{slope:+.3f} 波动分位{vol_q:.2f} 最大冲击{max_imp:.2f} 温和吸筹={gentle}"}


# ── 公共健康 + 派发签名 ──────────────────────────────────
def distribution_signature(ctx: Ctx, i: int) -> bool:
    if i < 25:
        return False
    new_high = ctx.H[i] > np.nanmax(ctx.H[i - 20:i])
    touched = ctx.pct[i] >= ctx.lim * 0.85 or ctx.H[i] >= ctx.PC[i] * (1 + ctx.lim / 100) * 0.998
    med = np.nanmedian(ctx.amount[max(0, i - 20):i])
    climax = np.isfinite(med) and med > 0 and ctx.amount[i] >= 2.5 * med
    weak = ctx.cloc[i] <= 0.35
    rng = max(ctx.H[i] - ctx.L[i], 1e-9)
    shadow = (ctx.H[i] - ctx.C[i]) / rng >= 0.40
    return bool((new_high or touched) and climax and weak and shadow)


def common_health(ctx: Ctx, run_date: str) -> dict:
    ok_date = str(ctx.df["date"].iloc[-1]) == run_date
    med_amt = float(np.nanmedian(ctx.amount[-20:]))
    liquid = np.isfinite(med_amt) and med_amt >= 3e7
    recent_dist = any(distribution_signature(ctx, ctx.n - 1 - k) for k in range(0, 3))
    ok = ok_date and liquid and not recent_dist
    return {"ok": ok, "detail": f"当日有数={ok_date} 流动性={liquid} 近3日派发={recent_dist}"}
