"""Case pages for S1 / S2 / S5 in the S3/S4 format: 10 winners (+15% before the -10% stop within 20 sessions) and 10
losers (the stop actually hit), K-line + EMA20/MA60 + MACD, buy / high / stop marked. Each card keeps the price-path
label apart from the trade result under the staged profit-taking exit (LAD, T+1), as the external review asked.
  S1 新结构突破     core events with entry_type BREAKOUT (case_pool event_domain + four_dim_v2 events)
  S2 调整后再转强   core events with entry_type REACCELERATION
  S5 底部反转       scenario_catalog BOTTOM_REVERSAL, strict variant (MA20 > MA60), user decision 2026-09-24
Cases: signals 2025-01..2026-08, holdings not touching 2026-03 / 2026-07, no names now carrying ST, most recent
first, winners at most one per month (S5: two, small sample), losers at most two.
python research/scenario_case_charts.py S1|S2|S5 -> results/research/scenario_cases/<s>_cases.html
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import accept_failure as AF  # noqa: E402
import exit_study_s4 as E  # noqa: E402
import four_dim_v2 as V2  # noqa: E402
import s3_case_charts as C3  # noqa: E402
import s4_case_charts as C  # noqa: E402
import s4_feature_discrimination as S  # noqa: E402
from mainwave_engine import strong_trend as ST  # noqa: E402

OUT = ROOT / "results/research/scenario_cases"
SPEC = {
    "S1": ("S1 新结构突破", "收盘突破此前 20 日最高价（新结构），强势趋势核心的突破入口（CT20）"),
    "S2": ("S2 调整后再转强", "调整之后重新转强（RA3），强势趋势核心的再加速入口"),
    "S5": ("S5 底部反转（MA20>MA60）", "此前 120 日高点跌 ≥20%，下跌放缓、缩量、窄幅整理后，MACD 在零轴附近金叉、放量 ≥1.3 倍、站上 MA20，且 MA20>MA60"),
}


def events(kind):
    if kind == "S5":
        b = pd.read_csv(ROOT / "results/research/scenarios_2025_2026/bottom_reversal.csv", encoding="utf-8-sig")
        ev = b[b.scenario.str.contains("MA20>MA60", regex=False)][["symbol", "date"]]
    else:
        v2 = pd.read_csv(ROOT / "results/research/four_dim_v2/events_excess.csv", encoding="utf-8-sig")
        pools = [pd.read_csv(ROOT / f"results/research/case_pool/20260923{tag}/event_domain.csv", encoding="utf-8-sig",
                             usecols=["symbol", "date", "entry_type"]) for tag in ("_2021", "")]
        kinds = pd.concat(pools).drop_duplicates(["symbol", "date"])
        v2 = v2.merge(kinds, on=["symbol", "date"], how="left")
        core = v2.core_event.astype(str).str.lower().isin(["true", "1", "1.0"])
        ev = v2[core & (v2.entry_type == ("BREAKOUT" if kind == "S1" else "REACCELERATION"))][["symbol", "date"]]
    ev = ev.drop_duplicates()
    return ev[ev.date.between("2025-01-01", "2026-08-24") & ~ev.date.str[:7].isin(C.EXCLUDE)]


def label_all(ev):
    rows = []
    for sym, g in ev.groupby("symbol"):
        try:
            raw, f = V2.frame(ROOT / "data" / f"{sym}_daily.csv")
        except (OSError, ValueError, KeyError):
            continue
        pos = {d: i for i, d in enumerate(f.date)}
        for d in g.date:
            t = pos.get(d)
            lab = S.label(raw, f, t, sym) if t is not None and t > 60 else None
            if lab:
                rows.append({"symbol": sym, "date": d, "label": lab})
    return pd.DataFrame(rows)


def card(sym, day, name, label):
    c = C.case(sym, day, name, C3._Ind(sym, day))
    f, x = E.prepare(sym)
    t = int(f.index[f.date == day][0])
    lad = AF.lad(x, t)
    after = "，之后又触及 −10%" if c["stop_day"] else ""
    path = ("先到 +15%" + after if label == "WIN15" else "先到 +10%" + after if label == "WIN10"
            else "先触及 −10%" if c["stop_day"] else "都未触及")
    c["ind"] = {"零成交日(持有窗口)": "、".join(c["zero_volume_days"]) or "无", "价格路径（20日）": path,
                "交易结果·分批止盈": f"{lad['ret'] * 100:+.1f}%（{lad['reason']}，{lad['days']} 日）" if lad else "—",
                "收盘高出MA60": f"{(float(f.close.iloc[t]) / float(f.close.rolling(60).mean().iloc[t]) - 1) * 100:.1f}%",
                **c["ind"]}
    for k in ("回踩深度", "回踩缩量比"):
        c["ind"].pop(k, None)
    return c


def main(kind):
    title, definition = SPEC[kind]
    OUT.mkdir(parents=True, exist_ok=True)
    d = label_all(events(kind))
    d.to_csv(OUT / f"{kind.lower()}_labelled.csv", index=False, encoding="utf-8-sig")
    d["year"] = d.date.str[:4]
    print(kind, d.groupby(["year", "label"]).size().unstack(fill_value=0).to_string())
    names = ST.names(ROOT)
    wins = C.pick(d[d.label == "WIN15"], names, per_month=2 if kind == "S5" else 1).head(10)
    losers = []
    for r in C.pick(d[d.label == "FAIL"], names, per_month=2).itertuples():
        raw, f = V2.frame(ROOT / "data" / f"{r.symbol}_daily.csv")
        if C.path_points(f, int(f.index[f.date == r.date][0]))[3] is not None:
            losers.append(r)
        if len(losers) == 10:
            break
    data = {"win": [card(r.symbol, r.date, names.get(r.symbol, ""), r.label) for r in wins.itertuples()],
            "fail": [card(r.symbol, r.date, names.get(r.symbol, ""), r.label) for r in losers]}
    template = (Path(__file__).resolve().parent / "s4_case_charts_template.html").read_text(encoding="utf-8")
    page = (template.replace("S4 回踩再启动案例图谱", f"{title}案例图谱")
            .replace("S4 回踩 EMA20 再启动：成功与失败案例", f"{title}：成功与失败案例")
            .replace("2025-01 至 2026-08 的 S4 信号", f"2025-01 至 2026-08 的 {kind} 信号（{definition}）")
            .replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)))
    out = OUT / f"{kind.lower()}_cases.html"
    out.write_text(page, encoding="utf-8")
    print(out, len(data["win"]), len(data["fail"]), round(out.stat().st_size / 1024), "KB")
    for k in ("win", "fail"):
        for c in data[k]:
            print(k, c["name"], c["code"], c["signal"], f"最高{c['peak_pct']}", c["ind"]["价格路径（20日）"], "|",
                  c["ind"]["交易结果·分批止盈"], "| MA60", c["ind"]["收盘高出MA60"])


if __name__ == "__main__":
    main(sys.argv[1])
