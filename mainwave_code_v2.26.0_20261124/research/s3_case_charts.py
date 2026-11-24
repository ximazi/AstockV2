"""S3 (platform breakout: base >= 7 sessions with range < 15%, volume dried then >= 2x, not overextended, no top
divergence) case charts, same page as the S4 one: 10 winners (+15% before the -10% stop within 20 sessions) and 10
losers (the stop actually hit), each marking the buy, the highest point and the stop, with a MACD panel.
Cases: 2025-01..2026-08 signals, holdings not touching 2026-03 / 2026-07, no names now carrying ST, winners at most one
per month, losers at most two, most recent first. Each card also shows the base facts and the LAD exit outcome.
python research/s3_case_charts.py -> results/research/s3_cases/s3_cases.html (+ s3_labelled.csv)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import four_dim_v2 as V2  # noqa: E402
import s4_case_charts as C  # noqa: E402
import s4_feature_discrimination as S  # noqa: E402
import exit_study_s4 as E  # noqa: E402
from mainwave_engine import strong_trend as ST  # noqa: E402

OUT = ROOT / "results/research/s3_cases"


def events():
    v3 = pd.read_csv(ROOT / "results/research/four_dim_v3/events.csv", encoding="utf-8-sig")
    for c in ("shrink", "expand", "not_extended", "diverge"):
        v3[c] = v3[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    s3 = v3[v3.shrink & v3.expand & v3.not_extended & ~v3.diverge]
    return s3[s3.date.between("2025-01-01", "2026-08-24") & ~s3.date.str[:7].isin(C.EXCLUDE)]


def label_all(ev):
    rows = []
    for sym, g in ev.groupby("symbol"):
        raw, f = V2.frame(ROOT / "data" / f"{sym}_daily.csv")
        pos = {d: i for i, d in enumerate(f.date)}
        for r in g.itertuples():
            t = pos.get(r.date)
            lab = S.label(raw, f, t, sym) if t is not None else None
            if lab:
                rows.append({"symbol": sym, "date": r.date, "label": lab, "base_len": r.base_len, "base_amp": r.base_amp,
                             "base_vol_ratio": r.base_vol_ratio, "vol_ratio": r.vol_ratio, "bias_d": r.bias_d})
    return pd.DataFrame(rows)


def card(sym, day, name, row):
    c = C.case(sym, day, name, _Ind(sym, day))
    f, x = E.prepare(sym)
    t = int(f.index[f.date == day][0])
    import accept_failure as AF
    lad = AF.lad(x, t)
    fail, warn = AF.s3_checks(x, t, int(row.base_len))["A1h"]
    lad_a1h = AF.lad(x, t, fail, warn)
    path = "先到 +15%" if row.label == "WIN15" else "先到 +10%" if row.label == "WIN10" else ("先触及 −10%" if c["stop_day"] else "都未触及")
    # three separately named volume ratios (external review 2026-09-26: the first page showed signal/base volume under
    # the name 整理期量比 and signal/prior-20 under 突破日量比)
    v = f.volume.astype(float)
    n = int(row.base_len)
    base_mean = float(v.iloc[t - n:t].mean())
    pre_mean = float(v.iloc[t - n - 20:t - n].mean())
    c["ind"] = {"零成交日(持有窗口)": "、".join(c["zero_volume_days"]) or "无", "整理天数": f"{n}", "整理振幅": f"{row.base_amp * 100:.1f}%",
                "整理均量/整理前20日均量": f"{base_mean / pre_mean:.2f}", "信号日量/整理均量": f"{row.base_vol_ratio:.2f}",
                "信号日量/前20日均量": f"{row.vol_ratio:.2f}",
                "偏离EMA20": f"{row.bias_d * 100:.1f}%", **c["ind"],
                "价格路径（20日）": path,
                "分批止盈结果": f"{lad['ret'] * 100:+.1f}%（{lad['reason']}，{lad['days']} 日）" if lad else "—",
                "分批止盈+否定即走(A1h)": f"{lad_a1h['ret'] * 100:+.1f}%（{lad_a1h['reason']}，{lad_a1h['days']} 日）" if lad_a1h else "—"}
    for k in ("回踩深度", "回踩缩量比", "信号日量比"):
        c["ind"].pop(k, None)
    return c


class _Ind:
    """Indicator row for C.case (T1..Q3), computed the same way as the S4 study."""

    def __init__(self, sym, day):
        raw, f = V2.frame(ROOT / "data" / f"{sym}_daily.csv")
        if "turnover" not in raw:
            full = pd.read_csv(ROOT / "data" / f"{sym}_daily.csv", usecols=["date", "turnover"])
            full["date"] = full.date.astype(str).str[:10]
            raw = raw.merge(full.drop_duplicates("date"), on="date", how="left")
        x = S.indicators(raw, f)
        self.__dict__.update(x.iloc[int(f.index[f.date == day][0])].to_dict())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = label_all(events())
    d.to_csv(OUT / "s3_labelled.csv", index=False, encoding="utf-8-sig")
    d["year"] = d.date.str[:4]
    print(d.groupby(["year", "label"]).size().unstack(fill_value=0).to_string())
    names = ST.names(ROOT)
    wins = C.pick(d[d.label == "WIN15"], names).head(10)
    losers = []
    for r in C.pick(d[d.label == "FAIL"], names, per_month=2).itertuples():
        raw, f = V2.frame(ROOT / "data" / f"{r.symbol}_daily.csv")
        if C.path_points(f, int(f.index[f.date == r.date][0]))[3] is not None:
            losers.append(r)
        if len(losers) == 10:
            break
    data = {"win": [card(r.symbol, r.date, names.get(r.symbol, ""), r) for r in wins.itertuples()],
            "fail": [card(r.symbol, r.date, names.get(r.symbol, ""), r) for r in losers]}
    template = (Path(__file__).resolve().parent / "s4_case_charts_template.html").read_text(encoding="utf-8")
    page = (template.replace("S4 回踩再启动案例图谱", "S3 平台突破案例图谱")
            .replace("S4 回踩 EMA20 再启动：成功与失败案例", "S3 平台突破（整理≥7天、缩量后倍量）：成功与失败案例")
            .replace("2025-01 至 2026-08 的 S4 信号", "2025-01 至 2026-08 的 S3 信号")
            .replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)))
    out = OUT / "s3_cases.html"
    out.write_text(page, encoding="utf-8")
    print(out, len(data["win"]), len(data["fail"]), round(out.stat().st_size / 1024), "KB")
    for k in ("win", "fail"):
        for c in data[k]:
            print(k, c["name"], c["code"], c["signal"], c["peak_pct"], c["ind"]["分批止盈结果"], "|", c["ind"]["分批止盈+否定即走(A1h)"])


if __name__ == "__main__":
    main()
