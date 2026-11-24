"""Headless PNG exports of frozen case evidence, independent of browser zoom."""
from __future__ import annotations

import io
import json
from threading import Lock

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mplfinance.original_flavor import candlestick_ohlc

VERSION = "review_chart_v3_pv_shadow"
RENDER_LOCK = Lock()
STYLES = {
    "TARGET_IGNITION": ("点火代理", "*", "#a06600"),
    "THEME_IGNITION": ("题材点火", "D", "#326a8b"),
    "BUY": ("回测买入", "^", "#c52e3b"),
    "SELL": ("回测卖出", "v", "#11734c"),
}
SHORT_LABELS = {"FAILED": "结构失效", "REBUILDING": "修复", "PULLBACK": "整理",
                "IGNITION": "启动", "MARKUP": "上涨", "DISTRIBUTION_RISK": "派发风险", "OVERHEATED": "过热"}


def chart_window(case, focused=False):
    bars = case["view"]["chart"]["bars"]
    if not bars:
        raise ValueError("案例缺少K线，不能导出空图")
    if not focused:
        return bars
    dates = [bar["date"] for bar in bars]
    events = case["view"]["case_review"]["events"]
    indices = [dates.index(event["date"]) for event in events if event["date"] in dates]
    if not indices:
        return bars
    # All actual events remain visible; include 25 sessions of context on both sides.
    return bars[max(0, min(indices) - 25):min(len(bars), max(indices) + 26)]


def chinese_font():
    for family in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        try:
            return FontProperties(fname=findfont(FontProperties(family=family), fallback_to_default=False))
        except ValueError:
            continue
    raise ValueError("K线导出需要中文字体：Microsoft YaHei / Noto Sans CJK SC / SimHei")


def figure_for(case, focused=False, observation=None):
    bars = chart_window(case, focused)
    dates = [bar["date"] for bar in bars]
    n = len(bars)
    positions = {day: i for i, day in enumerate(dates)}
    events = [e for e in case["view"]["case_review"]["events"] if e["date"] in positions and e["kind"] in STYLES]
    font = chinese_font()
    fig = Figure(figsize=(16, 9), dpi=160, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((.06, .285, .915, .535))
    band = fig.add_axes((.06, .845, .915, .032))
    volume = fig.add_axes((.06, .135, .915, .10), sharex=ax)
    for panel in (ax, volume):
        panel.set_axisbelow(True)
        panel.grid(color="#e3e7e4", linewidth=.65)
        for spine in panel.spines.values():
            spine.set_visible(False)
        panel.tick_params(axis="both", length=0, labelsize=10, colors="#4e5752")
    low = min(b["low"] for b in bars)
    high = max(b["high"] for b in bars)
    high = max([high] + [e["chart_price"] for e in events])
    low = min([low] + [e["chart_price"] for e in events])
    spread = max(high - low, high * .03)

    # Pack labels left-to-right above prices; arrows never move the event itself.
    label_width = max(n, 3) * .11
    labels = []
    ordered = sorted(events, key=lambda e: (e["date"], e["kind"]))
    lane_count = (len(ordered) + 7) // 8
    for offset in range(0, len(ordered), 8):
        group = ordered[offset:offset + 8]
        centers = []
        for event in group:
            center = max(label_width / 2 - .5, positions[event["date"]])
            centers.append(max(center, centers[-1] + label_width) if centers else center)
        overflow = max(0, centers[-1] - (n - .5 - label_width / 2))
        labels.extend((event, center - overflow, offset // 8) for event, center in zip(group, centers))
    lane_height = spread * .19
    ax.set_ylim(low - spread * .05, high + lane_height * (lane_count + .7))
    for segment in case["view"]["chart"].get("stage_segments", []):
        included = [i for i, day in enumerate(dates) if segment["start"] <= day <= segment["end"]]
        if not included:
            continue
        start, end = included[0] - .5, included[-1] + .5
        color = segment["color"]
        ax.axvspan(start, end, facecolor=color, edgecolor=color, alpha=.25, linewidth=.6, zorder=0)
        band.axvspan(start, end, color=color, ec="white", lw=.5)
        label = SHORT_LABELS.get(segment["stage"], segment["label"])
        if (end - start) / max(n, 1) * 14.64 * 72 > len(label) * 10 + 8:
            band.text((start + end) / 2, .5, label, ha="center", va="center", fontproperties=font,
                      fontsize=10, color="#1b2420" if segment["stage"] in {"PULLBACK", "OVERHEATED"} else "white")
    band.set(xlim=(-.5, n - .5), ylim=(0, 1))
    band.set_axis_off()
    candlestick_ohlc(ax, [(i, b["open"], b["high"], b["low"], b["close"]) for i, b in enumerate(bars)],
                     width=.62, colorup="#b83a3a", colordown="#16734a", alpha=1)
    handles = [Patch(color="#b83a3a", label="价格")]
    for field, color in (("ema10", "#94959b"), ("ema20", "#d28b2f"), ("ema60", "#326a8b")):
        ax.plot(range(n), [b.get(field) for b in bars], color=color, lw=1.15, zorder=3)
        handles.append(Line2D([], [], color=color, label=field.upper()))
    volume.bar(range(n), [b["amount"] for b in bars], color="#bcc7c0", width=.7)
    volume.set_ylabel("成交额(亿)", fontproperties=font, fontsize=10)
    if observation and observation.get("timeline"):
        by_date = {r["date"]: r for r in observation["timeline"]}
        pv_band = fig.add_axes((.06, .09, .915, .016))
        for i, day in enumerate(dates):
            state = by_date.get(day, {}).get("state", "UNRESOLVED")
            pv_band.axvspan(i - .5, i + .5, color=observation["colors"][state], lw=0)
        pv_band.set(xlim=(-.5, n - .5), ylim=(0, 1))
        pv_band.set_axis_off()
        fig.text(.06, .112, "量价观察（待校准）：灰=未明确  青=低量上行  黄=分歧  红=供给压力  蓝=低量下行  深灰=流动性异常",
                 fontproperties=font, fontsize=9, color="#56635b")
    for event, center, lane in labels:
        label, marker, color = STYLES[event["kind"]]
        x, y = positions[event["date"]], event["chart_price"]
        ax.axvline(x, color=color, lw=.8, ls="--", alpha=.5, zorder=2)
        ax.scatter([x], [y], marker=marker, s=125, c=color, edgecolors="white", linewidths=1.1, zorder=6)
        price = f"\n原价 {event['raw_price']:.3f}" if event.get("raw_price") is not None else ""
        if event["kind"] == "SELL" and sum(e["kind"] == "SELL" for e in events) > 1:
            price += f"\n原始仓位 {event['weight']:.0%}"
        ax.annotate(f"{label}\n{event['date']}{price}", xy=(x, y),
                    xytext=(center, high + lane_height * (lane + .4)), textcoords="data",
                    ha="center", va="bottom", fontproperties=font, fontsize=10, color=color,
                    bbox={"boxstyle": "square,pad=.35", "fc": "white", "ec": color, "lw": .8},
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1}, zorder=7)
    for kind in dict.fromkeys(e["kind"] for e in events):
        label, marker, color = STYLES[kind]
        handles.append(Line2D([], [], color=color, marker=marker, linestyle="None", markersize=8, label=label))
    ticks = sorted({round(i * (n - 1) / min(7, max(n - 1, 1))) for i in range(min(8, n))})
    ax.set_xlim(-.5, n - .5)
    ax.set_xticks(ticks, [dates[i] for i in ticks])
    volume.tick_params(axis="x", labelbottom=False)
    legend_font = font.copy()
    legend_font.set_size(10)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.058, .912), ncol=8,
               frameon=False, prop=legend_font, handlelength=2, columnspacing=2)
    s = case["summary"]
    caption = "买卖区间放大" if focused else "全周期观察"
    fig.text(.06, .975, f"{s['name']}  {s['symbol']}  |  {s['route']}  |  {caption}",
             fontproperties=font, fontsize=18, fontweight="bold", va="top", color="#202421")
    trade = case["source_row"]
    net = trade.get("net")
    outcome = f"单笔回测净价差 {net:+.2f}%" if net is not None else "尚无已实现净价差"
    fig.text(.06, .935, f"{dates[0]} 至 {dates[-1]}   |   信号 {s['signal_date']}   |   {s.get('exit_policy', '')}   |   {outcome}",
             fontproperties=font, fontsize=10, color="#56635b", va="top")
    fig.text(.06, .066, "K线/箭头：窗口末锚定复权价格；标注原价：当时原始价格。两者不必相等。退出后行情仍保留。",
             fontproperties=font, fontsize=10, color="#56635b")
    fig.text(.06, .033, f"独立历史回测，非实盘成交或买卖建议。  案例 {s['id']}  |  SHA256 {case['case_hash'][:16]}",
             fontproperties=font, fontsize=9, color="#68756d")
    return fig


def chart_files(case, observation=None):
    files, manifest = {}, {"version": VERSION, "case_hash": case["case_hash"], "images": []}
    if observation:
        manifest["price_volume_version"] = observation["version"]
        manifest["price_volume_code_sha256"] = observation["code_sha256"]
    # Matplotlib's shared font/cache internals are serialized across Flask threads.
    with RENDER_LOCK:
        for focused, name in ((False, "kline_full.png"), (True, "kline_trade_focus.png")):
            fig = figure_for(case, focused, observation)
            try:
                stream = io.BytesIO()
                fig.savefig(stream, format="png", dpi=160, metadata={"CaseSHA256": case["case_hash"], "Renderer": VERSION})
                files[name] = stream.getvalue()
            finally:
                fig.clear()
            bars = chart_window(case, focused)
            manifest["images"].append({"file": name, "width": 2560, "height": 1440,
                                       "start": bars[0]["date"], "end": bars[-1]["date"], "sessions": len(bars)})
    files["chart_exports.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    return files
