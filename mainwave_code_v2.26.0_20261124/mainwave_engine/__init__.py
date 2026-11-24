# mainwave_engine - lifecycle analysis engine v2.26.0
# v2.26.0: independent board-relay watch and opening-auction observations.
# v2.25.0: frozen opportunity/ranking registry and separate fill-driven paper account.
# v2.24.0: dated partial-evidence pool and independent ignition/EMA10 late-session tactics.
# v2.23.2: PowerShell-compatible route summaries, including legacy empty category names.
# v2.23.1: single after-close entry with consolidated step health and paper-risk summary.
# v2.23.0: manual close-price paper positions and entry-anchored daily risk review.
# v2.22.1: newest-first unreviewed cases, retaining reviewed-first priority.
# v2.22.0: read-only price/volume sequences, daily diagnostics and frozen-case attachments.
# v2.21.1: review case route filters and data-driven route labels.
# v2.21.0: formal close-confirmed exits, fixed hard floor and conditional extension.
# v2.20.0: formal half-budget initial entry and conditional pullback-reclaim adds.
# v2.19.0: causal retest research, phase-aware exit attribution and review attachments.
# v2.18.2: signal-level review status and reviewed/loss-first case ordering.
# v2.18.1: one hindsight-selected exit per review signal, preserving archived reviews.
# v2.18.0: full annual review catalog and background bulk case/review exports.
# v2.17.2: reproducible losing-case sampling with immutable review archives.
# v2.17.1: case exports include full-cycle and annotated trade-focus PNG charts.
# v2.17.0: frozen backtest review workbench, portable evidence exports and versioned human reviews.
# v2.16.0: stable preparation tracks, explicit entry contracts and diagnostic exit/calendar audits.
# v2.15.0: independent breakout restart, shared historically studied route evidence and daily full-universe scan.
# v2.14.1: provenance-aware discovery audit cases with continuity gaps and controls.
# v2.14.0: causal breakout links, untruncated discovery, versioned evidence evolution.
# v2.13.0: independent market context and shared explainable analysis core.
# v2.12.0: causal adjustment setups, asymmetric speed evidence and historical case viewer.
# v2.11.0: continuous stock-cycle observations and phase-aware relative turnover.
# v2.10.1: persistent exact-event reverse-trace cache and event-date membership.
# v2.10.0: slow-carrier lifecycle and weak-market mean-reversion shadow overlays.
# v2.9.0: daily dynamic theme mapping, multi-theme display, and expansion monitoring.
# v2.8.1: incremental weekly ignition panel and event replay caching.
# v2.8.0: categorical candidate value ranking and single-stock lifecycle analysis.
# v2.7.0: historically validated phase/event observations in the weekly candidate report.
# v2.6.1: unified weekly stock table separates discovery path, stock lifecycle and theme lifecycle.
# v2.6.0: dynamic theme semantics, reverse ignition trace, theme risk and unified three-direction report.
# v2.5.0: carrier hard gates and route-aware lexicographic target selection.
# v2.4.2: theme research context feeds proposal ranking and high-priority probe review.
# v2.4.1: confirmed climate catalyst raises research priority, never order permission.
# v2.4.0: categorical theme horizon (calendar x attention x carrier x repair).
# v2.3.1: recent carrier review, inactive-theme guard, and future-mapping guard.
# v2.3.0: P1.5 logical closure (candidate lifecycle, L0 bridge, proposals,
# reproducible snapshots, attention conflicts, and fail-fast run status).
__version__ = "2.26.0"
# 核心 = 预筛四通道 + 九原子特征 + READY 状态机 + 路径判定 + L1 板块五态
# Boundary: lifecycle discovery and virtual analysis only; legacy dual-track retired.
#
# v1 已知限制(诚实记录, 不隐藏):
#  - 价格为未复权序列: 涨跌停用 pct_chg(基于 preclose, 对除权稳健)识别;
#    结构窗口≤60日, 除权跳空视为可接受噪声, 后续接前复权再消除
#  - 基本面锚 = results/r4_fundamentals_panel.csv 点时取数(ann_date<=as_of),
#    规格原版"单季双正+环比加速"; 面板季度刷新: backfill_fundamentals_r4.py
#    (中报季进行中: 2026Q2 覆盖随披露递增, 8-31 截止后需补刷一次)
