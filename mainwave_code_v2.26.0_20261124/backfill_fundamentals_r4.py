# -*- coding: utf-8 -*-
"""backfill_fundamentals_r4 — R4 业绩面板回填 (point-in-time, 带 ann_date 公告日).

目的: 为"业绩作排序加分项"提供无前视的历史业绩面板。
口径: 按季度 period 用 fina_indicator_vip 批量拉全市场 (一次~7500只), 2019Q1→2026Q1;
保留 ann_date(公告日) = point-in-time 关键字段; 信号日 d 只能用 ann_date<=d 的报告 (无前视)。
去重: 同 (ts_code,end_date) 可能多行(原始+追溯重述), 保留 ann_date 最新且非空最多的一行。
输出: results/r4_fundamentals_panel.csv (供 as-of join 用)。

环境: TUSHARE_TOKEN 必须通过环境变量提供。
用法: python backfill_fundamentals_r4.py        # 全量回填
      python backfill_fundamentals_r4.py --smoke # 只拉最近2期试水
      python backfill_fundamentals_r4.py --recent 3  # 每周: 只拉最近3期并并入现有面板 (面板不存在时全量)
令牌从仓库根目录 .env 读取 (TUSHARE_TOKEN)。
"""
import sys, io, os, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

FIELDS = ('ts_code,ann_date,end_date,roe,roe_yoy,netprofit_yoy,q_profit_yoy,or_yoy,'
          'q_sales_yoy,grossprofit_margin,netprofit_margin,debt_to_assets,eps')
OUT = 'results/r4_fundamentals_panel.csv'


def quarters(lo=2019, hi=2026):
    qs = []
    for y in range(lo, hi + 1):
        for mmdd in ('0331', '0630', '0930', '1231'):
            p = f"{y}{mmdd}"
            if p <= time.strftime('%Y%m%d'):  # 报告期已结束的季度; 尚未披露的公司本次为空, 下次每周刷新补上
                qs.append(p)
    return qs


def pull(pro, period):
    for attempt in range(3):
        try:
            try:
                return pro.fina_indicator_vip(period=period, fields=FIELDS)
            except Exception:
                return pro.fina_indicator_vip(period=period)  # 字段不符时退回全字段
        except Exception as e:
            if attempt == 2:
                print(f"  {period} 失败(3次): {type(e).__name__}: {str(e)[:100]}")
                return pd.DataFrame()
            time.sleep(2.0)
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--recent', type=int, default=0, help='只拉最近 N 个报告期并并入现有面板')
    args = ap.parse_args()
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        raise RuntimeError('TUSHARE_TOKEN is missing from the environment')
    pro = ts.pro_api(token)
    qs = quarters()
    if args.smoke:
        qs = qs[-2:]
    merge = bool(args.recent) and os.path.exists(OUT)
    if merge:
        qs = qs[-args.recent:]
    print(f"{'刷新' if merge else '回填'} {len(qs)} 个季度: {qs[0]}→{qs[-1]}")
    frames = []
    for i, p in enumerate(qs, 1):
        df = pull(pro, p)
        if not df.empty:
            frames.append(df)
        print(f"  [{i}/{len(qs)}] {p}: {len(df)} 行")
        time.sleep(0.4)
    if not frames:
        print("无数据, 退出"); return
    A = pd.concat(frames, ignore_index=True)
    if merge:  # 新拉的报告期与现有面板合并, 下面的去重保留公告日最新、字段最全的一行
        A = pd.concat([pd.read_csv(OUT), A], ignore_index=True)
    A['end_date'] = A['end_date'].astype(str).str[:8]  # 旧面板读回是整数, 新拉的是字符串; 统一后去重才认得出同一期
    # 去重: 每 (ts_code,end_date) 保留 ann_date 最新 + 非空字段最多
    A['_nn'] = A.notna().sum(axis=1)
    A['ann_date'] = pd.to_numeric(A['ann_date'], errors='coerce')
    A = (A.sort_values(['ts_code', 'end_date', 'ann_date', '_nn'])
           .drop_duplicates(['ts_code', 'end_date'], keep='last').drop(columns='_nn'))
    A = A.dropna(subset=['ann_date'])
    os.makedirs('results', exist_ok=True)
    A.to_csv(OUT, index=False)
    print(f"\n保存 {OUT}: {len(A)} 行, {A['ts_code'].nunique()} 只股, "
          f"end_date {A['end_date'].min()}→{A['end_date'].max()}, ann_date 非空 {A['ann_date'].notna().mean():.0%}")
    print("字段:", list(A.columns))


if __name__ == '__main__':
    main()
