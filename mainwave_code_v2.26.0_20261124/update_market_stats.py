# -*- coding: utf-8 -*-
"""
增量更新 market_daily_stats.csv — 只算缓存里没有的新交易日
用法:
  python update_market_stats.py             # 自动补齐到最新
  python update_market_stats.py --from 2026-04-24
"""
import sys, io, os, glob, argparse
import pandas as pd
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CACHE = 'market_daily_stats.csv'

def get_existing_last_date():
    if not os.path.exists(CACHE):
        return None
    df = pd.read_csv(CACHE)
    df['date'] = pd.to_datetime(df['date'])
    return df['date'].max()

def scan_new_days(start_date):
    """扫描所有股票CSV里 >= start_date 的数据, 聚合成每日统计"""
    files = [f for f in glob.glob('data/*_daily.csv')
             if 'index' not in os.path.basename(f).lower()
             and '_SH_' not in os.path.basename(f)
             and '_SZ_' not in os.path.basename(f)]
    print(f'扫描 {len(files)} 只股票, 寻找 >= {start_date.date()} 的新数据...')
    rows = []
    for i, fp in enumerate(files):
        if i % 1000 == 0:
            print(f'  {i}/{len(files)}')
        try:
            df = pd.read_csv(fp, usecols=['date','pct_chg','isST','amount','volume'])
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= start_date]
            df = df[df['isST'] != 1]
            if len(df) > 0:
                rows.append(df[['date','pct_chg','amount','volume']])
        except: pass

    if not rows:
        print('无新数据'); return None

    big = pd.concat(rows, ignore_index=True)
    agg = big.groupby('date').agg(
        avg_pct=('pct_chg', 'mean'),
        median_pct=('pct_chg', 'median'),
        up_count=('pct_chg', lambda x: (x > 0).sum()),
        down_count=('pct_chg', lambda x: (x < 0).sum()),
        limit_up=('pct_chg', lambda x: (x > 9.5).sum()),
        limit_down=('pct_chg', lambda x: (x < -9.5).sum()),
        big_up=('pct_chg', lambda x: (x > 5).sum()),
        big_down=('pct_chg', lambda x: (x < -5).sum()),
        total_amount=('amount', 'sum'),
        stock_count=('pct_chg', 'count'),
    ).reset_index()
    agg['up_down_ratio'] = agg['up_count'] / agg['down_count'].replace(0, 1)
    return agg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', dest='from_date', type=str)
    args = parser.parse_args()

    last = get_existing_last_date()
    if args.from_date:
        start = pd.to_datetime(args.from_date)
    elif last is None:
        print('无缓存文件, 请先跑 analyze_market_risk.py'); return
    else:
        start = last + pd.Timedelta(days=1)
        print(f'缓存最后日期: {last.date()}, 从 {start.date()} 起增量更新')

    new_agg = scan_new_days(start)
    if new_agg is None or len(new_agg) == 0:
        print('无新交易日数据'); return

    # 合并
    if os.path.exists(CACHE):
        old = pd.read_csv(CACHE)
        old['date'] = pd.to_datetime(old['date'])
        combined = pd.concat([old, new_agg], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last').sort_values('date')
    else:
        combined = new_agg.sort_values('date')

    combined.to_csv(CACHE, index=False)
    print(f'\n新增 {len(new_agg)} 天:')
    for _, r in new_agg.iterrows():
        print(f'  {r["date"].date()}: 均涨{r["avg_pct"]:+.2f}% 涨跌{int(r["up_count"])}/{int(r["down_count"])} 涨停{int(r["limit_up"])} 跌停{int(r["limit_down"])} 成交额{r["total_amount"]/1e8:.0f}亿')
    print(f'\n缓存总行数: {len(combined)}, 最新日期: {combined["date"].max().date()}')

if __name__ == '__main__':
    main()
