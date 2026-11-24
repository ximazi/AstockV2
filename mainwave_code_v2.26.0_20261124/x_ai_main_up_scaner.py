"""
主升浪信号扫描器 V5 - 最终优化版本
基于回测结果，针对不同组合使用最优过滤条件：

组合A (涨停+连阳+缺口): 
  - 最佳过滤: 换手率3-12% + MA20向上
  - 回测收益: +2.44%, 成功率34.4%, 胜率42.7%

组合D (连阳+缺口+倍量):
  - 最佳过滤: MA20偏离<20%
  - 回测收益: +1.62%, 成功率24.8%, 胜率45.9%
"""

import pandas as pd
import os
import glob
import numpy as np
from datetime import datetime

# ================= 配置区域 =================
DATA_DIR = 'data'
CSV_PATTERN = '*_daily.csv'
MIN_YANG_COUNT = 4
VOL_MULTIPLIER = 2.0
SCAN_WINDOW = 20
MIN_LIMIT_UP_PCT = 0.095
SCAN_DAYS = 5              # 扫描最近N个交易日

# 组合A过滤条件
A_MIN_TURNOVER = 3.0       # 换手率下限3%
A_MAX_TURNOVER = 12.0      # 换手率上限12%

# 组合D过滤条件  
D_MAX_MA20_BIAS = 0.20     # MA20偏离不超过20%
# ============================================


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return None
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        if 'turnover' not in df.columns:
            df['turnover'] = 5.0
        return df
    except:
        return None


def check_continuous_yang(df, idx, min_count=4):
    count = 0
    for i in range(idx - 1, -1, -1):
        if df.loc[i, 'close'] > df.loc[i, 'open']:
            count += 1
        else:
            break
    return count, count >= min_count


def prepare_signals(df):
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['pct_chg'] = (df['close'] - df['prev_close']) / df['prev_close']
    df['vol_ma5'] = df['volume'].rolling(window=5).mean().shift(1)
    
    # 涨停
    df['is_limit_up'] = df['pct_chg'] > MIN_LIMIT_UP_PCT
    
    # 缺口和倍量
    df['is_gap_up'] = df['low'] > df['prev_high']
    df['is_double_vol'] = False
    mask = (df['vol_ma5'] > 0) & (~df['vol_ma5'].isna())
    df.loc[mask, 'is_double_vol'] = df.loc[mask, 'volume'] >= (df.loc[mask, 'vol_ma5'] * VOL_MULTIPLIER)
    
    # MA20及方向
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma20_prev'] = df['ma20'].shift(1)
    df['ma20_up'] = df['ma20'] > df['ma20_prev']
    df['ma20_bias'] = abs((df['close'] - df['ma20']) / df['ma20'])
    
    # 风险指标
    df['low_20d'] = df['low'].rolling(20).min()
    df['low_60d'] = df['low'].rolling(60).min()
    df['rise_20d'] = (df['close'] - df['low_20d']) / df['low_20d']
    df['rise_60d'] = (df['close'] - df['low_60d']) / df['low_60d']
    
    return df


def scan_stock_at_idx(df, idx, stock_name):
    """扫描特定日期的信号"""
    if idx < SCAN_WINDOW + 5:
        return []
    
    row = df.iloc[idx]
    results = []
    
    # 预计算信号
    yang_count, has_yang = check_continuous_yang(df, idx, MIN_YANG_COUNT)
    has_gap = row['is_gap_up']
    has_volume = row['is_double_vol']
    
    # 近期涨停
    start_idx = max(0, idx - SCAN_WINDOW)
    recent = df.iloc[start_idx : idx]
    limit_ups = recent[recent['is_limit_up'] == True]
    has_limit_up = len(limit_ups) > 0
    limit_date = limit_ups.iloc[-1]['date'].strftime('%m-%d') if has_limit_up else '-'
    
    # 指标值
    ma20_bias = row['ma20_bias'] if not pd.isna(row['ma20_bias']) else 999
    turnover = row['turnover'] if not pd.isna(row['turnover']) else 5.0
    ma20_up = row['ma20_up']
    vol_ratio = round(row['volume'] / row['vol_ma5'], 2) if row['vol_ma5'] > 0 else 0
    gap_size = round(row['low'] - row['prev_high'], 3) if has_gap else 0
    
    # 风险等级
    risk_flags = []
    if not pd.isna(row['rise_20d']) and row['rise_20d'] > 0.30:
        risk_flags.append(f"20日涨{row['rise_20d']*100:.0f}%")
    if not pd.isna(row['rise_60d']) and row['rise_60d'] > 0.50:
        risk_flags.append(f"60日涨{row['rise_60d']*100:.0f}%")
    if ma20_bias > 0.20:
        risk_flags.append(f"MA20偏离{ma20_bias*100:.0f}%")
    risk = "低" if len(risk_flags) == 0 else ("中" if len(risk_flags) == 1 else "高")
    risk_note = "; ".join(risk_flags) if risk_flags else "正常"
    
    base = {
        'Stock': stock_name,
        'Date': row['date'].strftime('%Y-%m-%d'),
        'Close': row['close'],
        'Risk': risk,
    }
    
    # === 四信号共振: 涨停+连阳+缺口+倍量 (原始策略) ===
    if has_limit_up and has_yang and has_gap and has_volume:
        results.append({
            **base,
            'Strategy': '◆四信号共振',
            'LimitDate': limit_date,
            'Yang': yang_count,
            'Gap': gap_size,
            'VolRatio': vol_ratio,
            'RiskNote': risk_note,
        })
    
    # === 组合A: 涨停+连阳+缺口 + 换手率3-12% + MA20向上 ===
    if (has_limit_up and has_yang and has_gap and 
        A_MIN_TURNOVER <= turnover <= A_MAX_TURNOVER and ma20_up):
        results.append({
            **base,
            'Strategy': '★A_涨停型(+2.44%)',
            'LimitDate': limit_date,
            'Yang': yang_count,
            'Gap': gap_size,
            'Turnover': f"{turnover:.1f}%",
            'MA20Up': 'Y',
        })
    
    # === 组合D: 连阳+缺口+倍量 + MA20偏离<20% ===
    if (has_yang and has_gap and has_volume and ma20_bias <= D_MAX_MA20_BIAS):
        results.append({
            **base,
            'Strategy': '★D_倍量型(+1.62%)',
            'Yang': yang_count,
            'Gap': gap_size,
            'VolRatio': vol_ratio,
            'MA20Bias': f"{ma20_bias*100:.0f}%",
        })
    
    return results


def scan_stock(file_path):
    """扫描单只股票最近N天的信号"""
    df = load_data(file_path)
    if df is None or len(df) < 60:
        return []
    
    df = prepare_signals(df)
    stock_name = os.path.basename(file_path).replace('.csv', '').replace('_daily', '')
    
    all_results = []
    
    # 扫描最近N天
    for i in range(SCAN_DAYS):
        idx = len(df) - 1 - i
        if idx >= SCAN_WINDOW + 5:
            results = scan_stock_at_idx(df, idx, stock_name)
            all_results.extend(results)
    
    return all_results


def main():
    print("=" * 100)
    print("主升浪信号扫描器 V5 - 最终优化版")
    print("=" * 100)
    print("扫描策略:")
    print("  ◆四信号共振: 涨停+连阳+缺口+倍量 (原始策略，信号稀少)")
    print("  ★A_涨停型: 涨停+连阳+缺口 + 换手率3-12% + MA20向上  [收益+2.44%, 成功率34.4%]")
    print("  ★D_倍量型: 连阳+缺口+倍量 + MA20偏离<20%           [收益+1.62%, 胜率45.9%]")
    print(f"扫描范围: 最近 {SCAN_DAYS} 个交易日")
    print("-" * 100)
    
    csv_files = glob.glob(os.path.join(DATA_DIR, CSV_PATTERN))
    if not csv_files:
        print("未找到数据文件")
        return
    
    print(f"扫描 {len(csv_files)} 只股票...")
    
    all_results = []
    processed = 0
    
    for file_path in csv_files:
        results = scan_stock(file_path)
        all_results.extend(results)
        processed += 1
        if processed % 1000 == 0:
            print(f"已扫描 {processed}/{len(csv_files)}...")
    
    print(f"\n扫描完成")
    print("=" * 100)
    
    if not all_results:
        print("\n当前没有股票满足任何策略条件")
        return
    
    df_all = pd.DataFrame(all_results)
    
    # 去重（同一股票同一天只保留一条）
    df_all = df_all.drop_duplicates(subset=['Stock', 'Date', 'Strategy'])
    
    # 按策略和日期排序
    df_all = df_all.sort_values(['Strategy', 'Date'], ascending=[True, False])
    
    # 按策略优先级显示
    strategies = ['◆四信号共振', '★A_涨停型(+2.44%)', '★D_倍量型(+1.62%)']
    
    for strategy in strategies:
        sub = df_all[df_all['Strategy'] == strategy]
        print(f"\n【{strategy}】发现 {len(sub)} 只")
        print("-" * 100)
        
        if len(sub) == 0:
            print("  无")
            continue
        
        # 根据策略选择列
        if '四信号共振' in strategy:
            cols = ['Stock', 'Date', 'Close', 'LimitDate', 'Yang', 'Gap', 'VolRatio', 'Risk', 'RiskNote']
        elif '★A_涨停型' in strategy:
            cols = ['Stock', 'Date', 'Close', 'LimitDate', 'Yang', 'Gap', 'Turnover', 'MA20Up', 'Risk']
        elif '★D_倍量型' in strategy:
            cols = ['Stock', 'Date', 'Close', 'Yang', 'Gap', 'VolRatio', 'MA20Bias', 'Risk']
        
        display = [c for c in cols if c in sub.columns]
        print(sub[display].to_string(index=False))
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = f'main_up_scan_{timestamp}.csv'
    df_all.to_csv(output, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output}")
    
    # 汇总
    print("\n【汇总】")
    for strategy in strategies:
        count = len(df_all[df_all['Strategy'] == strategy])
        print(f"  {strategy}: {count} 只")
    
    # 风险提示
    high_risk = df_all[df_all['Risk'] == '高']
    if len(high_risk) > 0:
        print(f"\n⚠️ 注意: {len(high_risk)} 只股票风险等级为【高】")
    
    # 今日信号单独提示
    today = df_all['Date'].max()
    today_signals = df_all[df_all['Date'] == today]
    if len(today_signals) > 0:
        print(f"\n【最新交易日({today})信号】")
        for strategy in strategies:
            sub = today_signals[today_signals['Strategy'] == strategy]
            if len(sub) > 0:
                stocks = ", ".join(sub['Stock'].tolist())
                print(f"  {strategy}: {stocks}")
    
    # 四信号共振特别提示
    four_signal = df_all[df_all['Strategy'] == '◆四信号共振']
    if len(four_signal) > 0:
        print(f"\n★★★ 发现 {len(four_signal)} 只四信号共振股票! ★★★")


if __name__ == "__main__":
    main()
