# x_stock_data_updater.py
"""
纯股票数据更新工具 V2.0 - Tushare版

功能:
1. 基于 tushare pro 接口增量更新全市场日线数据
2. 单个交易日只需 2次API调用 (daily + daily_basic) 即可覆盖全市场 ~5500只
3. 输出 CSV 格式与原 baostock 版完全兼容 (下游 DKX/分析脚本无需改动)
4. 对无本地CSV的"裸股"自动走按股票补全路径 (并发)

用法:
    python x_stock_data_updater.py                     # 默认全市场增量更新
    python x_stock_data_updater.py --skip-list          # 跳过股票列表刷新
    python x_stock_data_updater.py --backfill-workers 8 # 首次建库时的并发度
    python x_stock_data_updater.py --force-full         # 强制全量(慎用)

环境依赖:
    .env 中需配置 TUSHARE_TOKEN
"""
import os
import sys
import argparse
import time
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
import tushare as ts


# --- 配置 ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
STOCK_LIST_FILE = os.path.join(DATA_DIR, 'stock_list.csv')
HISTORY_DAYS = 1000       # 新股首次获取的天数
MAX_BACKFILL_BATCH_DAYS = 6000  # tushare 单次最多返回 6000 行
# tushare 5000积分限流: daily/daily_basic 500次/分钟, 足够宽松
TUSHARE_MIN_INTERVAL = 0.12   # 最小请求间隔(秒), 单进程下 ~500次/min
# 按 ts_code 反查时单调用能拉多长 (1000天够用)
PER_STOCK_LOOKBACK_DAYS = HISTORY_DAYS

# 输出 CSV schema (保持与原 baostock 版完全一致)
OUTPUT_COLUMNS = [
    'date', 'code', 'open', 'high', 'low', 'close', 'preclose',
    'volume', 'amount', 'adjustflag', 'turnover', 'tradestatus',
    'pct_chg', 'isST'
]


# ==================== 全局状态 ====================
_pro = None                 # tushare pro api 单例
_api_lock = threading.Lock()
_last_api_time = [0.0]       # 可变容器以便闭包写入
_st_name_cache = {}         # {ts_code: is_st(0/1)}


def init_tushare():
    """初始化 tushare pro 接口"""
    global _pro
    if _pro is not None:
        return _pro
    load_dotenv()
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        raise RuntimeError("未在 .env 中找到 TUSHARE_TOKEN")
    ts.set_token(token)
    _pro = ts.pro_api()
    return _pro


def throttle():
    """tushare 请求节流, 线程安全"""
    with _api_lock:
        now = time.time()
        elapsed = now - _last_api_time[0]
        if elapsed < TUSHARE_MIN_INTERVAL:
            time.sleep(TUSHARE_MIN_INTERVAL - elapsed)
        _last_api_time[0] = time.time()


# ==================== 字段转换 ====================
def tushare_to_baostock_code(ts_code: str) -> str:
    """600000.SH -> sh.600000 ; 000001.SZ -> sz.000001"""
    if '.' not in ts_code:
        return ts_code
    symbol, market = ts_code.split('.')
    return f"{market.lower()}.{symbol}"


def baostock_to_tushare_code(bs_code: str) -> str:
    """sh.600000 -> 600000.SH"""
    if '.' not in bs_code:
        return bs_code
    market, symbol = bs_code.split('.')
    return f"{symbol}.{market.upper()}"


def format_trade_date(ts_date: str) -> str:
    """20260421 -> 2026-04-21"""
    return f"{ts_date[:4]}-{ts_date[4:6]}-{ts_date[6:8]}"


def transform_to_baostock_schema(df_daily: pd.DataFrame,
                                 df_basic: pd.DataFrame) -> pd.DataFrame:
    """
    合并 tushare daily + daily_basic，输出 baostock schema。
    输入 df_daily/df_basic 都有 ts_code 和 trade_date 列。
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # 合并换手率 (daily_basic 的 turnover_rate)
    if df_basic is not None and not df_basic.empty:
        df = df_daily.merge(
            df_basic[['ts_code', 'trade_date', 'turnover_rate']],
            on=['ts_code', 'trade_date'],
            how='left'
        )
    else:
        df = df_daily.copy()
        df['turnover_rate'] = None

    # 字段映射
    out = pd.DataFrame()
    out['date'] = df['trade_date'].apply(format_trade_date)
    out['code'] = df['ts_code'].apply(tushare_to_baostock_code)
    out['open'] = pd.to_numeric(df['open'], errors='coerce')
    out['high'] = pd.to_numeric(df['high'], errors='coerce')
    out['low'] = pd.to_numeric(df['low'], errors='coerce')
    out['close'] = pd.to_numeric(df['close'], errors='coerce')
    out['preclose'] = pd.to_numeric(df['pre_close'], errors='coerce')
    # 单位换算: 手->股, 千元->元
    out['volume'] = (pd.to_numeric(df['vol'], errors='coerce') * 100).round().astype('Int64')
    out['amount'] = (pd.to_numeric(df['amount'], errors='coerce') * 1000).round(2)
    out['adjustflag'] = 3
    out['turnover'] = pd.to_numeric(df['turnover_rate'], errors='coerce')
    out['tradestatus'] = 1
    out['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
    # ST 标记从名称缓存查
    out['isST'] = df['ts_code'].apply(lambda c: _st_name_cache.get(c, 0))

    return out


# ==================== 股票列表 ====================
def refresh_stock_list():
    """使用 tushare stock_basic 刷新全市场股票列表, 同时更新 ST 名称缓存"""
    print("=" * 60)
    print("[股票列表] 使用 tushare 刷新...")
    os.makedirs(DATA_DIR, exist_ok=True)

    pro = init_tushare()
    throttle()
    df = pro.stock_basic(
        exchange='',
        list_status='L',
        fields='ts_code,symbol,name,area,industry,list_date'
    )

    # 只保留A股: 主板(60/00)、创业板(30), 排除科创板(688)? 原baostock版保留了sh.6所有
    # 保持兼容: 沪主板(sh.6), 深主板/中小板(sz.0), 创业板(sz.3)
    # tushare 前缀: 600/601/603/605/688 都在 SH; 000/001/002/003 在 SZ 主板; 300/301 创业板
    keep_prefix = ('600', '601', '603', '605', '688', '000', '001', '002', '003', '300', '301')
    df = df[df['symbol'].str.startswith(keep_prefix)].copy()

    # 转成 baostock 格式 (sh.600000)
    df['bs_code'] = df['ts_code'].apply(tushare_to_baostock_code)

    # 更新 ST 名称缓存 (当前状态; 非历史每日)
    global _st_name_cache
    _st_name_cache = {
        row['ts_code']: (1 if ('ST' in str(row['name']).upper()) else 0)
        for _, row in df.iterrows()
    }

    # 输出为与原 CSV 同 schema
    out = pd.DataFrame()
    out['ts_code'] = df['bs_code']
    out['tradeStatus'] = 1
    out['name'] = df['name']
    out['symbol'] = df['symbol']
    out['area'] = df['area']
    out['industry'] = df['industry'].fillna('')
    out['list_date'] = df['list_date'].fillna('')

    out.to_csv(STOCK_LIST_FILE, index=False, encoding='utf-8')
    print(f"[股票列表] 已保存 {len(out)} 只股票 -> {STOCK_LIST_FILE}")
    print(f"[股票列表] ST名称缓存: {sum(_st_name_cache.values())} 只")
    return out


def load_stock_list():
    if not os.path.exists(STOCK_LIST_FILE):
        return None
    try:
        return pd.read_csv(STOCK_LIST_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(STOCK_LIST_FILE, encoding='gbk')


def rebuild_st_cache_from_list(stocks: pd.DataFrame):
    """如果跳过了列表刷新, 从本地 stock_list.csv 也能重建 ST 缓存"""
    global _st_name_cache
    if stocks is None or stocks.empty or 'name' not in stocks.columns:
        return
    _st_name_cache = {
        baostock_to_tushare_code(row['ts_code']):
            (1 if ('ST' in str(row['name']).upper()) else 0)
        for _, row in stocks.iterrows()
    }


# ==================== 扫描本地 ====================
def scan_existing_data(stocks: pd.DataFrame) -> dict:
    """
    扫描所有本地 CSV, 返回 {bs_code: latest_date (datetime)} 。
    没有本地文件的股票不在返回中。
    """
    result = {}
    for bs_code in stocks['ts_code']:
        file_path = os.path.join(DATA_DIR, f'{bs_code.replace(".", "_")}_daily.csv')
        if not os.path.exists(file_path):
            continue
        try:
            # 只读 date 列, 快很多
            df = pd.read_csv(file_path, usecols=['date'], encoding='utf-8')
            if df.empty:
                continue
            latest = pd.to_datetime(df['date']).max()
            result[bs_code] = latest
        except Exception:
            continue
    return result


# ==================== 交易日历 ====================
def get_trade_dates(start_date: str, end_date: str) -> list:
    """获取区间内的交易日列表, 格式 'YYYYMMDD'"""
    pro = init_tushare()
    throttle()
    # tushare trade_cal: exchange 默认SSE(上交所), 沪深同步
    cal = pro.trade_cal(
        exchange='SSE',
        start_date=start_date.replace('-', ''),
        end_date=end_date.replace('-', ''),
        is_open='1'
    )
    # tushare 返回字段: exchange, cal_date, is_open, pretrade_date
    return sorted(cal['cal_date'].tolist())


# ==================== 批量增量更新 (核心路径) ====================
def fetch_day_bundle(trade_date: str, max_retries: int = 3) -> pd.DataFrame:
    """
    拉取某一交易日的全市场 daily + daily_basic, 返回 baostock schema 的 DataFrame。
    trade_date 格式 'YYYYMMDD'。
    """
    pro = init_tushare()

    daily_df = None
    for attempt in range(max_retries):
        try:
            throttle()
            daily_df = pro.daily(trade_date=trade_date)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"[错误] daily({trade_date}) 重试{max_retries}次仍失败: {e}")
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    basic_df = None
    for attempt in range(max_retries):
        try:
            throttle()
            basic_df = pro.daily_basic(trade_date=trade_date)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"[警告] daily_basic({trade_date}) 失败, 换手率将为空: {e}")
            basic_df = pd.DataFrame(columns=['ts_code', 'trade_date', 'turnover_rate'])
            break

    return transform_to_baostock_schema(daily_df, basic_df)


def write_stock_increment(bs_code: str, new_rows: pd.DataFrame):
    """把单只股票的新增行追加到 CSV, 做去重排序"""
    if new_rows is None or new_rows.empty:
        return 0
    file_path = os.path.join(DATA_DIR, f'{bs_code.replace(".", "_")}_daily.csv')

    new_rows = new_rows.copy()
    new_rows['date'] = pd.to_datetime(new_rows['date'])

    if os.path.exists(file_path):
        try:
            existing = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            existing = pd.read_csv(file_path, encoding='gbk')
        if not existing.empty:
            existing['date'] = pd.to_datetime(existing['date'])
            merged = pd.concat([existing, new_rows], ignore_index=True)
        else:
            merged = new_rows
    else:
        merged = new_rows

    merged = merged.drop_duplicates(subset=['date'], keep='last').sort_values('date').reset_index(drop=True)
    merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')
    # 保证列顺序
    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = None
    merged = merged[OUTPUT_COLUMNS]
    merged.to_csv(file_path, index=False, encoding='utf-8')
    return len(new_rows)


def _fetch_single_stock_range(bs_code: str, start_yyyymmdd: str,
                              end_yyyymmdd: str) -> pd.DataFrame:
    """单股拉取指定区间, 返回 baostock schema 的 DataFrame (用于老旧股补齐)"""
    pro = init_tushare()
    ts_code = baostock_to_tushare_code(bs_code)
    try:
        throttle()
        daily_df = pro.daily(ts_code=ts_code,
                             start_date=start_yyyymmdd, end_date=end_yyyymmdd)
    except Exception as e:
        print(f"  [{bs_code}] daily 失败: {e}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        throttle()
        basic_df = pro.daily_basic(ts_code=ts_code,
                                   start_date=start_yyyymmdd, end_date=end_yyyymmdd)
    except Exception:
        basic_df = pd.DataFrame(columns=['ts_code', 'trade_date', 'turnover_rate'])

    return transform_to_baostock_schema(daily_df, basic_df)


def run_batch_incremental(stocks: pd.DataFrame, latest_dates: dict,
                          recover_stale: bool = False):
    """
    核心增量路径: bulk 只从 max(latest_date)+1 开始, 不被个别停牌股拖累。

    设计:
      - bulk 起点 = max(latest_date) + 1天  → 通常只拉 1~2 个交易日
      - 落后股票(latest < max)默认**跳过**:它们多半是停牌, 强行单股查询也拿不到数据,
        浪费 API 配额。复牌时下次 bulk 自然能捞到 (CSV 会跳过停牌段, 符合实际)。
      - 仅当 recover_stale=True 时, 才对落后股票走单股回填 (适合长期未更新后恢复)。
    """
    if not latest_dates:
        print("[批量增量] 无已有数据, 跳过此路径")
        return

    today = datetime.now()
    today_ts = pd.Timestamp(today.date())
    # 过滤掉未来日期(防御数据异常)
    valid = {c: d for c, d in latest_dates.items() if d <= today_ts}
    if not valid:
        print("[批量增量] 所有 latest_date 异常, 跳过")
        return

    max_latest = max(valid.values())
    main_count = sum(1 for d in valid.values() if d == max_latest)
    stale = {c: d for c, d in valid.items() if d < max_latest}

    print(f"[批量增量] 主流 latest_date = {max_latest.strftime('%Y-%m-%d')} "
          f"({main_count} 只)")
    if stale:
        oldest = min(stale.values())
        if recover_stale:
            print(f"[批量增量] 落后 {len(stale)} 只 (最旧 {oldest.strftime('%Y-%m-%d')}), 启用单股回填")
        else:
            print(f"[批量增量] 落后 {len(stale)} 只 (最旧 {oldest.strftime('%Y-%m-%d')}), 推断为停牌, **跳过** (要回填用 --recover-stale)")

    pending_by_code = {}

    # === 1) Bulk 路径: max_latest+1 → today ===
    bulk_start = (max_latest + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    trade_dates = get_trade_dates(bulk_start, end_date) if bulk_start <= end_date else []

    if trade_dates:
        print(f"\n[批量增量] Bulk: {len(trade_dates)} 个交易日 ({bulk_start} ~ {end_date})")
        for i, td in enumerate(trade_dates, 1):
            t0 = time.time()
            day_df = fetch_day_bundle(td)
            elapsed = time.time() - t0
            if day_df.empty:
                print(f"  [{i}/{len(trade_dates)}] {td} 无数据, 跳过")
                continue
            td_dt = pd.to_datetime(format_trade_date(td))
            for bs_code, group in day_df.groupby('code'):
                if bs_code not in valid:
                    continue
                if td_dt <= valid[bs_code]:
                    continue
                pending_by_code.setdefault(bs_code, []).append(group)
            print(f"  [{i}/{len(trade_dates)}] {td} 拉取 {len(day_df)} 行, "
                  f"累计 {sum(len(v) for v in pending_by_code.values())} 行 ({elapsed:.1f}s)")
    else:
        print(f"\n[批量增量] 主流已最新, 无需 bulk")

    # === 2) 单股回填 (仅 recover_stale=True 时执行) ===
    if recover_stale and stale:
        max_latest_yyyymmdd = max_latest.strftime('%Y%m%d')
        print(f"\n[批量增量] 单股回填落后股票 ({len(stale)} 只)...")
        filled = 0
        suspended = 0
        for bs_code, their_latest in stale.items():
            start = (their_latest + timedelta(days=1)).strftime('%Y%m%d')
            if start > max_latest_yyyymmdd:
                continue
            df = _fetch_single_stock_range(bs_code, start, max_latest_yyyymmdd)
            if df.empty:
                suspended += 1
                continue
            pending_by_code.setdefault(bs_code, []).append(df)
            filled += 1
            if (filled + suspended) % 50 == 0:
                print(f"  ...进度 {filled + suspended}/{len(stale)} (有数据{filled} 停牌{suspended})")
        print(f"[批量增量] 单股回填完成: 有数据 {filled} 只, 停牌 {suspended} 只")

    # === 3) 落盘 ===
    if not pending_by_code:
        print(f"\n[批量增量] 所有股票均已最新, 无需落盘")
        return

    print(f"\n[批量增量] 开始落盘, 共 {len(pending_by_code)} 只股票有新增")
    total_written = 0
    touched = 0
    for bs_code, parts in pending_by_code.items():
        combined = pd.concat(parts, ignore_index=True)
        n = write_stock_increment(bs_code, combined)
        if n > 0:
            touched += 1
            total_written += n
    print(f"[批量增量] 落盘完成: {touched} 只股票, 新增 {total_written} 行")


# ==================== 按股票补全 (首次建库) ====================
def backfill_single_stock(bs_code: str) -> tuple:
    """
    对没有本地 CSV 的股票做长历史补全。
    返回 (bs_code, status, rows_written)
    """
    pro = init_tushare()
    ts_code = baostock_to_tushare_code(bs_code)
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=PER_STOCK_LOOKBACK_DAYS)).strftime('%Y%m%d')

    try:
        throttle()
        daily_df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if daily_df is None or daily_df.empty:
            return (bs_code, 'empty', 0)

        throttle()
        basic_df = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)

        out = transform_to_baostock_schema(daily_df, basic_df)
        n = write_stock_increment(bs_code, out)
        return (bs_code, 'ok', n)
    except Exception as e:
        return (bs_code, f'error:{e}', 0)


def run_backfill(stocks: pd.DataFrame, missing_codes: list, workers: int = 6):
    """按股票并发补全首次建库"""
    if not missing_codes:
        return
    print(f"\n[首次补全] 共 {len(missing_codes)} 只股票无本地CSV, 使用 {workers} 线程并发补全")

    done = 0
    failed = 0
    total_rows = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(backfill_single_stock, code): code for code in missing_codes}
        for fut in as_completed(futures):
            bs_code = futures[fut]
            try:
                _, status, n = fut.result()
            except Exception as e:
                status, n = f'error:{e}', 0

            if status == 'ok' and n > 0:
                done += 1
                total_rows += n
            elif status == 'empty':
                pass
            else:
                failed += 1

            completed = done + failed
            if completed % 100 == 0 or completed == len(missing_codes):
                print(f"  进度 {completed}/{len(missing_codes)} "
                      f"(成功:{done} 失败:{failed} 累计{total_rows}行)")

    print(f"[首次补全] 完成: 成功{done} 失败{failed} 写入{total_rows}行")


# ==================== 主流程 ====================
def run(skip_list: bool = False, backfill_workers: int = 6,
        force_full: bool = False, recover_stale: bool = False):
    t_start = time.time()

    init_tushare()

    # 1) 股票列表
    if skip_list:
        print("[跳过] 股票列表刷新")
        stocks = load_stock_list()
        if stocks is None:
            print("[错误] 本地无 stock_list.csv, 请先不加 --skip-list 运行一次")
            return
        rebuild_st_cache_from_list(stocks)
    else:
        stocks = refresh_stock_list()

    # 2) 扫描本地
    print("\n" + "=" * 60)
    print("[扫描] 正在扫描本地已有数据...")
    latest_dates = scan_existing_data(stocks) if not force_full else {}
    have = len(latest_dates)
    miss = len(stocks) - have
    print(f"[扫描] 已有本地数据: {have} 只, 无本地数据: {miss} 只")

    # 3) 批量增量 (已有数据的股票)
    if latest_dates:
        print("\n" + "=" * 60)
        run_batch_incremental(stocks, latest_dates, recover_stale=recover_stale)

    # 4) 按股票补全 (无本地数据的股票)
    all_codes = set(stocks['ts_code'].tolist())
    missing_codes = sorted(all_codes - set(latest_dates.keys()))
    if missing_codes:
        print("\n" + "=" * 60)
        run_backfill(stocks, missing_codes, workers=backfill_workers)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"[完成] 全市场数据更新完毕! 耗时 {int(elapsed//60)}分{int(elapsed%60)}秒")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="股票数据更新工具 V2 (tushare)")
    parser.add_argument('--skip-list', action='store_true', help='跳过股票列表刷新')
    parser.add_argument('--backfill-workers', type=int, default=6,
                        help='首次建库补全的并发度 (默认6)')
    parser.add_argument('--force-full', action='store_true',
                        help='强制全量重拉 (等同于忽略本地, 走补全路径)')
    parser.add_argument('--recover-stale', action='store_true',
                        help='对落后股票(停牌嫌疑)走单股回填; 默认跳过')
    args = parser.parse_args()

    run(
        skip_list=args.skip_list,
        backfill_workers=args.backfill_workers,
        force_full=args.force_full,
        recover_stale=args.recover_stale,
    )


if __name__ == '__main__':
    main()
