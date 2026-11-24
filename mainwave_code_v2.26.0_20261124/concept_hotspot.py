#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
概念热点匹配器 — 股票概念×市场热点的共振检测
============================================================
核心目标:
  同样的技术面信号(S级·潜龙)，在热点板块中爆发概率远高于非热点。
  本工具为每只股票计算"概念热度分"，供early_scanner_v4排序加权。

三层架构:
  ① 数据层: Tushare获取 股票→概念映射 + 每日概念热度
  ② 分析层: 从涨停板概念分布自动检测当前热点
  ③ 匹配层: 股票概念∩市场热点 → 热度加分

用法:
  # Step 1: 获取概念映射（首次/每月更新一次）
  python concept_hotspot.py --fetch-mapping

  # Step 2: 获取今日热点（每天盘后跑一次）
  python concept_hotspot.py --fetch-hotspot --date 20260328

  # Step 3: 分析概念热度（配合early_scanner使用）
  python concept_hotspot.py --analyze --scan-csv early_scan_V4.0_xxx.csv

  # 查看某只股票的概念信息
  python concept_hotspot.py --stock sz_002192

数据文件:
  concept_mapping.json  — 股票→概念映射（~5000只股票）
  hotspot_daily.json    — 每日概念热度排名
  hotspot_config.json   — 手动配置的事件热点（可选）
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ============== 配置 ==============
DATA_DIR = 'data'
MAPPING_FILE = 'concept_mapping.json'
HOTSPOT_FILE = 'hotspot_daily.json'
HOTSPOT_HISTORY_FILE = 'hotspot_history.json'  # 多日历史（用于生命周期判断）
DYNAMIC_MAPPING_DIR = 'concept_dynamic_mapping'
CONFIG_FILE = 'hotspot_config.json'
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / '.env')
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')  # 令牌放在 .env，不写在代码里

# 概念生命周期评分 → 越高越适合介入
LIFECYCLE_SCORE = {
    '发酵期': 100,   # 最佳介入! 涨停扩散+趋势加速
    '启动期': 85,    # 不错，板块效应形成中
    '萌芽期': 60,    # 有潜力但不确定
    '高潮期': 30,    # 危险! 情绪到顶，追入必套
    '分化期': 15,    # 晚了，强弱分化
    '退潮期': 5,     # 别碰
}
# ==================================


# ==============================================================================
# 数据获取层（Tushare API）
# ==============================================================================

def get_tushare_pro():
    """获取Tushare Pro API"""
    try:
        import tushare as ts
        token = TUSHARE_TOKEN or os.environ.get('TUSHARE_TOKEN', '')
        if not token:
            print("  ⚠️  请在 .env 中设置 TUSHARE_TOKEN")
            return None
        return ts.pro_api(token)
    except ImportError:
        print("  ⚠️  请先安装tushare: pip install tushare")
        return None


# ==============================================================================
# 概念生命周期判断（多日趋势对比）
# ==============================================================================

def load_hotspot_history():
    """加载多日热点历史"""
    if not os.path.exists(HOTSPOT_HISTORY_FILE):
        return {}
    with open(HOTSPOT_HISTORY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_hotspot_history(history):
    """保存多日热点历史"""
    with open(HOTSPOT_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def estimate_lifecycle(concept_name, today_limit_count, history, today_date):
    """
    根据多日趋势判断概念生命周期
    核心逻辑: 不能只看今天涨停数，要看趋势方向
      - 昨天3家→今天8家 = 发酵期(加速升温)
      - 昨天8家→今天3家 = 分化期(在退潮)
      - 同样是5家涨停，方向完全不同!

    返回: (lifecycle, trend_detail)
    """
    # 获取该概念最近3天的涨停数
    concept_history = history.get(concept_name, {})
    recent_days = sorted(concept_history.keys())[-5:]  # 最近5天

    prev_counts = []
    for d in recent_days:
        if d < today_date:
            prev_counts.append(concept_history[d])

    today = today_limit_count
    yesterday = prev_counts[-1] if prev_counts else 0
    day_before = prev_counts[-2] if len(prev_counts) >= 2 else 0

    # 趋势判断
    trend = today - yesterday  # 正=升温, 负=降温
    trend_2d = today - day_before if day_before > 0 else trend

    # 峰值（历史最高涨停数）
    all_counts = prev_counts + [today]
    peak = max(all_counts) if all_counts else today

    # ---- 生命周期判断 ----

    # 1. 高潮期: 涨停≥10家
    if today >= 10:
        return '高潮期', f'涨停{today}家(危险!)'

    # 2. 退潮期: 曾经高潮(peak≥8)但现在大幅回落
    if peak >= 8 and today <= peak * 0.3:
        return '退潮期', f'涨停{today}家(峰值{peak}→退潮)'

    # 3. 分化期: 在下降趋势中(今天<昨天<前天)
    if today < yesterday and yesterday > 0:
        if yesterday < day_before and day_before > 0:
            return '分化期', f'{day_before}→{yesterday}→{today}家(连续下降)'
        if today <= peak * 0.5 and peak >= 5:
            return '分化期', f'涨停{today}家(峰值{peak}→分化)'
        return '分化期', f'{yesterday}→{today}家(在降温)'

    # 4. 发酵期: 涨停≥5且在上升趋势
    if today >= 5 and trend >= 0:
        if trend > 0:
            return '发酵期', f'{yesterday}→{today}家(加速扩散!)'
        return '发酵期', f'涨停{today}家(持续高位)'

    # 5. 启动期: 涨停3-4家，或从低位上升
    if today >= 3:
        if trend > 0:
            return '启动期', f'{yesterday}→{today}家(在升温)'
        return '启动期', f'涨停{today}家(板块效应形成)'

    # 6. 萌芽期: 涨停1-2家
    if today >= 1:
        if yesterday == 0:
            return '萌芽期', f'涨停{today}家(首次出现!)'
        if trend > 0:
            return '萌芽期', f'{yesterday}→{today}家(开始萌动)'
        return '萌芽期', f'涨停{today}家'

    # 无涨停
    if peak > 0:
        return '退潮期', f'涨停0家(曾有{peak}家→已退潮)'
    return '萌芽期', '无涨停'


def update_hotspot_history(hotspot, today_date):
    """将今日热点数据写入历史文件"""
    history = load_hotspot_history()

    concepts = hotspot.get('concepts', {})
    for concept, info in concepts.items():
        limit_count = info.get('limit_count', 0)
        if limit_count > 0:
            if concept not in history:
                history[concept] = {}
            history[concept][today_date] = limit_count

    # 清理超过30天的旧数据
    cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    for concept in list(history.keys()):
        history[concept] = {d: c for d, c in history[concept].items() if d >= cutoff}
        if not history[concept]:
            del history[concept]

    save_hotspot_history(history)
    return history


def safe_call(pro, func, max_retries=3, **kwargs):
    """临时错误重试；权限拒绝立即返回并保留缺失原因。"""
    import time
    for attempt in range(max_retries):
        try:
            result = func(**kwargs)
            time.sleep(0.3)  # 控制调用频率
            return result
        except Exception as e:
            message = str(e).lower()
            denied = any(text in message for text in (
                '没有接口', '没有访问权限', '无访问权限', '权限不足', '未开通',
                'permission denied', 'access denied',
            ))
            if denied or attempt == max_retries - 1:
                status = 'PERMISSION_DENIED' if denied else 'FAILED'
                note = '接口未授权，跳过该数据源，不重复重试' if denied else 'API重试后仍失败'
                print(f"    [WARN] {note}: {e}")
                result = pd.DataFrame()
                result.attrs['api_status'] = status
                return result
            time.sleep(2)


def _normalize_member_code(code_raw):
    code_text = str(code_raw).strip()
    if not code_text:
        return ""
    if "." in code_text:
        number, exchange = code_text.split(".", 1)
        return f"{exchange.lower()}_{number}"
    prefix = "sh_" if code_text.startswith(("5", "6")) else (
        "sz_" if code_text.startswith(("0", "3")) else "bj_"
    )
    return f"{prefix}{code_text}"


def fetch_dynamic_hot_mapping(pro, date, concept_names):
    """Fetch members only for today's active themes and persist a point-in-time overlay."""
    names = list(dict.fromkeys(str(value).strip() for value in concept_names if value))[:30]
    if not names:
        return {}
    output = Path(DYNAMIC_MAPPING_DIR) / f"concept_dynamic_{date}.json"
    previous = {}
    if output.exists():
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    if date != datetime.now().strftime("%Y%m%d"):
        return previous.get("mapping", {})
    captured = set(previous.get("captured_concepts", []))
    missing = [name for name in names if name not in captured]
    overlay = {
        symbol: {**item, "concepts": list(item.get("concepts", []))}
        for symbol, item in previous.get("mapping", {}).items()
    }
    if not missing:
        return overlay
    ths_index = safe_call(pro, pro.ths_index, exchange="A", type="N")
    if ths_index is None or ths_index.empty:
        return overlay
    code_by_name = {
        str(row.get("name", "")).strip(): row.get("ts_code", "")
        for _, row in ths_index.iterrows()
    }
    for name in missing:
        concept_code = code_by_name.get(name)
        if not concept_code:
            continue
        members = safe_call(pro, pro.ths_member, ts_code=concept_code)
        if members is None or members.empty:
            continue
        captured.add(name)
        for _, member in members.iterrows():
            symbol = _normalize_member_code(
                member.get("code", member.get("con_code", ""))
            )
            if not symbol:
                continue
            item = overlay.setdefault(symbol, {
                "name": str(member.get("name", "")), "concepts": [],
            })
            if name not in item["concepts"]:
                item["concepts"].append(name)
    if not captured:
        return overlay
    payload = {
        "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "TUSHARE_THS_ACTIVE_TOP30",
        "captured_concepts": sorted(captured),
        "mapping": overlay,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, output)
    print(f"    动态题材覆盖: {len(captured)} 个题材 / {len(overlay)} 只股票")
    return overlay


def fetch_akshare_concepts(mapping):
    """Populate concept membership from Eastmoney through AkShare."""
    try:
        import akshare as ak
    except ImportError:
        print("  ERROR: AkShare is unavailable; cannot use the fallback concept source.")
        return 0
    print("  THS access is unavailable; falling back to AkShare/Eastmoney concepts...")
    concepts = safe_call(None, ak.stock_board_concept_name_em, max_retries=3)
    if concepts is None or concepts.empty or '板块名称' not in concepts.columns:
        print("  ERROR: AkShare concept list is incomplete.")
        return 0
    names = [str(value) for value in concepts['板块名称'].dropna().unique()]
    if len(names) < 50:
        print(f"  ERROR: AkShare returned only {len(names)} concepts.")
        return 0
    successes = 0
    for index, concept_name in enumerate(names, start=1):
        members = safe_call(
            None, ak.stock_board_concept_cons_em, max_retries=3, symbol=concept_name
        )
        if members is None or members.empty or '代码' not in members.columns:
            continue
        successes += 1
        for _, member in members.iterrows():
            code_raw = str(member.get('代码', '')).split('.')[0].zfill(6)
            prefix = 'sh_' if code_raw.startswith(('5', '6')) else (
                'sz_' if code_raw.startswith(('0', '3')) else 'bj_'
            )
            code = f"{prefix}{code_raw}"
            if code not in mapping:
                mapping[code] = {
                    'name': str(member.get('名称', '')), 'industry': '', 'concepts': [],
                }
            if concept_name not in mapping[code]['concepts']:
                mapping[code]['concepts'].append(concept_name)
        if index % 50 == 0:
            print(f"    AkShare concepts processed: {index}/{len(names)}")
    print(f"    AkShare concepts with members: {successes}/{len(names)}")
    return successes


def fetch_concept_mapping(pro):
    """
    获取股票→概念映射表（每月更新一次即可）
    数据源: 同花顺概念指数 + 开盘啦概念
    输出: concept_mapping.json
    """
    print("获取股票概念映射...")
    mapping = {}  # stock_code -> {name, industry, concepts: []}

    # 1. 获取所有A股基本信息
    print("  [1/4] 获取股票列表...")
    stock_basic = safe_call(pro, pro.stock_basic, exchange='', list_status='L',
                            fields='ts_code,name,industry,market')
    if stock_basic is None or stock_basic.empty:
        print("  ❌ 无法获取股票列表")
        try:
            previous = json.loads(
                Path(MAPPING_FILE).read_text(encoding='utf-8')
            ).get('mapping', {})
        except Exception:
            previous = {}
        if not previous:
            print("  ERROR: stock basic and the local mapping base are both unavailable.")
            return {}
        mapping = {
            code: {
                'name': item.get('name', ''), 'industry': item.get('industry', ''),
                'concepts': [],
            } for code, item in previous.items()
        }
        print(f"    stock_basic limited; using local stock base: {len(mapping)}")

    for _, row in stock_basic.iterrows():
        ts_code = row['ts_code']
        # 转换代码格式: 000001.SZ -> sz_000001
        parts = ts_code.split('.')
        if len(parts) == 2:
            prefix = parts[1].lower()
            code = f"{prefix}_{parts[0]}"
        else:
            code = ts_code
        mapping[code] = {
            'name': row.get('name', ''),
            'industry': row.get('industry', ''),
            'concepts': [],
        }

    print(f"    股票总数: {len(mapping)}")

    # 2. 获取同花顺概念列表
    print("  [2/4] 获取同花顺概念列表...")
    ths_index = safe_call(pro, pro.ths_index, exchange='A', type='N')
    concept_list = []
    if ths_index is not None and not ths_index.empty:
        concept_list = ths_index[['ts_code', 'name']].values.tolist()
    mapping_source = 'TUSHARE_THS'
    rich_concept_count = len(concept_list)
    print(f"    THS concepts: {len(concept_list)}")
    if len(concept_list) < 50:
        akshare_count = fetch_akshare_concepts(mapping)
        if akshare_count < 50:
            print("  ERROR: all rich concept sources failed; keep the existing mapping unchanged.")
            return {}
        mapping_source = 'AKSHARE_EASTMONEY'
        rich_concept_count = akshare_count
        concept_list = []
        print(f"    同花顺概念: {len(concept_list)}个")

    # 3. 获取每个概念的成分股
    print(f"  [3/4] 获取概念成分股（{len(concept_list)}个概念）...")
    processed = 0
    for concept_code, concept_name in concept_list:
        members = safe_call(pro, pro.ths_member, ts_code=concept_code)
        if members is not None and not members.empty:
            for _, m in members.iterrows():
                code_raw = m.get('code', m.get('con_code', ''))
                # 尝试匹配到mapping
                code_text = str(code_raw).strip()
                if '.' in code_text:
                    number, exchange = code_text.split('.', 1)
                    code = f"{exchange.lower()}_{number}"
                else:
                    code = ('sh_' if code_text.startswith(('5', '6')) else
                            'sz_' if code_text.startswith(('0', '3')) else 'bj_') + code_text
                if code in mapping and concept_name not in mapping[code]['concepts']:
                    mapping[code]['concepts'].append(concept_name)

        processed += 1
        if processed % 50 == 0:
            print(f"    已处理 {processed}/{len(concept_list)} ...")

    # 4. 补充行业分类
    print("  [4/4] 补充行业分类...")
    # 行业也作为一个"概念"加入
    for code, info in mapping.items():
        if info['industry'] and info['industry'] not in info['concepts']:
            info['concepts'].insert(0, info['industry'])

    # 统计
    has_concepts = sum(1 for v in mapping.values() if v['concepts'])
    avg_concepts = np.mean([len(v['concepts']) for v in mapping.values() if v['concepts']])
    multi_concept = sum(1 for v in mapping.values() if len(v['concepts']) > 1)
    print(f"\n  ✅ 映射完成: {len(mapping)}只股票, {has_concepts}只有概念标签, 均{avg_concepts:.1f}个概念/股")

    # 保存
    minimum_multi = max(100, int(len(mapping) * 0.10))
    if multi_concept < minimum_multi:
        print(f"  ERROR: rich concept membership is incomplete ({multi_concept} multi-concept "
              f"stocks; need >= {minimum_multi}); keep the existing mapping unchanged.")
        return {}

    mapping_path = Path(MAPPING_FILE)
    if mapping_path.exists():
        try:
            previous = json.loads(mapping_path.read_text(encoding='utf-8'))
            stamp = str(previous.get('generated', 'unknown'))
            stamp = stamp.replace('-', '').replace(':', '').replace(' ', '_')
            archive = Path('concept_mapping_history') / f"concept_mapping_{stamp}.json"
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                archive.write_bytes(mapping_path.read_bytes())
        except Exception as exc:
            print(f"  ERROR: cannot archive the existing mapping: {exc}")
            return {}

    save_data = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_stocks': len(mapping),
        'concept_count': rich_concept_count,
        'multi_concept_stocks': multi_concept,
        'source': mapping_source,
        'mapping': mapping,
    }
    temp_path = mapping_path.with_suffix('.tmp')
    temp_path.write_text(json.dumps(save_data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp_path, mapping_path)
    print(f"  保存: {MAPPING_FILE}")

    return mapping


def fetch_daily_hotspot(pro, date=None):
    """
    获取当日概念热度（每天盘后跑一次）
    数据源: 涨停股概念分布 + 同花顺热榜 + 开盘啦概念
    输出: hotspot_daily.json
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    print(f"获取 {date} 概念热度...")
    hotspot = {'date': date, 'concepts': {}}
    dynamic_names = []

    # 1. 涨停板概念分布（最直接的热点信号）
    print("  [1/3] 获取涨停板数据...")
    limit_df = safe_call(pro, pro.limit_list_d, trade_date=date, limit_type='U')
    if limit_df is not None and not limit_df.empty:
        print(f"    涨停股: {len(limit_df)}只")

        # 加载概念映射
        mapping = load_concept_mapping(date)
        if mapping:
            # 统计涨停股的概念分布
            concept_limit_count = defaultdict(int)
            concept_limit_stocks = defaultdict(list)
            for _, row in limit_df.iterrows():
                ts_code = row.get('ts_code', '')
                parts = ts_code.split('.')
                if len(parts) == 2:
                    code = f"{parts[1].lower()}_{parts[0]}"
                else:
                    continue
                if code in mapping:
                    for concept in mapping[code].get('concepts', []):
                        concept_limit_count[concept] += 1
                        concept_limit_stocks[concept].append(code)

            # 按涨停股数量排序
            sorted_concepts = sorted(concept_limit_count.items(), key=lambda x: -x[1])
            for concept, count in sorted_concepts:
                if count >= 2:  # 至少2只涨停才算热点
                    hotspot['concepts'][concept] = {
                        'limit_count': count,
                        'stocks': concept_limit_stocks[concept][:10],
                        'source': 'limit_up',
                    }
            print(f"    热点概念(≥2只涨停): {len(hotspot['concepts'])}个")

    ths_hot = kpl = None
    # 2. 同花顺热榜
    print("  [2/3] 获取同花顺热榜...")
    try:
        ths_hot = safe_call(pro, pro.ths_hot, trade_date=date, market='概念板块')
        if ths_hot is not None and not ths_hot.empty:
            if 'rank' in ths_hot:
                ths_hot = ths_hot.assign(
                    rank=pd.to_numeric(ths_hot['rank'], errors='coerce')
                ).sort_values('rank')
            for i, (_, row) in enumerate(ths_hot.head(20).iterrows()):
                name = row.get('ts_name', row.get('name', ''))
                name = str(name).strip() if pd.notna(name) else ''
                if name:
                    dynamic_names.append(name)
                    if name not in hotspot['concepts']:
                        hotspot['concepts'][name] = {'limit_count': 0, 'stocks': [], 'source': 'ths_hot'}
                    rank = row.get('rank', i + 1)
                    hotspot['concepts'][name]['ths_rank'] = int(rank) if pd.notna(rank) else i + 1
            print(f"    同花顺热榜TOP20已加入")
    except Exception as e:
        print(f"    同花顺热榜获取失败: {e}")

    # 3. 开盘啦概念
    print("  [3/3] 获取开盘啦概念...")
    try:
        kpl = safe_call(pro, pro.kpl_concept, trade_date=date)
        if kpl is not None and not kpl.empty:
            for _, row in kpl.iterrows():
                name = row.get('name', '')
                if name:
                    dynamic_names.append(name)
                    if name not in hotspot['concepts']:
                        hotspot['concepts'][name] = {
                            'limit_count': 0, 'stocks': [], 'source': 'kpl',
                        }
            print(f"    开盘啦概念已加入")
    except Exception as e:
        print(f"    开盘啦数据获取失败: {e}")

    hotspot['sources'] = {
        name: {
            'status': frame.attrs.get('api_status', 'EMPTY' if frame.empty else 'OK')
                      if frame is not None else 'FAILED',
            'rows': len(frame) if frame is not None else 0,
        }
        for name, frame in (
            ('limit_list_d', limit_df), ('ths_hot', ths_hot), ('kpl_concept', kpl),
        )
    }
    statuses = [item['status'] for item in hotspot['sources'].values()]
    hotspot['data_status'] = ('OK' if all(status == 'OK' for status in statuses)
                              else 'PARTIAL' if 'OK' in statuses
                              else 'EMPTY' if all(status == 'EMPTY' for status in statuses)
                              else 'UNAVAILABLE')
    coverage = ' | '.join(f"{name}={item['status']}({item['rows']})"
                          for name, item in hotspot['sources'].items())
    print(f"  数据源覆盖: {hotspot['data_status']} | {coverage}")
    if hotspot['data_status'] == 'UNAVAILABLE':
        raise RuntimeError('热榜数据源不可用，未覆盖已有热点文件；旧文件不代表今日数据。')
    if hotspot['data_status'] == 'PARTIAL':
        print('  [WARN] 使用其余可用来源继续；缺失榜单未补零或沿用旧排名，题材覆盖可能不完整。')

    # Active themes change faster than the monthly full mapping. Fetch only their
    # memberships, then use that overlay to correct today's limit-up attribution.
    dynamic_mapping = fetch_dynamic_hot_mapping(pro, date, dynamic_names)
    if dynamic_mapping and limit_df is not None and not limit_df.empty:
        limit_symbols = {
            _normalize_member_code(row.get('ts_code', ''))
            for _, row in limit_df.iterrows()
        }
        dynamic_counts = defaultdict(int)
        dynamic_stocks = defaultdict(list)
        for symbol, item in dynamic_mapping.items():
            if symbol not in limit_symbols:
                continue
            for concept in item.get('concepts', []):
                dynamic_counts[concept] += 1
                dynamic_stocks[concept].append(symbol)
        for concept, count in dynamic_counts.items():
            info = hotspot['concepts'].setdefault(concept, {
                'limit_count': 0, 'stocks': [], 'source': 'dynamic_hot_mapping',
            })
            info['limit_count'] = count
            info['stocks'] = dynamic_stocks[concept][:10]
            info['membership_source'] = 'dynamic_hot_mapping'

    # 计算热度分 + 生命周期
    history = load_hotspot_history()
    for concept, info in hotspot['concepts'].items():
        score = 0
        limit_count = info.get('limit_count', 0)
        score += limit_count * 10      # 每只涨停+10
        if info.get('ths_rank'):
            score += max(0, 25 - info['ths_rank'])     # 热榜排名加分
        if info.get('source') == 'limit_up':
            score += 5                                  # 涨停来源额外加分

        # 生命周期判断
        lifecycle, trend_detail = estimate_lifecycle(concept, limit_count, history, date)
        info['lifecycle'] = lifecycle
        info['trend_detail'] = trend_detail
        info['lifecycle_score'] = LIFECYCLE_SCORE.get(lifecycle, 30)

        # 热度分 = 基础分 × 生命周期系数
        # 发酵期×1.5(鼓励介入), 高潮期×0.3(警告), 退潮期×0.1(别碰)
        lifecycle_mult = {
            '发酵期': 1.5, '启动期': 1.2, '萌芽期': 1.0,
            '高潮期': 0.3, '分化期': 0.5, '退潮期': 0.1,
        }
        info['heat_score'] = round(score * lifecycle_mult.get(lifecycle, 1.0))

    # 更新历史
    update_hotspot_history(hotspot, date)

    # 排序
    hotspot['ranked'] = sorted(
        [(k, v['heat_score']) for k, v in hotspot['concepts'].items()],
        key=lambda x: -x[1]
    )[:50]  # 只保留TOP50热点

    # 保存
    with open(HOTSPOT_FILE, 'w', encoding='utf-8') as f:
        json.dump(hotspot, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ 热点保存: {HOTSPOT_FILE}")
    print(f"  TOP10热点概念(含生命周期):")
    lifecycle_emoji = {'发酵期':'🔥','启动期':'🚀','萌芽期':'🌱','高潮期':'⚠️','分化期':'📉','退潮期':'💀'}
    for i, (concept, score) in enumerate(hotspot['ranked'][:10], 1):
        info = hotspot['concepts'][concept]
        lc = info.get('lifecycle', '?')
        emoji = lifecycle_emoji.get(lc, '?')
        detail = info.get('trend_detail', '')
        print(f"    {i:>2}. {concept:<12} 热度{score:>3}  涨停{info.get('limit_count',0)}只  "
              f"{emoji}{lc}  {detail}")

    return hotspot


# ==============================================================================
# 离线数据加载
# ==============================================================================

def load_concept_mapping(as_of=''):
    """加载概念映射表"""
    if as_of:
        from mainwave_engine.theme_semantics import load_mapping_as_of
        text = str(as_of)
        date = text if '-' in text else f"{text[:4]}-{text[4:6]}-{text[6:8]}"
        mapping, _ = load_mapping_as_of(date)
        return mapping
    if not os.path.exists(MAPPING_FILE):
        return {}
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('mapping', {})


def load_hotspot():
    """加载热点数据"""
    if not os.path.exists(HOTSPOT_FILE):
        return {}
    with open(HOTSPOT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_event_config():
    """加载手动事件热点配置"""
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==============================================================================
# 热点检测层（从数据自动识别）
# ==============================================================================

def detect_hotspots_from_data(data_dir, mapping, days=3):
    """
    从日线数据自动检测热点（不依赖Tushare API）
    逻辑: 统计最近N天涨停股的概念分布 → 出现频率最高的概念=热点
    """
    csv_files = glob.glob(os.path.join(data_dir, '*_daily.csv'))
    csv_files = [f for f in csv_files if '000001' not in f and '000300' not in f and '399' not in f]

    concept_surge = defaultdict(lambda: {'limit_count': 0, 'surge_count': 0, 'stocks': []})

    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
            df.columns = df.columns.str.lower()
            if 'close' not in df.columns or 'date' not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)

            if len(df) < 10:
                continue

            stock = os.path.basename(fp).replace('.csv', '').replace('_daily', '')
            recent = df.tail(days)

            # 计算涨幅
            for _, row in recent.iterrows():
                prev_close = df[df['date'] < row['date']].tail(1)
                if prev_close.empty:
                    continue
                pct = (row['close'] / prev_close.iloc[0]['close'] - 1)

                if stock in mapping and mapping[stock].get('concepts'):
                    concepts = mapping[stock]['concepts']
                    if pct > 0.095:  # 涨停
                        for c in concepts:
                            concept_surge[c]['limit_count'] += 1
                            if stock not in concept_surge[c]['stocks']:
                                concept_surge[c]['stocks'].append(stock)
                    elif pct > 0.05:  # 大涨>5%
                        for c in concepts:
                            concept_surge[c]['surge_count'] += 1
        except:
            continue

    # 计算热度分
    hotspots = {}
    for concept, info in concept_surge.items():
        score = info['limit_count'] * 10 + info['surge_count'] * 3
        if score >= 10:  # 最低门槛
            hotspots[concept] = {
                'heat_score': score,
                'limit_count': info['limit_count'],
                'surge_count': info['surge_count'],
                'stocks': info['stocks'][:10],
                'source': 'data_detect',
            }

    return dict(sorted(hotspots.items(), key=lambda x: -x[1]['heat_score'])[:50])


# ==============================================================================
# 匹配层（股票概念×市场热点 → 热度分）
# ==============================================================================

def calc_concept_score(stock_code, mapping, hotspots, event_config=None):
    """
    计算单只股票的概念热度分（含生命周期加权）
    返回: (score, matched_concepts, detail, best_lifecycle)
    """
    if stock_code not in mapping:
        return 0, [], '无概念数据', ''

    stock_concepts = set(mapping[stock_code].get('concepts', []))
    if not stock_concepts:
        return 0, [], '无概念标签', ''

    # 获取热点概念集合
    hot_concepts = {}
    hot_lifecycles = {}  # concept -> lifecycle
    if isinstance(hotspots, dict):
        if 'concepts' in hotspots:
            for concept, info in hotspots['concepts'].items():
                hot_concepts[concept] = info.get('heat_score', 0)
                hot_lifecycles[concept] = info.get('lifecycle', '萌芽期')
        else:
            for concept, info in hotspots.items():
                hot_concepts[concept] = info.get('heat_score', 0)
                hot_lifecycles[concept] = info.get('lifecycle', '萌芽期')

    # 加入手动事件热点
    if event_config:
        for event_name, event_data in event_config.items():
            for concept in event_data.get('related_concepts', []):
                bonus = event_data.get('heat_bonus', 20)
                hot_concepts[concept] = hot_concepts.get(concept, 0) + bonus
                if concept not in hot_lifecycles:
                    hot_lifecycles[concept] = '启动期'  # 手动事件默认启动期

    # 计算匹配
    matched = stock_concepts & set(hot_concepts.keys())
    if not matched:
        return 0, [], f'概念{len(stock_concepts)}个但无热点匹配', ''

    # 热度分 = 匹配概念的热度之和（已含生命周期加权）
    score = sum(hot_concepts[c] for c in matched)
    if len(matched) >= 3:
        score *= 1.5
    elif len(matched) >= 2:
        score *= 1.2

    # 找最佳生命周期（取评分最高的概念的生命周期）
    best_lc = ''
    best_lc_score = 0
    for c in matched:
        lc = hot_lifecycles.get(c, '萌芽期')
        lcs = LIFECYCLE_SCORE.get(lc, 30)
        if lcs > best_lc_score:
            best_lc_score = lcs
            best_lc = lc

    lifecycle_emoji = {'发酵期':'🔥','启动期':'🚀','萌芽期':'🌱','高潮期':'⚠️','分化期':'📉','退潮期':'💀'}
    lc_str = f"{lifecycle_emoji.get(best_lc,'')}{best_lc}" if best_lc else ''
    detail = '+'.join(sorted(matched, key=lambda c: -hot_concepts[c])[:3])
    if lc_str:
        detail = f'{detail}({lc_str})'

    return round(score, 1), list(matched), detail, best_lc


def batch_calc_concept_scores(scan_csv, mapping, hotspots, event_config=None):
    """批量计算scan结果中所有股票的概念热度"""
    df = pd.read_csv(scan_csv)
    results = []

    for _, row in df.iterrows():
        stock = row['股票']
        score, matched, detail, lifecycle = calc_concept_score(stock, mapping, hotspots, event_config)
        results.append({
            'stock': stock,
            'concept_score': score,
            'matched_count': len(matched),
            'matched_concepts': detail,
            'best_lifecycle': lifecycle,
            'priority': row.get('优先级', 9),
            'priority_tag': row.get('优先级标签', ''),
            'gain': row.get('信号后涨幅', 0),
        })

    return pd.DataFrame(results)


# ==============================================================================
# 报告输出
# ==============================================================================

def print_hotspot_report(hotspots, mapping=None):
    """输出热点概念报告"""
    print(f"\n{'═' * 70}")
    print(f"  当日市场热点概念")
    print(f"{'═' * 70}")

    ranked = hotspots.get('ranked', [])
    if not ranked:
        ranked = sorted([(k, v['heat_score']) for k, v in hotspots.items()], key=lambda x: -x[1])

    lifecycle_emoji = {'发酵期':'🔥','启动期':'🚀','萌芽期':'🌱','高潮期':'⚠️','分化期':'📉','退潮期':'💀'}

    for i, item in enumerate(ranked[:20], 1):
        if isinstance(item, (list, tuple)):
            concept, score = item[0], item[1]
        else:
            continue

        info = hotspots.get('concepts', hotspots).get(concept, {})
        limit_cnt = info.get('limit_count', 0)
        lc = info.get('lifecycle', '?')
        emoji = lifecycle_emoji.get(lc, '?')
        detail = info.get('trend_detail', '')

        bar = '█' * min(int(score / 5), 15)
        print(f"  {i:>2}. {concept:<12} 热度{score:>4.0f}  涨停{limit_cnt:>2}只  {emoji}{lc:<4} {detail:<20} {bar}")


def print_match_report(results_df, hotspots):
    """输出概念匹配报告"""
    print(f"\n{'═' * 70}")
    print(f"  S/A级股票 × 概念热点匹配")
    print(f"{'═' * 70}")

    sa = results_df[results_df['priority'].isin([1, 2])].sort_values('concept_score', ascending=False)

    if len(sa) == 0:
        print("  无S/A级股票")
        return

    # 有热点匹配的
    matched = sa[sa['concept_score'] > 0]
    unmatched = sa[sa['concept_score'] == 0]

    print(f"\n  🔥 有热点加成的S/A级 ({len(matched)}只):")
    print(f"  {'股票':<14} {'等级':<10} {'热度':>5} {'匹配概念':<30} {'涨幅':>6}")
    print(f"  {'─' * 70}")
    for _, r in matched.head(20).iterrows():
        print(f"  {r['stock']:<14} {r['priority_tag']:<10} {r['concept_score']:>5.0f} "
              f"{r['matched_concepts']:<30} {r['gain']:>+5.1f}%")

    if len(unmatched) > 0:
        print(f"\n  ⚪ 无热点匹配的S/A级 ({len(unmatched)}只)")


# ==============================================================================
# 手动事件热点配置
# ==============================================================================

def create_default_config():
    """创建默认事件热点配置模板"""
    config = {
        "_说明": "手动配置当前事件驱动的热点概念，related_concepts填概念名称",
        "_更新频率": "根据新闻事件手动更新，通常每周1~2次",
        "伊朗局势": {
            "heat_bonus": 30,
            "related_concepts": ["石油", "天然气", "化工", "军工", "黄金"],
            "start_date": "2026-03-25",
            "note": "中东紧张局势→能源+避险"
        },
        "AI大模型": {
            "heat_bonus": 20,
            "related_concepts": ["人工智能", "算力", "芯片", "服务器", "数据中心"],
            "start_date": "2026-01-01",
            "note": "持续性主线"
        },
    }
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  事件配置模板已创建: {CONFIG_FILE}")
    print(f"  请根据当前市场情况编辑此文件")
    return config


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    # 任何控制台(含 Windows GBK)下 emoji print 不再崩溃 (直接运行时生效)。
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description='概念热点匹配器')
    parser.add_argument('--fetch-mapping', action='store_true', help='从Tushare获取概念映射')
    parser.add_argument('--fetch-hotspot', action='store_true', help='从Tushare获取今日热点')
    parser.add_argument('--detect', action='store_true', help='从日线数据自动检测热点(无需API)')
    parser.add_argument('--analyze', action='store_true', help='分析S级股票的概念匹配度')
    parser.add_argument('--stock', type=str, default='', help='查看单只股票的概念信息')
    parser.add_argument('--scan-csv', type=str, default='', help='扫描结果CSV路径')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--date', type=str, default='', help='日期(YYYYMMDD)')
    parser.add_argument('--init-config', action='store_true', help='创建事件热点配置模板')
    args = parser.parse_args()

    # 初始化事件配置
    if args.init_config:
        create_default_config()
        return

    # 获取概念映射
    if args.fetch_mapping:
        pro = get_tushare_pro()
        if pro:
            return 0 if fetch_concept_mapping(pro) else 1
        return 1

    # 获取每日热点
    if args.fetch_hotspot:
        pro = get_tushare_pro()
        if pro:
            date = args.date or datetime.now().strftime('%Y%m%d')
            fetch_daily_hotspot(pro, date)
        return

    # 从数据自动检测热点
    if args.detect:
        mapping = load_concept_mapping()
        if not mapping:
            print("  ❌ 请先运行 --fetch-mapping 获取概念映射")
            return
        hotspots = detect_hotspots_from_data(args.data_dir, mapping)
        print_hotspot_report(hotspots)

        # 保存检测结果
        with open(HOTSPOT_FILE, 'w', encoding='utf-8') as f:
            json.dump({'date': datetime.now().strftime('%Y-%m-%d'),
                       'concepts': hotspots, 'source': 'data_detect'}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n  热点已保存: {HOTSPOT_FILE}")
        return

    # 查看单只股票
    if args.stock:
        mapping = load_concept_mapping()
        if args.stock in mapping:
            info = mapping[args.stock]
            print(f"\n  {args.stock} ({info.get('name','')})")
            print(f"  行业: {info.get('industry','')}")
            print(f"  概念({len(info.get('concepts',[]))}个):")
            for c in info.get('concepts', []):
                print(f"    · {c}")
        else:
            print(f"  {args.stock} 不在映射表中（请先运行 --fetch-mapping）")
        return

    # 分析概念匹配度
    if args.analyze:
        mapping = load_concept_mapping()
        hotspots = load_hotspot()
        event_config = load_event_config()

        if not mapping:
            print("  ❌ 请先运行 --fetch-mapping 获取概念映射")
            return
        if not hotspots:
            print("  ⚠️  无热点数据，尝试从日线数据自动检测...")
            hotspots = detect_hotspots_from_data(args.data_dir, mapping)

        if args.scan_csv:
            results = batch_calc_concept_scores(args.scan_csv, mapping, hotspots, event_config)
            print_hotspot_report(hotspots)
            print_match_report(results, hotspots)

            # 保存匹配结果到CSV（合并回原始扫描数据）
            scan_df = pd.read_csv(args.scan_csv)
            concept_map = {}
            for _, r in results.iterrows():
                concept_map[r['stock']] = (r['concept_score'], r['matched_concepts'],
                                            r['matched_count'], r.get('best_lifecycle',''))
            scan_df['概念热度'] = scan_df['股票'].map(lambda x: concept_map.get(x, (0,'',0,''))[0])
            scan_df['匹配概念'] = scan_df['股票'].map(lambda x: concept_map.get(x, (0,'',0,''))[1])
            scan_df['概念命中数'] = scan_df['股票'].map(lambda x: concept_map.get(x, (0,'',0,''))[2])
            scan_df['概念生命周期'] = scan_df['股票'].map(lambda x: concept_map.get(x, (0,'',0,''))[3])

            out_name = args.scan_csv.replace('.csv', '_concept.csv')
            scan_df.to_csv(out_name, index=False, encoding='utf-8-sig')
            print(f"\n  ✅ 概念匹配结果已保存: {out_name}")
            print(f"     新增列: 概念热度 / 匹配概念 / 概念命中数 / 概念生命周期")

            # 也单独保存热点匹配的S/A级列表
            sa_hot = results[(results['priority'].isin([1,2])) & (results['concept_score'] > 0)]
            if len(sa_hot) > 0:
                hot_name = args.scan_csv.replace('.csv', '_hotspot_SA.csv')
                sa_hot.sort_values('concept_score', ascending=False).to_csv(hot_name, index=False, encoding='utf-8-sig')
                print(f"     热点S/A级单独保存: {hot_name} ({len(sa_hot)}只)")
        else:
            print_hotspot_report(hotspots)
        return

    # 默认: 打印帮助
    parser.print_help()
    print(f"\n工作流程:")
    print(f"  Step 1: python concept_hotspot.py --fetch-mapping              # 首次/每月")
    print(f"  Step 2: python concept_hotspot.py --fetch-hotspot --date 20260328  # 每天盘后")
    print(f"  Step 3: python concept_hotspot.py --analyze --scan-csv xxx.csv     # 配合扫描")
    print(f"\n快捷方式:")
    print(f"  python concept_hotspot.py --detect --data-dir data    # 无需API，从数据自动检测")
    print(f"  python concept_hotspot.py --init-config               # 创建事件热点配置模板")
    print(f"  python concept_hotspot.py --stock sz_002192           # 查看单股概念")


if __name__ == "__main__":
    raise SystemExit(main())
