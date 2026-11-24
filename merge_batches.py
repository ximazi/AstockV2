"""把评审批次合并成单个 .txt。

第 1 批（主引擎核心代码）由 merge_core.py 生成，本脚本只管第 2/3/4 批。

每个源文件的内容逐字复制（按字节保留 CRLF），文件之间用分隔标记隔开，
可以按 "FILE:" 标记切回单个文件：

    python merge_batches.py            # 生成全部三批
    python merge_batches.py 2 3        # 只生成指定批次
"""
import hashlib
import sys
from pathlib import Path

V1 = Path('D:/AICode/Astock')
V2 = Path('D:/AICode/AStockV2')
OUT_DIR = V2

SEP = '=' * 100

BATCHES = {
    2: {
        'title': '第 2 批 · 信号与验收审计（8 个文件）',
        'out': 'mainwave_batch2_signal_audit_20261124.txt',
        'root': V1,
        'source_label': 'D:/AICode/Astock（v1 工作区，含未提交改动的当前状态）',
        'focus': ('R1 83.5% 漏失窗口与归因口径、原突破位与 2ATR 实现、'
                  'S4 回放是否存在未来泄漏及排除月份条件'),
        'files': [
            'mainwave_engine/route_evidence.py',
            'mainwave_engine/breakout_events.py',
            'mainwave_engine/entry_policy.py',
            'mainwave_engine/exit_policy.py',
            'research/screening_recall_audit.py',
            'research/r1_open_hold_audit.py',
            'research/r1_extension_audit.py',
            'research/s4_selection_replay.py',
        ],
    },
    3: {
        'title': '第 3 批 · 交易执行与风控门（6 个文件）',
        'out': 'mainwave_batch3_trading_gates_20261124.txt',
        'root': V1,
        'source_label': 'D:/AICode/Astock（v1 工作区，含未提交改动的当前状态）',
        'focus': ('缺失数据降级是否阻断下单、AUTO100W/NEW100K/旧虚拟账隔离、'
                  '盘后退出与大盘/中期趋势门逻辑'),
        'notes': ('本批取自 v1 工作区当前磁盘状态。'
                  '其中 mainwave_engine/trading/daily.py 与 mainwave_engine/index_trend.py '
                  '在 mainwave_code_v2.26.0_20261124 快照之后又被改过'
                  '（跌停价参考读取、ST_LIMIT_REFERENCE_WAIT 拦截、'
                  '指数日线收盘价有效性校验），与快照里的同名文件不同，'
                  '本文件以较新的磁盘内容为准。'),
        'files': [
            'mainwave_engine/trading/daily.py',
            'mainwave_engine/trading/preflight.py',
            'mainwave_engine/trading/books.py',
            'mainwave_engine/trading/position_monitor.py',
            'mainwave_engine/index_trend.py',
            'mainwave_engine/virtual_portfolio.py',
        ],
    },
    4: {
        'title': '第 4 批 · 证伪台与验收结论（8 个文件）',
        'out': 'mainwave_batch4_acceptance_20261124.txt',
        'root': V2,
        'source_label': 'D:/AICode/AStockV2（证伪台工作区）',
        'focus': ('独立验收引擎的撮合假设、对照组严谨性，'
                  '以及与主引擎 mainwave_engine 的口径一致性'),
        'files': [
            'docs/核心问题诊断与验收结论_20261120.md',
            'docs/市场环境分层与入场时机_20261120.md',
            'results/verdict/verdict.md',
            'astockv2/execution.py',
            'astockv2/control.py',
            'astockv2/s4_arm.py',
            'astockv2/regime.py',
            'astockv2/verdict.py',
        ],
    },
}


def read_raw(path: Path) -> str:
    """按字节读入并保留原有换行符（不做 universal-newline 转换）。"""
    with open(path, 'r', encoding='utf-8', newline='') as fh:
        return fh.read()


def build(spec) -> tuple:
    root = spec['root']
    parts = []
    counts = []
    total_lines = 0
    total_bytes = 0

    for rel in spec['files']:
        p = root / rel
        if not p.is_file():
            raise SystemExit(f'缺失文件: {p}')
        text = read_raw(p)
        raw = text.encode('utf-8')
        lines = text.count('\n')
        digest = hashlib.sha256(raw).hexdigest()[:16]
        header = (f'{SEP}\n'
                  f'FILE: {rel}\n'
                  f'LINES: {lines}  SHA256[:16]: {digest}  BYTES: {len(raw)}\n'
                  f'SOURCE: {p}\n'
                  f'{SEP}\n')
        body = text if text.endswith('\n') else text + '\n'
        parts.append(header + body)
        counts.append((rel, lines))
        total_lines += lines
        total_bytes += len(raw)

    prefix = [f'大富翁系统 评审批次合并 · {spec["title"]}', '']
    prefix.append('生成日期: 2026-11-24')
    prefix.append(f'来源: {spec["source_label"]}')
    prefix.append('说明: 每个文件的内容逐字复制，未做任何改动；分隔线之间是完整文件内容。')
    prefix.append('      按 "FILE:" 标记可以把本文件切回单个文件。')
    prefix.append('')
    prefix.append(f'重点审计目标: {spec["focus"]}')
    prefix.append('')
    if spec.get('notes'):
        prefix.append(f'来源状态: {spec["notes"]}')
        prefix.append('')
    prefix.append('目录（起始行号指本文件里 "FILE:" 标记所在行）:')
    suffix = ['',
              f'合计 {len(spec["files"])} 个文件、{total_lines} 行、{total_bytes} 字节源码。',
              '']

    # 前言长度依赖于目录本身，所以先按行数算出偏移再回填目录
    def head_lines(start_of_first_file: int) -> list:
        toc = []
        cursor = start_of_first_file
        for rel, lines in counts:
            toc.append(f'  {cursor:>6}  {lines:>5} 行  {rel}')
            cursor += lines + 6      # 5 行分隔头 + 正文行数
        return prefix + toc + suffix

    # 前言 join 之后最后一行是空串, 所以正文第一行 = 分隔线行, 第二行才是 FILE:
    first = len(head_lines(0)) + 1
    head = head_lines(first)

    return '\n'.join(head) + '\n'.join(parts), total_lines


def main() -> None:
    keys = [int(a) for a in sys.argv[1:]] or sorted(BATCHES)
    for key in keys:
        if key not in BATCHES:
            raise SystemExit(f'没有第 {key} 批')
        spec = BATCHES[key]
        text, lines = build(spec)
        out = OUT_DIR / spec['out']
        with open(out, 'w', encoding='utf-8', newline='') as fh:
            fh.write(text)
        print(f'written: {out}  bytes={out.stat().st_size}  code_lines={lines}')


if __name__ == '__main__':
    main()
