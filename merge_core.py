import hashlib
from pathlib import Path

SRC = Path('D:/AICode/Astock')
OUT = Path('D:/AICode/AStockV2/mainwave_core_20261124.txt')

FILES = [
    'mainwave_engine/trading/paper.py',
    'mainwave_engine/trading/ranking.py',
    'mainwave_engine/trading/ranking_shadow.py',
    'mainwave_engine/trading/allocation.py',
    'mainwave_engine/trading/rules.py',
    'mainwave_engine/s4_scan.py',
    'mainwave_engine/route_scan.py',
]

SEP = '=' * 100
parts = []
toc = []
offset = 0
total_lines = 0

for rel in FILES:
    p = SRC / rel
    text = p.read_text(encoding='utf-8')
    lines = text.count('\n') + (0 if text.endswith('\n') else 1)
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    header = (f'{SEP}\n'
              f'FILE: {rel}\n'
              f'LINES: {lines}  SHA256[:16]: {digest}\n'
              f'SOURCE: {p}\n'
              f'{SEP}\n')
    body = text if text.endswith('\n') else text + '\n'
    parts.append(header + body)
    toc.append((rel, lines, offset + 1))
    offset += lines + 5
    total_lines += lines

head = [f'大富翁系统 主引擎核心代码合并（{len(FILES)} 个文件）', '']
head.append('生成日期: 2026-11-24')
head.append('来源: D:/AICode/Astock（v1 工作区，未提交改动的当前状态）')
head.append('说明: 每个文件的内容逐字复制，未做任何改动；分隔线之间是完整文件内容。')
head.append('      按 "FILE:" 标记可以把本文件切回单个 .py。')
head.append('')
head.append('目录（起始行号以本文件为准）:')
for rel, lines, start in toc:
    head.append(f'  {start:>6}  {lines:>5} 行  {rel}')
head.append('')
head.append(f'合计 {total_lines} 行代码。')
head.append('')
head.append(SEP)
head.append('')

OUT.write_text('\n'.join(head) + '\n'.join(parts), encoding='utf-8')
print(f'written: {OUT}  bytes={OUT.stat().st_size}  code_lines={total_lines}')
