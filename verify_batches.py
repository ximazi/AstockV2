"""校验合并 txt 里的每个文件与磁盘上的源文件逐字节一致。

用法: python verify_batches.py [批次号...]
不带参数则校验全部批次。返回码非 0 表示有文件对不上。
"""
import hashlib
import re
import sys
from pathlib import Path

from merge_batches import BATCHES

BLOCK = re.compile(
    r'={100}\nFILE: (?P<rel>.+?)\n'
    r'LINES: (?P<lines>\d+)  SHA256\[:16\]: (?P<digest>\w+)  BYTES: (?P<bytes>\d+)\n'
    r'SOURCE: (?P<source>.+?)\n'
    r'={100}\n'
)


def verify(key) -> list:
    spec = BATCHES[key]
    txt = Path('D:/AICode/AStockV2') / spec['out']
    raw = txt.read_bytes()
    text = raw.decode('utf-8')
    problems = []
    matches = list(BLOCK.finditer(text))
    if len(matches) != len(spec['files']):
        problems.append(f'文件块数不符: {len(matches)} != {len(spec["files"])}')
    for i, m in enumerate(matches):
        rel = m.group('rel')
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        # 合并时给没有结尾换行的文件补了一个 \n，切回时要还回去
        src = Path(m.group('source'))
        src_bytes = src.read_bytes()
        if body.encode('utf-8') != src_bytes:
            if body.rstrip('\n').encode('utf-8') + b'\n' == src_bytes:
                body = body.rstrip('\n') + '\n'   # 补过换行, 仍视为一致
            else:
                problems.append(f'{rel}: 内容与源文件不一致')
                continue
        digest = hashlib.sha256(body.encode('utf-8')).hexdigest()[:16]
        if digest != m.group('digest'):
            problems.append(f'{rel}: SHA256 不符 {digest} != {m.group("digest")}')
        if len(body.encode('utf-8')) != int(m.group('bytes')):
            problems.append(f'{rel}: 字节数不符')
        if len(src_bytes) != int(m.group('bytes')):
            problems.append(f'{rel}: 源文件字节数与标注不符')
    problems += verify_toc(text)
    return problems


def verify_toc(text: str) -> list:
    """目录里写的行号必须正好落在 "FILE:" 标记上。"""
    problems = []
    lines = text.split('\n')
    for line in lines:
        m = re.match(r'^\s*(\d+)\s+(\d+) 行  (.+)$', line)
        if not m:
            continue
        start, rel = int(m.group(1)), m.group(3)
        if start < 1 or start > len(lines):
            problems.append(f'{rel}: 目录行号 {start} 越界')
        elif lines[start - 1] != f'FILE: {rel}':
            problems.append(f'{rel}: 目录行号 {start} 不在 FILE 标记上 (实际 {lines[start-1]!r})')
    return problems


def main() -> None:
    keys = [int(a) for a in sys.argv[1:]] or sorted(BATCHES)
    bad = 0
    for key in keys:
        spec = BATCHES[key]
        problems = verify(key)
        status = 'OK' if not problems else 'FAIL'
        print(f'第 {key} 批 {spec["out"]}: {status} ({len(spec["files"])} 个文件)')
        for p in problems:
            print(f'    - {p}')
        bad += len(problems)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
