# -*- coding: utf-8 -*-
"""接續／倒回判定測試：用 2026-11-06 重點棋譜 CSV 裡被拆開的幾盤（NUVn、m3ek、cLa9、hs3M、iopA）。
python tests/reach_check.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video_scan import Tracker, reach, START, main_line
from xq_core import Position

def play(pos, cn_list):
    out = []
    for cn in cn_list:
        for mv in pos.legal_moves():
            if pos.move_cn(mv) == cn:
                pos = pos.make(mv); out.append((mv, pos)); break
        else:
            raise SystemExit('走不出：%s' % cn)
    return out

def tr_play(tr, cns):
    for mv, _ in play(tr.cur.pos, cns):
        tr._apply(('move', [mv]), 0)

ok = True
def check(name, cond):
    global ok
    print(('OK  ' if cond else 'NG  ') + name); ok &= bool(cond)

NUVn = '炮二平五 馬８進７ 兵三進一 卒３進１ 馬二進三 車９平８ 車一平二 馬２進３ 炮八平七 馬３進２'.split()
NUVn1 = '馬三進四 象３進５ 馬四進五 馬７進５ 炮五進四 士４進５ 車二進五 車１平３ 兵七進一 車３進３ 炮五退一 卒３進１'.split()
# 1) 接續：主線 10 步後漏看 k 步
for k in (2, 3, 5, 7, 8):
    tr = Tracker(); tr_play(tr, NUVn)
    tgt = play(tr.cur.pos, NUVn1[:k])[-1][1]
    t0 = time.time(); act = tr.match(tgt)
    good = isinstance(act, tuple) and act[0] == 'move'
    if good:
        tr._apply(act, 0)
        good = tr.cur.pos.b == tgt.b and len(main_line(tr.root)) == 10 + k and not tr.segments
    check('NUVn 漏看 %d 步 → 接回主線（%.2f 秒）' % (k, time.time() - t0), good)

# 2) 倒回開局重播（cLa9：主線 70 步之後回到開局）
cLa9 = '炮二平五 馬８進７ 馬二進三 卒７進１ 車一平二 車９平８ 車二進六 馬２進３ 兵七進一 象３進５ 馬八進七 炮８平９ 車二平三 車８進２'.split()
tr = Tracker(); tr_play(tr, cLa9)
for _ in range(20):
    tr_play(tr, ['炮八平九', '炮２平１', '炮九平八', '炮１平２'])  # 拉長主線到 94 步
check('主線 %d 步' % len(main_line(tr.root)), len(main_line(tr.root)) > 60)
act = tr.match(START)
check('倒回開局 → back 到根', act == ('back', tr.root))

# 3) hs3M：主線 → 擺局段落 → 倒回主線第 5 步重走、第 16 步分岔
hs = '炮二平五 馬２進３ 馬二進三 象３進５ 車一平二 炮８平６ 兵七進一 士４進５ 炮八平六 車１平４ 仕六進五 卒７進１ 馬八進七 馬８進７ 車九平八 炮２退２ 車二進四 馬７進６ 炮五進四 馬３進５'.split()
tr = Tracker(); tr_play(tr, hs)
setup = Position.from_fen('6b2/3N1k3/4bc1r1/p1p1P3p/3N5/2Pnn1p2/P3C3P/9/4A4/2B1KAB2 b - - 0 1')
tr.reanchor(setup, 0)
tgt = play(START, hs[:5])[-1][1]
act = tr.match(tgt)
check('擺局後倒回主線第 5 步 → back', isinstance(act, tuple) and act[0] == 'back' and act[1].pos.b == tgt.b)
tr._apply(act, 0)
tr_play(tr, hs[5:16] + ['車八進三'])
br = tr.root.children[0]
node = br
for _ in range(15):
    node = node.children[0]
check('第 16 步分岔存成同一盤的變化', len(node.children) == 2 and len(tr.root.children) == 1)

# 4) 不相干的局面仍視為新擺局
tr = Tracker(); tr_play(tr, NUVn)
t0 = time.time(); act = tr.match(setup)
check('完全不同局面 → None（新擺局，%.2f 秒）' % (time.time() - t0), act is None)
# 5) 主線結束局面又被當成擺局讀到（NUVn_1 的情形）→ back 到主線，不另開段落
tr = Tracker(); tr_play(tr, NUVn); end = tr.cur
tr.reanchor(setup, 0)
act = tr.match(end.pos)
check('擺局段落中讀到主線結束局面 → back', act == ('back', end))

print('全部通過' if ok else '有失敗')
sys.exit(0 if ok else 1)
