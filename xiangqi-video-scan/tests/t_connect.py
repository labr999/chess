# v1.5.1 中局擺子接主棋譜：比對、補步、簡體／國字數字棋譜
import sys, os, time, types; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import video_scan as V
from xq_core import Position, START_FEN
S = Position.from_fen(START_FEN)
def play(iccs_list, pos=S):
    mvs, cn = [], []
    for s in iccs_list:
        c1, r1, c2, r2 = 'abcdefghi'.index(s[0]), 9 - int(s[1]), 'abcdefghi'.index(s[2]), 9 - int(s[3])
        mv = ((r1, c1), (r2, c2)); cn.append(pos.move_cn(mv)); mvs.append(mv); pos = pos.make(mv)
    return mvs, cn, pos
L = ['h2e2','h9g7','h0g2','i9h9','i0h0','b9c7','e3e4','g6g5','b0c2','b7b3','a0a1','a9b9','a1d1','c6c5','d1d6','g7f5']
mvs, cn, _ = play(L)
simp = [c.replace('車','车').replace('馬','马').replace('進','进').replace('炮','砲') for c in cn]
# 黑方改國字數字，紅方改阿拉伯數字（東萍格式差異）
mix = []
for i, c in enumerate(simp):
    t = c.translate(str.maketrans('１２３４５６７８９', '一二三四五六七八九')) if i % 2 else c.translate(str.maketrans('一二三四五六七八九', '123456789'))
    mix.append(t)
games = [{'id': 501, 'tokens': mix, 'rawExport': '{}'}, {'id': 502, 'tokens': cn[:10], 'rawExport': '{}'},
         {'id': 503, 'tokens': cn, 'rawExport': '{"startFen":"x","setupOnly":true}'}]
V._get_json = lambda base, path, **k: {'items': games}
a = types.SimpleNamespace(upload='http://x', out='/tmp/tconn'); os.makedirs(a.out, exist_ok=True)
ok = 0
# 1 完全相同局面
_, _, tgt = play(L[:14])
c = V.connect_setup(tgt, a); print('exact', c and (c['gid'], c['plies'], len(c['bridge'])))
ok += bool(c and c['gid'] == 501 and c['plies'] == 14 and not c['bridge'])
# 2 只差 3 步（小棋聖沒有這一路）
_, _, tgt = play(L[:12] + ['c3c4', 'h7h3', 'g3g4'])
t0 = time.time(); c = V.connect_setup(tgt, a); print('bridge', c and (c['gid'], c['plies'], len(c['bridge'])), '%.1fs' % (time.time() - t0))
ok += bool(c and c['plies'] + len(c['bridge']) == 15)
# 3 接不上
tgt = Position.from_fen('4k4/9/9/9/9/9/9/9/9/4K4 w')
c = V.connect_setup(tgt, a); print('none', c); ok += c is None
# 4 setup_chain 接起來
_, _, tgt = play(L[:14]); sg = V.Node(tgt, t=50.0)
m2, c2, p2 = play(L[14:], tgt); cur = sg
for mv, cc in zip(m2, c2):
    n = V.Node(cur.pos.make(mv), mv, cc, cur, 60.0); cur.children = [n]; cur = n
br, ex = V.setup_chain(sg, V.connect_setup(tgt, a))
line = [br] + V.main_line(br); print('chain', len(line), ex); ok += len(line) == 16 and line[-1].pos.fen() == p2.fen()
print('PASS %d/4' % ok)
