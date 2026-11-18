# 合成測試（v1.5.1）：只亮開局 → 老師直接擺出第 14 步的中局 → 續走 10 步（沒有主棋譜）
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import make_test_video as M, cv2
from xq_core import Position, START_FEN
out = sys.argv[1]
L = ['h2e2','h9g7','h0g2','i9h9','i0h0','b9c7','e3e4','g6g5','b0c2','b7b3','a0a1','a9b9','a1d1','c6c5']
CONT = ['d1d6','g7f5','g2f4','f5d4','c2e1','a6a5','h0h4','c9e7','d6d4','b3b4']
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), M.FPS, (M.W, M.H))
pos = Position.from_fen(START_FEN)
def hold(p, sec):
    for _ in range(int(sec * M.FPS)): vw.write(M.render(p))
def play(s, sec=1.2):
    global pos
    mv = M.iccs(pos, s)
    assert mv in pos.legal_moves(), s
    for k in range(5): vw.write(M.render(pos, mv, (k + 1) / 6))
    pos = pos.make(mv); hold(pos, sec)
hold(pos, 4)
setup = Position.from_fen(START_FEN)
for s in L: setup = setup.make(M.iccs(setup, s))
for k in range(12):
    mix = setup.copy()
    for r in range(10):
        for c in range(9):
            if (r * 9 + c + k) % 5 == 0: mix.b[r][c] = pos.b[r][c]
    hold(mix, 0.5)
pos = setup; hold(pos, 8)
import random; random.seed(7)
for _ in range(10):
    ms = [m for m in pos.legal_moves() if pos.b[m[1][0]][m[1][1]] is None]
    mv = random.choice(sorted(ms))
    play('%s%d%s%d' % ('abcdefghi'[mv[0][1]], 9 - mv[0][0], 'abcdefghi'[mv[1][1]], 9 - mv[1][0]))
hold(pos, 3); vw.release()
print('setup FEN', setup.fen())
