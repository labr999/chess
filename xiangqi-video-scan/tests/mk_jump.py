# 合成測試：開局只停 2 秒就直接跳到中局（投影片式教學）
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import make_test_video as M, cv2
from xq_core import Position, START_FEN
out = sys.argv[1]
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), M.FPS, (M.W, M.H))
pos = Position.from_fen(START_FEN)
def hold(p, sec):
    for _ in range(int(sec * M.FPS)): vw.write(M.render(p))
def play(s, sec=1.2):
    global pos
    mv = M.iccs(pos, s)
    for k in range(5): vw.write(M.render(pos, mv, (k + 1) / 6))
    pos = pos.make(mv); hold(pos, sec)
hold(pos, 2)
pass
pass
setup = Position.from_fen('2bakab2/9/2n1c1n2/p1p1p3p/6p2/2P3R2/P3P3P/2N1C1N2/4A4/2B1KAB2 b')
for k in range(12):
    mix = setup.copy()
    for r in range(10):
        for c in range(9):
            if (r * 9 + c + k) % 5 == 0: mix.b[r][c] = pos.b[r][c]
    hold(mix, 0.5)
pos = setup; hold(pos, 8)
for m in ['g7f5', 'g4g5', 'a6a5', 'g5g6', 'f5d4', 'e2e6', 'e7d7', 'g6g7']: play(m)
hold(pos, 3); vw.release()
