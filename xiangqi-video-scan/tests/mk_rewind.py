# 合成測試：開局走 8 步 → 老師擺出中局走 8 步 → 倒回開局第 4 步的局面，改走別的變化
# 預期：主線 8 步的棋譜裡多一條變化；擺局段落 8 步；不會多出第三段。
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
def morph(dst, n=12):
    for k in range(n):
        mix = dst.copy()
        for r in range(10):
            for c in range(9):
                if (r * 9 + c + k) % 5 == 0: mix.b[r][c] = pos.b[r][c]
        hold(mix, 0.5)
hold(pos, 2)
main = ['h2e2', 'h9g7', 'h0g2', 'i9h9', 'i0h0', 'b9c7', 'e3e4', 'g6g5']
snap = []
for m in main:
    snap.append(pos.copy()); play(m)
hold(pos, 10)
setup = Position.from_fen('2bakab2/9/2n1c1n2/p1p1p3p/6p2/2P3R2/P3P3P/2N1C1N2/4A4/2B1KAB2 b')
morph(setup); pos = setup; hold(pos, 8)
for m in ['g7f5', 'g4g5', 'a6a5', 'g5g6', 'f5d4', 'e2e6', 'e7d7', 'g6g7']: play(m)
hold(pos, 3)
back = snap[4]           # 主線走了 4 步之後的局面（輪紅走）
morph(back); pos = back; hold(pos, 8)
for m in ['b0c2', 'b9c7', 'a0b0', 'a9b9']: play(m)
hold(pos, 3); vw.release()
