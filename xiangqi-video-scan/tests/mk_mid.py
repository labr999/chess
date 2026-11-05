# 合成測試：影片一開始就是中局（沒有開局畫面），黑方先走 8 步。
# 先跑 make_test_video.py 那支（學字樣存到 _templates），再跑這支才認得出來。
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import make_test_video as M, cv2
from xq_core import Position
out = sys.argv[1]
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), M.FPS, (M.W, M.H))
pos = Position.from_fen('2bakab2/9/2n1c1n2/p1p1p3p/6p2/2P3R2/P3P3P/2N1C1N2/4A4/2B1KAB2 b')
def hold(p, sec):
    for _ in range(int(sec * M.FPS)): vw.write(M.render(p))
def play(s, sec=1.2):
    global pos
    mv = M.iccs(pos, s)
    for k in range(5): vw.write(M.render(pos, mv, (k + 1) / 6))
    pos = pos.make(mv); hold(pos, sec)
hold(pos, 20)
for m in ['g7f5', 'g4g5', 'a6a5', 'g5g6', 'f5d4', 'e2e6', 'e7d7', 'g6g7']: play(m)
hold(pos, 3); vw.release()
