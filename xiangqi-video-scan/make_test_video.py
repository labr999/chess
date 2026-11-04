# 產生模擬「2D 棋盤教學影片」做自我測試（紅方在上、有字幕遮擋、有退回變化、有滑動動畫）
import cv2, numpy as np, sys
from xq_core import Position, START_FEN

W, H, FPS = 1280, 720, 25
X0, Y0, S = 290, 45, 70
FLIP = True


def vpos(r, c):
    if FLIP:
        r, c = 9 - r, 8 - c
    return X0 + c * S, Y0 + r * S


def board_bg():
    img = np.full((H, W, 3), (200, 200, 205), np.uint8)
    cv2.rectangle(img, (X0 - 45, Y0 - 40), (X0 + 8 * S + 45, Y0 + 9 * S + 40), (150, 200, 235), -1)
    for r in range(10):
        cv2.line(img, (X0, Y0 + r * S), (X0 + 8 * S, Y0 + r * S), (60, 90, 120), 2)
    for c in range(9):
        cv2.line(img, (X0 + c * S, Y0), (X0 + c * S, Y0 + 4 * S), (60, 90, 120), 2)
        cv2.line(img, (X0 + c * S, Y0 + 5 * S), (X0 + c * S, Y0 + 9 * S), (60, 90, 120), 2)
    cv2.putText(img, 'CHU HE      HAN JIE', (X0 + 80, Y0 + int(4.6 * S)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 110, 130), 2)
    # 右下角主播視窗
    cv2.rectangle(img, (900, 400), (1280, 720), (90, 70, 60), -1)
    cv2.circle(img, (1080, 520), 60, (150, 170, 200), -1)
    return img


BG = board_bg()


def draw_piece(img, x, y, p):
    cv2.circle(img, (int(x), int(y)), 31, (120, 170, 200), -1)
    cv2.circle(img, (int(x), int(y)), 28, (175, 215, 240), -1)
    cv2.circle(img, (int(x), int(y)), 24, (100, 140, 170), 2)
    col = (30, 30, 200) if p.isupper() else (25, 30, 25)
    cv2.putText(img, p.upper(), (int(x) - 13, int(y) + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)


def render(pos, moving=None, frac=0.0, overlay=False):
    img = BG.copy()
    for r in range(10):
        for c in range(9):
            p = pos.b[r][c]
            if p and not (moving and (r, c) == moving[0]):
                draw_piece(img, *vpos(r, c), p)
    if moving:
        (r1, c1), (r2, c2) = moving
        x1, y1 = vpos(r1, c1); x2, y2 = vpos(r2, c2)
        draw_piece(img, x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac, pos.b[r1][c1])
    if overlay:
        cv2.putText(img, 'PO NIU TOU', (330, 260), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 0), 14)
        cv2.putText(img, 'PO NIU TOU', (330, 260), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 255, 255), 5)
    return img


def iccs(pos, s):
    c1, r1, c2, r2 = 'abcdefghi'.index(s[0]), 9 - int(s[1]), 'abcdefghi'.index(s[2]), 9 - int(s[3])
    mv = ((r1, c1), (r2, c2))
    assert mv in pos.legal_moves(), s
    return mv


def main(out):
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    pos = Position.from_fen(START_FEN)
    hist = [pos]

    def hold(p, sec, overlay=False):
        for _ in range(int(sec * FPS)):
            vw.write(render(p, overlay=overlay))

    def play(s, sec=1.2):
        nonlocal pos
        mv = iccs(pos, s)
        for k in range(5):
            vw.write(render(pos, mv, (k + 1) / 6))
        pos = pos.make(mv); hist.append(pos)
        hold(pos, sec)

    # 片頭：主播畫面（沒有棋盤）
    for _ in range(FPS * 2):
        vw.write(np.full((H, W, 3), (80, 60, 50), np.uint8))
    hold(pos, 2)
    main_line = ['h2e2', 'h9g7', 'h0g2', 'i9h9', 'i0h0', 'b9c7', 'e3e4', 'g6g5']
    for i, m in enumerate(main_line):
        play(m)
        if i == 3:
            hold(pos, 1.0, overlay=True)  # 大字幕擋住棋盤
    # 退回 3 步示範變化
    for _ in range(3):
        hist.pop(); pos = hist[-1]; hold(pos, 0.8)
    for m in ['b9c7', 'b0c2']:
        play(m)
    # 再回主線並續走兩步（快速連走：0.2 秒一步）
    hist.pop(); hist.pop(); pos = hist[-1]; hold(pos, 0.8)
    play('b9c7', 0.2); play('e3e4', 0.2); play('g6g5', 1.0)
    play('e4e5', 1.0); play('g5g4', 1.5)
    vw.release()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'test.mp4')
