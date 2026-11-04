# -*- coding: utf-8 -*-
"""象棋核心：棋盤、合法著法、中文棋譜、FEN、開局分類。

座標：board[r][c]，r=0 為黑方底線（畫面上方，標準方向），r=9 為紅方底線；
c=0 為紅方視角的最左邊（紅方「九路」、黑方「1路」）。
棋子：大寫紅、小寫黑。K帥 A仕 B相 N馬 R車 C炮 P兵。
"""

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"

RED_NAME = {'K': '帥', 'A': '仕', 'B': '相', 'N': '馬', 'R': '車', 'C': '炮', 'P': '兵'}
BLK_NAME = {'k': '將', 'a': '士', 'b': '象', 'n': '馬', 'r': '車', 'c': '炮', 'p': '卒'}
CN_NUM = '零一二三四五六七八九'
FW_NUM = '０１２３４５６７８９'


def is_red(p):
    return p is not None and p.isupper()


def side_of(p):
    if p is None:
        return None
    return 'w' if p.isupper() else 'b'


class Position:
    __slots__ = ('b', 'turn')

    def __init__(self, b=None, turn='w'):
        self.b = b if b is not None else [[None] * 9 for _ in range(10)]
        self.turn = turn

    @staticmethod
    def from_fen(fen):
        parts = fen.split()
        rows = parts[0].split('/')
        b = []
        for row in rows:
            line = []
            for ch in row:
                if ch.isdigit():
                    line.extend([None] * int(ch))
                else:
                    line.append(ch)
            b.append(line)
        turn = parts[1] if len(parts) > 1 else 'w'
        if turn == 'r':
            turn = 'w'
        return Position(b, turn)

    def fen(self):
        out = []
        for row in self.b:
            s, e = '', 0
            for p in row:
                if p is None:
                    e += 1
                else:
                    if e:
                        s += str(e)
                        e = 0
                    s += p
            if e:
                s += str(e)
            out.append(s)
        return '/'.join(out) + ' ' + self.turn + ' - - 0 1'

    def copy(self):
        return Position([row[:] for row in self.b], self.turn)

    def key(self):
        return self.fen()

    def color_grid(self):
        """每格 'E' / 'R' / 'B'。"""
        return [['E' if p is None else ('R' if p.isupper() else 'B') for p in row] for row in self.b]

    def make(self, mv):
        (r1, c1), (r2, c2) = mv
        q = self.copy()
        q.b[r2][c2] = q.b[r1][c1]
        q.b[r1][c1] = None
        q.turn = 'b' if self.turn == 'w' else 'w'
        return q

    # ---------- 著法產生 ----------
    def _pseudo(self, side):
        b = self.b
        mine = str.isupper if side == 'w' else str.islower
        res = []
        for r in range(10):
            for c in range(9):
                p = b[r][c]
                if p is None or not mine(p):
                    continue
                t = p.upper()
                tgts = []
                if t == 'R' or t == 'C':
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        jumped = False
                        while 0 <= rr < 10 and 0 <= cc < 9:
                            q = b[rr][cc]
                            if t == 'R':
                                if q is None:
                                    tgts.append((rr, cc))
                                else:
                                    if not mine(q):
                                        tgts.append((rr, cc))
                                    break
                            else:
                                if not jumped:
                                    if q is None:
                                        tgts.append((rr, cc))
                                    else:
                                        jumped = True
                                else:
                                    if q is not None:
                                        if not mine(q):
                                            tgts.append((rr, cc))
                                        break
                            rr += dr
                            cc += dc
                elif t == 'N':
                    for dr, dc, lr, lc in ((2, 1, 1, 0), (2, -1, 1, 0), (-2, 1, -1, 0), (-2, -1, -1, 0),
                                           (1, 2, 0, 1), (-1, 2, 0, 1), (1, -2, 0, -1), (-1, -2, 0, -1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < 10 and 0 <= cc < 9 and b[r + lr][c + lc] is None:
                            tgts.append((rr, cc))
                elif t == 'B':
                    for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
                        rr, cc = r + dr, c + dc
                        if not (0 <= rr < 10 and 0 <= cc < 9):
                            continue
                        if side == 'w' and rr < 5:
                            continue
                        if side == 'b' and rr > 4:
                            continue
                        if b[r + dr // 2][c + dc // 2] is None:
                            tgts.append((rr, cc))
                elif t == 'A':
                    for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        rr, cc = r + dr, c + dc
                        if 3 <= cc <= 5 and ((side == 'w' and 7 <= rr <= 9) or (side == 'b' and 0 <= rr <= 2)):
                            tgts.append((rr, cc))
                elif t == 'K':
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if 3 <= cc <= 5 and ((side == 'w' and 7 <= rr <= 9) or (side == 'b' and 0 <= rr <= 2)):
                            tgts.append((rr, cc))
                elif t == 'P':
                    fwd = -1 if side == 'w' else 1
                    rr = r + fwd
                    if 0 <= rr < 10:
                        tgts.append((rr, c))
                    crossed = (r <= 4) if side == 'w' else (r >= 5)
                    if crossed:
                        for cc in (c - 1, c + 1):
                            if 0 <= cc < 9:
                                tgts.append((r, cc))
                for (rr, cc) in tgts:
                    q = b[rr][cc]
                    if q is None or not mine(q):
                        res.append(((r, c), (rr, cc)))
        return res

    def find_king(self, side):
        k = 'K' if side == 'w' else 'k'
        for r in range(10):
            for c in range(3, 6):
                if self.b[r][c] == k:
                    return (r, c)
        return None

    def in_check(self, side):
        kp = self.find_king(side)
        if kp is None:
            return True
        other = 'b' if side == 'w' else 'w'
        # 對面將
        ok = self.find_king(other)
        if ok and ok[1] == kp[1]:
            lo, hi = sorted((ok[0], kp[0]))
            if all(self.b[r][kp[1]] is None for r in range(lo + 1, hi)):
                return True
        for (_, to) in self._pseudo(other):
            if to == kp:
                return True
        return False

    def legal_moves(self):
        res = []
        for mv in self._pseudo(self.turn):
            q = self.make(mv)
            if not q.in_check(self.turn):
                res.append(mv)
        return res

    # ---------- 中文棋譜 ----------
    def move_cn(self, mv):
        (r1, c1), (r2, c2) = mv
        p = self.b[r1][c1]
        red = p.isupper()
        name = RED_NAME[p] if red else BLK_NAME[p]

        def fnum(c):
            return 9 - c if red else c + 1

        def numch(n):
            return CN_NUM[n] if red else FW_NUM[n]

        # 同一路同種棋子（前/後）
        same = [r for r in range(10) if self.b[r][c1] == p]
        prefix = None
        if len(same) >= 2:
            same.sort()  # 由上到下
            if red:
                order = same  # 紅：上面是前
            else:
                order = same[::-1]  # 黑：下面是前
            idx = order.index(r1)
            if len(order) == 2:
                prefix = '前' if idx == 0 else '後'
            elif len(order) == 3:
                prefix = '前中後'[idx]
            else:
                prefix = CN_NUM[idx + 1] if red else FW_NUM[idx + 1]
        head = (prefix + name) if prefix else (name + numch(fnum(c1)))

        fwd = (r2 < r1) if red else (r2 > r1)
        if r1 == r2:
            act, n = '平', fnum(c2)
        else:
            act = '進' if fwd else '退'
            if p.upper() in ('N', 'B', 'A'):
                n = fnum(c2)
            else:
                n = abs(r2 - r1)
        return head + act + numch(n)

    def move_iccs(self, mv):
        (r1, c1), (r2, c2) = mv
        return '%s%d%s%d' % ('abcdefghi'[c1], 9 - r1, 'abcdefghi'[c2], 9 - r2)


# ---------------- 開局分類 ----------------

def _norm_moves(cn_moves):
    """把黑方全形數字轉成半形方便比對。"""
    t = str.maketrans(FW_NUM, '0123456789')
    return [m.translate(t) for m in cn_moves]


def classify_opening(cn_moves):
    """依前十幾步中文棋譜判斷開局名稱。回傳 (大類, 細分名稱)。"""
    ms = _norm_moves(cn_moves)
    red = ms[0::2][:12]
    blk = ms[1::2][:12]
    if not red:
        return ('未知', '未知')
    r1 = red[0]

    def rhas(*xs):
        return any(m in red[:8] for m in xs)

    def bhas(*xs, n=8):
        return any(m in blk[:n] for m in xs)

    # 紅方第一步
    if r1 in ('炮二平五', '炮八平五'):
        family = '中炮'
    elif r1 in ('兵七進一', '兵三進一'):
        family = '仙人指路'
    elif r1 in ('相三進五', '相七進五'):
        family = '飛相局'
    elif r1 in ('馬二進三', '馬八進七'):
        family = '起馬局'
    elif r1 in ('炮二平六', '炮八平四'):
        family = '過宮炮'
    elif r1 in ('炮二平四', '炮八平六'):
        family = '士角炮'
    elif r1 in ('仕四進五', '仕六進五'):
        family = '上仕局'
    elif r1 in ('炮二平三', '炮八平七'):
        family = '金鉤炮'
    elif r1.startswith('炮') and '進' in r1:
        family = '巡河炮'
    else:
        family = '其他（%s）' % r1

    detail = family
    if family == '中炮':
        left = (r1 == '炮八平五')
        # 黑方應法
        same_side_c = '炮8平5' if not left else '炮2平5'
        oppo_side_c = '炮2平5' if not left else '炮8平5'
        if bhas(same_side_c, n=2):
            reply = '順炮'
        elif bhas(oppo_side_c, n=2):
            reply = '列炮'
        elif bhas('馬8進7') and bhas('馬2進3') and bhas('炮8平6', '炮2平4'):
            reply = '反宮馬'
        elif bhas('馬8進7') and bhas('炮8平9'):
            reply = '三步虎'
        elif bhas('馬2進1', '馬8進9') and bhas('馬8進7', '馬2進3'):
            reply = '單提馬'
        elif bhas('馬8進7') and bhas('馬2進3'):
            reply = '屏風馬'
        elif bhas('炮8平6', '炮2平4', n=2):
            reply = '士角炮'
        elif bhas('炮2進4', '炮8進4', n=3):
            reply = '巡河炮'
        elif bhas('卒7進1', '卒3進1', n=1):
            reply = '進卒'
        elif bhas('象3進5', '象7進5', n=1):
            reply = '飛象'
        else:
            reply = '其他'
        red_var = []
        if rhas('兵五進一'):
            red_var.append('衝中兵')
        if rhas('車二進六', '車八進六'):
            red_var.append('過河車')
        elif rhas('車二進四', '車八進四'):
            red_var.append('巡河車')
        if rhas('炮八平七', '炮二平三'):
            red_var.append('五七炮')
        elif rhas('炮八平六', '炮二平四'):
            red_var.append('五六炮')
        elif rhas('炮八進四', '炮二進四'):
            red_var.append('五八炮')
        name = '中炮' + ''.join(red_var)
        detail = name + '對' + reply
    elif family == '仙人指路':
        b1 = blk[0] if blk else ''
        if b1 in ('卒7進1', '卒3進1'):
            reply = '對兵局'
        elif b1 in ('炮2平3', '炮8平7'):
            reply = '卒底炮'
        elif b1 in ('象3進5', '象7進5'):
            reply = '飛象'
        elif b1 in ('炮8平5', '炮2平5'):
            reply = '中炮'
        elif b1 in ('馬2進3', '馬8進7'):
            reply = '起馬'
        else:
            reply = '其他'
        detail = '仙人指路對' + reply
    elif family in ('飛相局', '起馬局', '過宮炮', '士角炮', '上仕局'):
        b1 = blk[0] if blk else ''
        if b1 in ('炮8平5', '炮2平5'):
            reply = '中炮'
        elif b1 in ('卒7進1', '卒3進1'):
            reply = '進卒'
        elif b1 in ('馬8進7', '馬2進3'):
            reply = '起馬'
        elif b1 in ('象3進5', '象7進5'):
            reply = '飛象'
        elif b1 in ('炮8平4', '炮2平6'):
            reply = '過宮炮'
        elif b1 in ('炮8平6', '炮2平4'):
            reply = '士角炮'
        else:
            reply = '其他'
        detail = family + '對' + reply
    return (family, detail)


if __name__ == '__main__':
    pos = Position.from_fen(START_FEN)
    assert len(pos.legal_moves()) == 44, len(pos.legal_moves())
    seq = [((7, 7), (7, 4)), ((0, 7), (2, 6)), ((9, 7), (7, 6)), ((0, 1), (2, 2))]
    cn = []
    for mv in seq:
        cn.append(pos.move_cn(mv))
        pos = pos.make(mv)
    print(cn, classify_opening(cn))
