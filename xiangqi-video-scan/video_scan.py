# -*- coding: utf-8 -*-
"""
象棋影片自動覆盤：YouTube 播放清單 → 下載 → 辨識棋步 → 開局分類 → 依勝方分先後手

用法（在 xiangqi_web_demo 資料夾裡）：
    python video_scan.py "播放清單或影片網址"
    python video_scan.py "網址" --browser edge      # 私人清單：借用 Edge 的 YouTube 登入
    python video_scan.py "網址" --cookies cookies.txt
    python video_scan.py --local 某資料夾或某個.mp4   # 不下載，直接辨識本機影片

輸出在 video_results/：
    總表.csv                         每盤一列（Excel 可直接開）
    先手勝(紅)/<開局>/xxx.pgn        中文棋譜，含老師示範的變化
    後手勝(黑)/<開局>/xxx.pgn
    未分勝負/<開局>/xxx.pgn
    _debug/                          棋盤定位截圖（確認格點有對準）

只適用電腦 2D 棋盤畫面。原理：先在影片中找到「開局完整盤面」定出 9x10 格點，
之後每 0.25 秒判斷每個交叉點是「空 / 紅子 / 黑子」，再用合法著法比對出每一步，
所以不需要認棋子上的字，被字幕或特效短暫遮住也能撐過去。
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np

from xq_core import Position, START_FEN, classify_opening

VERSION = '1.3.2'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(BASE_DIR, 'engine')
START = Position.from_fen(START_FEN)
START_GRID = START.color_grid()

SIDE_DIR = {'紅方': '先手開局', '黑方': '後手開局'}  # 依畫面下方（主播）執哪一方分
RESULT_DIR = {'w': '先手勝(紅)', 'b': '後手勝(黑)', None: '未分勝負'}


def log(*a):
    print(*a, flush=True)


# ====================================================================
# 下載
# ====================================================================

_DENO = None


def deno_path():
    """yt-dlp 下載 YouTube 需要 deno；用 pip 安裝的 deno 套件，不必另外裝。"""
    global _DENO
    if _DENO is None:
        try:
            from deno import find_deno_bin
            _DENO = find_deno_bin() or ''
        except Exception:
            _DENO = ''
    return _DENO


def _ydl_opts(args, extra=None):
    o = {'quiet': True, 'no_warnings': True, 'ignoreerrors': True}
    if deno_path():
        o['js_runtimes'] = {'deno': {'path': deno_path()}}
    if args.browser:
        o['cookiesfrombrowser'] = (args.browser,)
    if args.cookies:
        o['cookiefile'] = args.cookies
    if extra:
        o.update(extra)
    return o


def list_videos(url, args):
    import yt_dlp
    with yt_dlp.YoutubeDL(_ydl_opts(args, {'extract_flat': 'in_playlist'})) as y:
        info = y.extract_info(url, download=False)
    if not info:
        raise RuntimeError('讀不到這個網址。私人清單請加 --browser edge（或 chrome / firefox），或改成「不公開」。')
    if info.get('_type') == 'playlist' or 'entries' in info:
        out, seen = [], set()
        for e in info.get('entries') or []:
            if not e or not e.get('id') or e['id'] in seen:
                continue
            seen.add(e['id'])
            out.append({'id': e['id'], 'title': e.get('title') or e['id'],
                        'url': 'https://www.youtube.com/watch?v=' + e['id']})
        return info.get('title') or '播放清單', out
    return info.get('title'), [{'id': info['id'], 'title': info.get('title') or info['id'],
                                'url': 'https://www.youtube.com/watch?v=' + info['id']}]


def download(video, args, cache_dir):
    import yt_dlp
    hits = glob.glob(os.path.join(cache_dir, video['id'] + '.*'))
    hits = [h for h in hits if not h.endswith('.part')]
    if hits:
        return hits[0]
    opts = _ydl_opts(args, {
        # 只要畫面，不要聲音；720p 以下就夠辨識
        'format': 'bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]/b',
        'outtmpl': os.path.join(cache_dir, '%(id)s.%(ext)s'),
        'ignoreerrors': False,
    })
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([video['url']])
    hits = [h for h in glob.glob(os.path.join(cache_dir, video['id'] + '.*')) if not h.endswith('.part')]
    if not hits:
        raise RuntimeError('下載失敗')
    return hits[0]


# ====================================================================
# 棋盤定位與每格判斷
# ====================================================================

def norm_frame(f):
    h, w = f.shape[:2]
    if h != 720:
        s = 720.0 / h
        f = cv2.resize(f, (int(round(w * s)), 720), interpolation=cv2.INTER_AREA)
    return f


def imwrite_u(path, img):
    """cv2.imwrite 在 Windows 遇到中文路徑會靜靜失敗，改用 imencode + tofile。"""
    try:
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            buf.tofile(path)
            return True
    except Exception as e:
        log('    （存截圖失敗：%s）' % e)
    return False


def debug_frame(frame, t):
    """找不到開局時留一張畫面：畫出偵測到的圓，左上角寫時間與圓數。"""
    h = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cs = hough(gray, h / 60.0, h / 13.0, h / 40.0)
    img = frame.copy()
    for (x, y, r) in cs:
        cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(img, '%d:%02d  circles=%d' % (int(t) // 60, int(t) % 60, len(cs)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img


def stall_frame(frame, grid, obs, tracker, t):
    """追蹤卡住時的畫面：每格畫出程式看到的（紅點=紅子、藍點=黑子），
    黃圈標出和目前局面不同的格子。"""
    img = frame.copy()
    exp = tracker.cur.grid
    for vr in range(10):
        for vc in range(9):
            nr, nc = grid.to_norm(vr, vc)
            x, y = grid.px(vr, vc)
            o = obs[nr][nc]
            if o == 'R':
                cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)
            elif o == 'B':
                cv2.circle(img, (int(x), int(y)), 6, (255, 0, 0), -1)
            if o != exp[nr][nc]:
                cv2.circle(img, (int(x), int(y)), int(grid.s * 0.45), (0, 255, 255), 2)
    cv2.putText(img, '%d:%02d stuck  diff=%d  moves=%d' % (int(t) // 60, int(t) % 60, mismatch(exp, obs),
                tracker.nodes - 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img


def hough(gray, rmin, rmax, mindist, p2=0.8):
    g = cv2.GaussianBlur(gray, (5, 5), 1.2)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=mindist, param1=120, param2=p2,
                         minRadius=int(rmin), maxRadius=int(rmax))
    if c is None:
        return []
    return [tuple(map(float, x)) for x in c[0]]


def color_frac(hsv, x, y, r):
    """棋子上的字：紅字像素 /（紅字+深色字）。紅子接近 1，黑子接近 0。"""
    r = max(4, int(r * 0.62))
    x, y = int(round(x)), int(round(y))
    h, w = hsv.shape[:2]
    x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.5
    yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
    m = (xx * xx + yy * yy) <= r * r
    H, S, V = patch[..., 0][m], patch[..., 1][m], patch[..., 2][m]
    red = np.count_nonzero(((H <= 10) | (H >= 165)) & (S >= 90) & (V >= 60))
    dark = np.count_nonzero((V <= 95) & ~(((H <= 10) | (H >= 165)) & (S >= 90)))
    return red / (red + dark + 1.0)


def color_lab(lab, x, y, r):
    """字的顏色偏紅程度：字（較暗像素）的 Lab a* 減去棋子底色的 a*。
    紅字明顯為正，黑字接近 0；不靠飽和度，暗紅字、褪色畫面也分得開。"""
    r = max(4, int(r * 0.62))
    x, y = int(round(x)), int(round(y))
    h, w = lab.shape[:2]
    x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    patch = lab[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
    m = (xx * xx + yy * yy) <= r * r
    L, A = patch[..., 0][m].astype(np.float32), patch[..., 1][m].astype(np.float32)
    med = float(np.median(L))
    ink = L < med - 25
    if ink.sum() < 5:
        ink = L < np.percentile(L, 20)
    if ink.all() or not ink.any():
        return 0.0
    return float(A[ink].mean() - A[~ink].mean())


def color_feat(frame_hsv, frame_lab, x, y, r, mode):
    if mode == 'lab':
        return color_lab(frame_lab, x, y, r)
    return color_frac(frame_hsv, x, y, r)


def pick_color_mode(hsv, lab, top, bot):
    """開局時上下兩半各 16 子顏色已知：選分得最開的特徵，回傳 (mode, flip, thr) 或 None。"""
    best = None
    for mode in ('frac', 'lab'):
        ft = [color_feat(hsv, lab, *c, mode) for c in top]
        fb = [color_feat(hsv, lab, *c, mode) for c in bot]
        mt, mb = float(np.median(ft)), float(np.median(fb))
        flip = mt > mb
        red, blk = (ft, fb) if flip else (fb, ft)
        sd = (np.std(red) + np.std(blk)) / 2.0 + 1e-6
        sep = abs(mt - mb) / sd
        lo, hi = float(np.percentile(red, 10)), float(np.percentile(blk, 90))
        thr = (lo + hi) / 2.0 if lo > hi else (mt + mb) / 2.0
        mr, mbk = float(np.median(red)), float(np.median(blk))
        if best is None or sep > best[0]:
            best = (sep, mode, flip, thr, mr, mbk)
    if best[0] < 1.5:
        return None
    return best[1:]


class Grid:
    """影片畫面上的 9x10 格點。vr/vc 是畫面座標，flip=True 表示紅方在畫面上方。"""

    def __init__(self, x0, y0, sx, sy, flip, thr, mode='frac', mr=None, mb=None):
        self.x0, self.y0, self.sx, self.sy, self.flip, self.thr = x0, y0, sx, sy, flip, thr
        self.mode, self.mr, self.mb = mode, mr, mb
        self.s = (sx + sy) / 2.0

    def px(self, vr, vc):
        return self.x0 + vc * self.sx, self.y0 + vr * self.sy

    def to_norm(self, vr, vc):
        return (9 - vr, 8 - vc) if self.flip else (vr, vc)

    def to_json(self):
        return dict(x0=self.x0, y0=self.y0, sx=self.sx, sy=self.sy, flip=self.flip, thr=self.thr, mode=self.mode)

    def observe(self, frame):
        """回傳 10x9（標準方向，紅方在下）的 'E'/'R'/'B'。"""
        s = self.s
        x_lo, x_hi = int(max(0, self.x0 - s)), int(min(frame.shape[1], self.x0 + 8 * self.sx + s))
        y_lo, y_hi = int(max(0, self.y0 - s)), int(min(frame.shape[0], self.y0 + 9 * self.sy + s))
        roi = frame[y_lo:y_hi, x_lo:x_hi]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB) if self.mode == 'lab' else None
        circles = hough(gray, s * 0.30, s * 0.58, s * 0.62)
        obs = [['E'] * 9 for _ in range(10)]
        best = {}
        for (x, y, r) in circles:
            gx, gy = x + x_lo, y + y_lo
            vc = int(round((gx - self.x0) / self.sx))
            vr = int(round((gy - self.y0) / self.sy))
            if not (0 <= vr < 10 and 0 <= vc < 9):
                continue
            cx, cy = self.px(vr, vc)
            d = ((gx - cx) ** 2 + (gy - cy) ** 2) ** 0.5
            if d > 0.33 * s:
                continue
            if (vr, vc) not in best or d < best[(vr, vc)][0]:
                best[(vr, vc)] = (d, x, y, r)
        feats = {k: color_feat(hsv, lab, x, y, r, self.mode) for k, (_, x, y, r) in best.items()}
        thr = self.thr
        if self.mr is not None and len(feats) >= 8:
            # 畫面亮度、壓縮會讓顏色整體偏移：以開局學到的紅/黑中心做兩群分類，跟著這張畫面調整門檻
            mr, mb = self.mr, self.mb
            vals = np.array(list(feats.values()))
            for _ in range(5):
                t2 = (mr + mb) / 2.0
                rv, bv = vals[vals >= t2], vals[vals < t2]
                if len(rv) == 0 or len(bv) == 0:
                    break
                mr, mb = float(np.median(rv)), float(np.median(bv))
            if mr - mb >= 0.5 * (self.mr - self.mb):
                thr = (mr + mb) / 2.0
        for (vr, vc), f in feats.items():
            nr, nc = self.to_norm(vr, vc)
            obs[nr][nc] = 'R' if f >= thr else 'B'
        return obs


_A_SQ = {(9, 3), (9, 5), (8, 4), (7, 3), (7, 5)}
_B_SQ = {(9, 2), (9, 6), (7, 0), (7, 4), (7, 8), (5, 2), (5, 6)}


def square_ok(pc, r, c):
    """棋子能不能出現在這格（標準化座標：紅方在下，第 9 列是紅方底線）。"""
    u = pc.upper()
    if pc.islower():
        r = 9 - r  # 黑子翻到紅方視角檢查
    if u == 'K':
        return 7 <= r <= 9 and 3 <= c <= 5
    if u == 'A':
        return (r, c) in _A_SQ
    if u == 'B':
        return (r, c) in _B_SQ
    if u == 'P':
        if r >= 7:
            return False
        if r >= 5:
            return c % 2 == 0
        return True
    return True


class PieceReader:
    """從影片本身學每種棋子的字樣（局面已確定時記下每格的字），
    之後用來辨認老師用手擺出來的局面。"""

    SZ = 24
    LIMIT = {'K': 1, 'A': 2, 'B': 2, 'N': 2, 'R': 2, 'C': 2, 'P': 5}

    def __init__(self):
        self.sum = {}
        self.cnt = {}
        self.frames = 0
        self.last_t = -999.0

    def _ink(self, hsv, grid, vr, vc, red, ang=0):
        x, y = grid.px(vr, vc)
        r = int(grid.s * 0.36)
        x, y = int(round(x)), int(round(y))
        h, w = hsv.shape[:2]
        if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
            return None
        patch = hsv[y - r:y + r + 1, x - r:x + r + 1]
        V = patch[..., 2].astype(np.float32)
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx * xx + yy * yy) <= (r * r)
        med = float(np.median(V[disk]))
        m = ((V < med - 35) & disk).astype(np.float32)
        if ang:
            M = cv2.getRotationMatrix2D((r, r), ang, 1.0)
            m = cv2.warpAffine(m, M, (2 * r + 1, 2 * r + 1))
        m = cv2.resize(m, (self.SZ, self.SZ), interpolation=cv2.INTER_AREA)
        m = cv2.GaussianBlur(m, (3, 3), 0)
        m = m - m.mean()
        n = float(np.linalg.norm(m))
        return m / n if n > 1e-6 else None

    def ready(self):
        return self.frames >= 1 and len(self.sum) >= 12

    def learn(self, frame, grid, pos, t, obs=None):
        if self.frames >= 8 or t - self.last_t < 2:
            return
        self.last_t = t
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for vr in range(10):
            for vc in range(9):
                nr, nc = grid.to_norm(vr, vc)
                pc = pos.b[nr][nc]
                if pc is None:
                    continue
                if obs is not None and obs[nr][nc] != ('R' if pc.isupper() else 'B'):
                    continue
                m = self._ink(hsv, grid, vr, vc, pc.isupper())
                if m is None:
                    continue
                self.sum[pc] = self.sum.get(pc, 0) + m
                self.cnt[pc] = self.cnt.get(pc, 0) + 1
        self.frames += 1

    def read(self, frame, grid, obs):
        """回傳 Position（輪走方先設紅）或 None（認不出合理局面）。"""
        if not self.ready():
            return None
        tpl = {}
        for pc, m in self.sum.items():
            v = m / self.cnt[pc]
            v = v - v.mean()
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                tpl[pc] = v / n
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b = [[None] * 9 for _ in range(10)]
        cand = []
        for vr in range(10):
            for vc in range(9):
                nr, nc = grid.to_norm(vr, vc)
                o = obs[nr][nc]
                if o == 'E':
                    continue
                red = o == 'R'
                scores = {}
                for ang in (0, -15, 15, 180):
                    m = self._ink(hsv, grid, vr, vc, red, ang)
                    if m is None:
                        continue
                    for pc, v in tpl.items():
                        if pc.isupper() != red:
                            continue
                        sc = float((m * v).sum())
                        if sc > scores.get(pc, -9):
                            scores[pc] = sc
                if not scores:
                    return None
                ranked = sorted(scores.items(), key=lambda kv: -kv[1])
                cand.append(((nr, nc), ranked))
        # 先放把握最大的，數量超過上限就改用次佳
        cand.sort(key=lambda it: -it[1][0][1])
        used = {}
        for (nr, nc), ranked in cand:
            for pc, sc in ranked:
                if used.get(pc, 0) < self.LIMIT[pc.upper()] and sc >= 0.35 and square_ok(pc, nr, nc):
                    b[nr][nc] = pc
                    used[pc] = used.get(pc, 0) + 1
                    break
            else:
                return None
        if used.get('K') != 1 or used.get('k') != 1:
            return None
        pos = Position(b, 'w')
        rk, bk = pos.find_king('w'), pos.find_king('b')
        if not (rk and bk and 7 <= rk[0] <= 9 and 3 <= rk[1] <= 5 and 0 <= bk[0] <= 2 and 3 <= bk[1] <= 5):
            return None
        if not position_sane(pos):
            return None
        return pos


def position_sane(pos, turn_known=False):
    """將帥不能照面；輪走方不能可以直接吃掉對方的將（否則局面不合法，引擎會當掉）。"""
    try:
        rc, bc = pos.in_check('w'), pos.in_check('b')
    except Exception:
        return False
    if rc and bc:
        return False  # 將帥照面或雙方同時被將
    if turn_known:
        return not (bc if pos.turn == 'w' else rc)
    return True


def _uniform_run(xs, n=9, tol=0.22):
    """在排序好的 x 座標中找出 n 個等距點，回傳 (起點, 間距) 或 None。"""
    xs = sorted(xs)
    if len(xs) < n:
        return None
    for i in range(len(xs) - n + 1):
        seg = xs[i:i + n]
        d = np.diff(seg)
        md = np.median(d)
        if md > 0 and np.all(np.abs(d - md) <= tol * md):
            return seg[0], (seg[-1] - seg[0]) / (n - 1)
    return None


def dedupe_circles(cs, d):
    """同一顆棋子常被偵測成內外兩圈；圓心距離小於 d 的只留一個（半徑取中間值）。"""
    out = []
    for c in sorted(cs, key=lambda c: -c[2]):
        for i, o in enumerate(out):
            if (c[0] - o[0]) ** 2 + (c[1] - o[1]) ** 2 < d * d:
                break
        else:
            out.append(c)
    return out


def find_start_grid(frame):
    """在畫面中找開局完整盤面；找到回傳 Grid，否則 None。"""
    h = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cs = hough(gray, h / 60.0, h / 13.0, h / 40.0)
    if len(cs) < 28:
        return None
    rmed = float(np.median([c[2] for c in cs]))
    cs = [c for c in cs if 0.7 * rmed <= c[2] <= 1.35 * rmed]
    cs = dedupe_circles(cs, 0.6 * rmed)
    if len(cs) < 28:
        return None
    # 依 y 分列
    cs.sort(key=lambda c: c[1])
    rows, cur = [], [cs[0]]
    for c in cs[1:]:
        if abs(c[1] - cur[-1][1]) <= 0.5 * rmed:
            cur.append(c)
        else:
            rows.append(cur)
            cur = [c]
    rows.append(cur)
    backs = []
    for row in rows:
        run = _uniform_run([c[0] for c in row])
        if run:
            backs.append((float(np.mean([c[1] for c in row])), run))
    if len(backs) == 1:
        # 另一條底線少了一兩子（影片開頭已走過）：找格點對得上的最遠一列
        yb, (bx0, bsx) = backs[0]
        cand = None
        for row in rows:
            if len(row) < 6:
                continue
            ym = float(np.mean([c[1] for c in row]))
            dy = abs(ym - yb)
            if not (0.8 * bsx * 9 <= dy <= 1.25 * bsx * 9):
                continue
            al = sum(1 for c in row
                     if abs((c[0] - bx0) / bsx - round((c[0] - bx0) / bsx)) < 0.25
                     and 0 <= round((c[0] - bx0) / bsx) <= 8)
            if al >= 6 and (cand is None or abs(dy - 9 * bsx) < cand[0]):
                cand = (abs(dy - 9 * bsx), ym)
        if cand is not None:
            backs.append((cand[1], (bx0, bsx)))
    if len(backs) < 2:
        return None
    backs.sort()
    (ytop, (x0a, sxa)), (ybot, (x0b, sxb)) = backs[0], backs[-1]
    if abs(x0a - x0b) > 0.3 * sxa or abs(sxa - sxb) > 0.1 * sxa:
        return None
    x0, sx = (x0a + x0b) / 2, (sxa + sxb) / 2
    sy = (ybot - ytop) / 9.0
    if not (0.8 * sx <= sy <= 1.25 * sx):
        return None
    if not (0.3 * sx <= rmed <= 0.6 * sx):
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 對照開局 32 子
    occ = {}
    for (x, y, r) in cs:
        vc = int(round((x - x0) / sx))
        vr = int(round((y - ytop) / sy))
        if 0 <= vr < 10 and 0 <= vc < 9:
            if abs(x - (x0 + vc * sx)) < 0.3 * sx and abs(y - (ytop + vr * sy)) < 0.3 * sy:
                occ[(vr, vc)] = (x, y, r)
    need = [(r, c) for r in range(10) for c in range(9) if START_GRID[r][c] != 'E']
    hit = sum(1 for k in need if k in occ)
    extra = sum(1 for k in occ if START_GRID[k[0]][k[1]] == 'E')
    if hit < 29 or extra > 3:
        return None
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    top = [occ[k] for k in occ if k[0] <= 4 and START_GRID[k[0]][k[1]] != 'E']
    bot = [occ[k] for k in occ if k[0] >= 5 and START_GRID[k[0]][k[1]] != 'E']
    pick = pick_color_mode(hsv, lab, top, bot)
    if pick is None:
        return None
    mode, flip, thr, mr, mbk = pick  # flip=True：紅方在上
    return Grid(x0, ytop, sx, sy, flip, thr, mode, mr, mbk)


def mismatch(grid_a, obs):
    n = 0
    for r in range(10):
        ra, ro = grid_a[r], obs[r]
        for c in range(9):
            if ra[c] != ro[c]:
                n += 1
    return n


# ====================================================================
# 追蹤棋步（含老師退回去示範的變化）
# ====================================================================

class Node:
    __slots__ = ('pos', 'grid', 'move', 'cn', 'parent', 'children', 't')

    def __init__(self, pos, move=None, cn=None, parent=None, t=0.0):
        self.pos, self.move, self.cn, self.parent, self.t = pos, move, cn, parent, t
        self.grid = pos.color_grid()
        self.children = []


class Tracker:
    def __init__(self, persist=2):
        self.root = Node(START)
        self.cur = self.root
        self.pending = None
        self.persist = persist
        self.takebacks = 0
        self.nodes = 1
        self.segments = []      # 老師擺出中局後重新開始的段落（各自的根節點）
        self.setups = set()     # 擺局根節點：還不知道輪誰走

    def reanchor(self, pos, t):
        root = Node(pos, t=t)
        self.segments.append(root)
        self.setups.add(id(root))
        self.cur = root
        self.pending = None
        self.nodes += 1

    def _succ(self, node):
        out = []
        turns = 'wb' if (id(node) in self.setups and not node.children) else node.pos.turn
        for tn in turns:
            p = node.pos.copy()
            p.turn = tn
            for mv in p.legal_moves():
                q = p.make(mv)
                out.append((mv, q, q.color_grid()))
        return out

    def feed(self, obs, t):
        d_cur = mismatch(self.cur.grid, obs)
        if d_cur <= 1:
            self.pending = None
            return
        best = (99, None, None)  # (d, key, action)；絕殺後沒有合法著法時 succ 為空
        succ = self._succ(self.cur)
        for mv, q, g in succ:
            d = mismatch(g, obs)
            if d < best[0]:
                best = (d, ('m', mv), ('move', [mv]))
        anc, k = self.cur.parent, 0
        while anc is not None and k < 60:
            d = mismatch(anc.grid, obs)
            if d < best[0]:
                best = (d, ('b', id(anc)), ('back', anc))
            anc, k = anc.parent, k + 1
        if best[0] > 2 and d_cur >= 4:
            # 影片跳太快：試連走兩步
            for mv, q, g in succ:
                if mismatch(g, obs) >= d_cur:
                    continue
                for mv2 in q.legal_moves():
                    q2 = q.make(mv2)
                    d = mismatch(q2.color_grid(), obs)
                    if d < best[0]:
                        best = (d, ('m2', mv, mv2), ('move', [mv, mv2]))
        if best[0] <= 2 and best[0] + 2 <= d_cur:
            if self.pending and self.pending[0] == best[1]:
                self.pending[1] += 1
            else:
                self.pending = [best[1], 1]
            if self.pending[1] >= self.persist:
                self._apply(best[2], t)
                self.pending = None
        else:
            self.pending = None

    def _apply(self, action, t):
        kind, val = action
        if kind == 'back':
            self.cur = val
            self.takebacks += 1
            return
        if id(self.cur) in self.setups and not self.cur.children:
            (r1, c1) = val[0][0]
            pc = self.cur.pos.b[r1][c1]
            if pc is not None:
                self.cur.pos.turn = 'w' if pc.isupper() else 'b'
        for mv in val:
            node = next((ch for ch in self.cur.children if ch.move == mv), None)
            if node is None:
                cn = self.cur.pos.move_cn(mv)
                node = Node(self.cur.pos.make(mv), mv, cn, self.cur, t)
                self.cur.children.append(node)
                self.nodes += 1
            self.cur = node


def main_line(node):
    line = []
    while node.children:
        node = node.children[0]
        line.append(node)
    return line


def pgn_moves(start_node, start_ply=0):
    """start_node 之後的著法（第一個孩子是主線，其餘放括號當變化）。"""
    out = []

    def num(ply, force):
        if ply % 2 == 0:
            return '%d. ' % (ply // 2 + 1)
        return ('%d... ' % (ply // 2 + 1)) if force else ''

    def walk(node, ply, force):
        while node.children:
            main = node.children[0]
            out.append(num(ply, force) + main.cn)
            for alt in node.children[1:]:
                out.append('(' + num(ply, True) + alt.cn)
                walk(alt, ply + 1, False)
                out[-1] += ')'
            force = len(node.children) > 1
            node, ply = main, ply + 1

    walk(start_node, start_ply, start_ply % 2 == 1)
    return ' '.join(out)


# ====================================================================
# Pikafish 判斷終局
# ====================================================================

class Engine:
    def __init__(self):
        pats = ['Pikafish-Windows*.exe', 'pikafish*.exe'] if sys.platform == 'win32' else ['pikafish*']
        path = None
        for p in pats:
            hits = [h for h in sorted(glob.glob(os.path.join(ENGINE_DIR, p)) +
                                      glob.glob(os.path.join(ENGINE_DIR, '**', p), recursive=True) +
                                      glob.glob(os.path.join(BASE_DIR, 'engine*', '**', p), recursive=True))
                    if not h.endswith('.nnue')]
            if hits:
                path = hits[0]
                break
        if not path:
            raise RuntimeError('%s 裡找不到 Pikafish*.exe' % ENGINE_DIR)
        self.dir = os.path.dirname(path)
        self.path = path
        self._start()

    def _start(self):
        flags = 0x08000000 if sys.platform == 'win32' else 0  # 不跳出黑視窗
        self.p = subprocess.Popen([self.path], cwd=self.dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1, creationflags=flags)
        self._send('uci')
        self._wait('uciok')
        self._send('isready')
        self._wait('readyok')

    def _send(self, s):
        self.p.stdin.write(s + '\n')
        self.p.stdin.flush()

    def _wait(self, tok):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('Pikafish 結束了')
            if line.startswith(tok):
                return line

    def eval(self, fen, depth=16):
        """回傳走棋方視角分數（centipawn），將死以 ±30000 表示。引擎當掉會自動重開再試一次。"""
        try:
            return self._eval(fen, depth)
        except (OSError, ValueError, RuntimeError):
            self._kill()
            self._start()
            return self._eval(fen, depth)

    def _kill(self):
        for f in (self.p.stdin, self.p.stdout):
            try:
                f.close()
            except Exception:
                pass
        try:
            self.p.kill()
            self.p.wait(timeout=5)
        except Exception:
            pass

    def _eval(self, fen, depth):
        self._send('position fen ' + fen)
        self._send('go depth %d' % depth)
        score = 0
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('Pikafish 結束了')
            m = re.search(r'score (cp|mate) (-?\d+)', line)
            if m and ' pv ' in line:
                v = int(m.group(2))
                score = v if m.group(1) == 'cp' else (30000 - abs(v)) * (1 if v > 0 else -1)
            if line.startswith('bestmove'):
                break
        return score

    def close(self):
        try:
            self._send('quit')
            self.p.wait(timeout=3)
        except Exception:
            pass
        self._kill()


TITLE_HINT = [(r'红胜|紅勝|红方胜|紅方勝|先手胜|先手勝', 'w'), (r'黑胜|黑勝|黑方胜|黑方勝|后手胜|後手勝', 'b')]


def judge(pos, engine, title, depth, margin):
    if not position_sane(pos, turn_known=True):
        return None, '終局局面不合法（辨識有誤）'
    if not pos.legal_moves():
        win = 'b' if pos.turn == 'w' else 'w'
        return win, '絕殺/困斃（%s無棋可走）' % ('紅方' if pos.turn == 'w' else '黑方')
    if engine is not None:
        try:
            sc = engine.eval(pos.fen(), depth)
            red = sc if pos.turn == 'w' else -sc
            if red >= margin:
                return 'w', '終局引擎評分 紅%+d' % red
            if red <= -margin:
                return 'b', '終局引擎評分 紅%+d' % red
            reason = '終局引擎評分 紅%+d（差距不足）' % red
        except Exception as e:
            reason = '引擎錯誤：%s' % e
    else:
        reason = '未啟用引擎'
    for pat, side in TITLE_HINT:
        if re.search(pat, title or ''):
            return side, reason + '；依標題'
    return None, reason


# ====================================================================
# 單支影片
# ====================================================================

def scan_video(path, step=0.25, persist=2, debug_dir=None, vid='video'):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError('打不開影片')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    every = max(1, int(round(fps * step)))
    search_every = max(1, int(round(fps * 0.5)))
    grid, tracker = None, None
    idx = 0
    t0 = time.time()
    dur = total / fps if total else 0
    snap_times = sorted(set([5, 15, 30, 60] + ([dur * k / 6 for k in range(1, 6)] if dur else [])))
    snaps = []
    stalls, last_ok, last_stall = [], 0.0, -999.0
    pend_read = None
    reader, prev_obs, stable = PieceReader(), None, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        want = (idx % (every if grid else search_every)) == 0
        if want:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame = norm_frame(frame)
            t = idx / fps
            if grid is None:
                if debug_dir and snap_times and t >= snap_times[0]:
                    while snap_times and t >= snap_times[0]:
                        snap_times.pop(0)
                    snaps.append(debug_frame(frame, t))
                grid = find_start_grid(frame)
                if grid is not None:
                    tracker = Tracker(persist)
                    last_ok = t
                    reader.learn(frame, grid, START, t, grid.observe(frame))
                    log('    找到開局盤面 @ %d:%02d（%s在上方，顏色判斷 %s）' % (int(t) // 60, int(t) % 60,
                                                       '紅方' if grid.flip else '黑方', grid.mode))
                    if debug_dir:
                        dbg = frame.copy()
                        for vr in range(10):
                            for vc in range(9):
                                x, y = grid.px(vr, vc)
                                cv2.circle(dbg, (int(x), int(y)), 4, (0, 255, 0), -1)
                        imwrite_u(os.path.join(debug_dir, vid + '_grid.jpg'), dbg)
            else:
                obs = grid.observe(frame)
                stable = stable + 1 if obs == prev_obs else 0
                prev_obs = obs
                n0 = tracker.nodes
                tracker.feed(obs, t)
                dnow = mismatch(tracker.cur.grid, obs)
                if dnow == 0 and stable >= 2:
                    reader.learn(frame, grid, tracker.cur.pos, t)
                if tracker.nodes != n0 or dnow <= 2:
                    last_ok = t
                elif t - last_ok > 6 and stable >= 6 and dnow >= 5:
                    pos = reader.read(frame, grid, obs)
                    key = pos.fen() if pos is not None else None
                    if key is not None and key != pend_read:
                        pend_read, pos = key, None  # 連續兩次認出同一局面才採用
                    if pos is not None:
                        pend_read = None
                        tracker.reanchor(pos, t)
                        last_ok = t
                        log('    老師擺出新局面 @ %d:%02d，重新認盤（%s）' % (int(t) // 60, int(t) % 60,
                                                                 pos.fen().split()[0]))
                elif debug_dir and len(stalls) < 3 and t - last_ok > 40 and t - last_stall > 60:
                    last_stall = t
                    stalls.append(t)
                    imwrite_u(os.path.join(debug_dir, '%s_stall_%d.jpg' % (vid, len(stalls))),
                              stall_frame(frame, grid, obs, tracker, t))
                    imwrite_u(os.path.join(debug_dir, '%s_stall_%d_raw.jpg' % (vid, len(stalls))), frame)
            if idx % int(fps * 60) == 0 and total:
                log('    %d%%（已辨識 %d 個局面，%.0f 秒）' % (100 * idx // total, tracker.nodes if tracker else 0,
                                                    time.time() - t0))
        idx += 1
    cap.release()
    if grid is None and debug_dir:
        for k, img in enumerate(snaps, 1):
            imwrite_u(os.path.join(debug_dir, '%s_nostart_%d.jpg' % (vid, k)), img)
        if snaps:
            log('    已存 %d 張畫面到 _debug（%s_nostart_*.jpg）' % (len(snaps), vid))
    return grid, tracker


# ====================================================================
# 輸出
# ====================================================================

def safe_name(s, n=40):
    s = re.sub(r'[\\/:*?"<>|\r\n\t#]+', '_', s or '').strip(' ._')
    return s[:n] or 'untitled'


def write_game(out_dir, video, gi, branch_root, grid, tracker, engine, args, csv_rows, setup=None):
    line = [branch_root] + main_line(branch_root)
    cn = [n.cn for n in line]
    if setup is None:
        family, detail = classify_opening(cn)
    else:
        family, detail = '擺局', '中局擺子'
    final = line[-1].pos
    win, reason = judge(final, engine, video.get('title'), args.depth, args.margin)
    res_tag = {'w': '1-0', 'b': '0-1', None: '*'}[win]
    host = '黑方' if grid.flip else '紅方'
    sub = os.path.join(out_dir, SIDE_DIR[host], RESULT_DIR[win], safe_name(detail))
    os.makedirs(sub, exist_ok=True)
    fname = '%s_%s%s.pgn' % (safe_name(video.get('title'), 30), video['id'], ('_%d' % gi) if gi else '')
    fpath = os.path.join(sub, fname)
    # 這條分支（含其中的變化）
    if setup is None:
        fake = Node(START)
        fake.children = [branch_root]
        moves = pgn_moves(fake)
    else:
        moves = pgn_moves(setup, 0 if setup.pos.turn == 'w' else 1)
    host = '黑方' if grid.flip else '紅方'
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write('[Game "Chinese Chess"]\n')
        f.write('[Event "%s"]\n' % (video.get('title') or '').replace('"', "'"))
        f.write('[Site "%s"]\n' % video.get('url', ''))
        f.write('[Result "%s"]\n' % res_tag)
        f.write('[Opening "%s"]\n' % detail)
        if setup is not None:
            f.write('[SetUp "1"]\n[FEN "%s"]\n' % setup.pos.fen())
        f.write('[Annotator "video_scan %s"]\n' % VERSION)
        f.write('[Format "Chinese"]\n')
        f.write('{勝方判定：%s；畫面下方（主播）執%s；退回示範 %d 次}\n' % (reason, host, tracker.takebacks))
        f.write(moves + ' ' + res_tag + '\n')
    csv_rows.append({
        '結果': RESULT_DIR[win], '開局大類': family, '開局': detail, '主線步數': len(cn),
        '判定依據': reason, '主播執': host, '標題': video.get('title'), '網址': video.get('url', ''),
        '前20步': ' '.join(cn[:20]), '終局FEN': final.fen(), '棋譜檔': os.path.relpath(fpath, out_dir),
        '小棋聖ID': '',
    })
    log('    → %s｜%s｜%d 步｜%s' % (RESULT_DIR[win], detail, len(cn), reason))
    if setup is not None and args.upload:
        log('    （擺局段落只存本機棋譜，不上傳小棋聖）')
    if args.upload and len(cn) >= 2 and setup is None:
        red, black = parse_players(video.get('title'))
        t0 = int(line[0].t)
        yt = str(video.get('url', '')).startswith('http')
        url = ('https://www.youtube.com/watch?v=%s' % video['id']) if yt else ''
        ud = video.get('upload_date') or ''
        cn_up = [c.translate(str.maketrans('１２３４５６７８９', '123456789')) for c in cn]
        payload = {
            'red': red, 'black': black, 'result': WIN_TEXT[win],
            'event': video.get('title') or video['id'],
            'date': ('%s-%s-%s' % (ud[:4], ud[4:6], ud[6:8])) if len(ud) == 8 else '',
            'year': int(ud[:4]) if len(ud) == 8 else None,
            'opening': detail, 'tokens': cn_up, 'moves': cn_up,
            'source': 'video-scan', 'format': 'video-scan %s' % VERSION,
            'sourceUrl': url, 'detailUrl': ('%s&t=%ds' % (url, max(0, t0 - 3))) if yt else '',
            'fen': final.fen(), 'validation': reason,
            'rawExport': json.dumps({'videoId': video['id'], 'moveTimes': [round(n.t, 1) for n in line],
                                     'game': gi, 'host': host, 'takebacks': tracker.takebacks}, ensure_ascii=False),
        }
        try:
            gid, dup = upload_game(args.upload, payload)
            csv_rows[-1]['小棋聖ID'] = gid
            log('    ↑ 已上傳小棋聖（編號 %s%s）' % (gid, '，覆蓋舊的' if dup else ''))
        except Exception as e:
            log('    ✗ 上傳小棋聖失敗：%s（棋譜檔仍在本機，可之後重跑）' % e)


XQS_DEFAULT = 'https://my-backend-api-1-e19n.onrender.com'
WIN_TEXT = {'w': '紅勝', 'b': '黑勝', None: '未知'}


def parse_players(title):
    """標題若有「甲 對/vs/先負/勝 乙」就取出雙方名字，否則用影片紅方／影片黑方。"""
    bad = set('炮砲马馬车車兵卒象相仕士帅帥将將中局开開布式变變')
    for m in re.finditer(r'([\u4e00-\u9fff]{2,3})\s*(?:先负|先負|先胜|先勝|先和|对|對|vs\.?|VS|Vs|负|負|胜|勝)\s*([\u4e00-\u9fff]{2,3})', title or ''):
        a, b = m.group(1), m.group(2)
        if not (set(a) & bad or set(b) & bad):
            return a, b
    return '影片紅方', '影片黑方'


def upload_game(base, payload, tries=3):
    """POST 到小棋聖 /api/games。Render 免費機睡著時第一次要等約 1 分鐘喚醒。"""
    import urllib.request
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(base.rstrip('/') + '/api/games', data=data,
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode('utf-8'))
                g = d.get('game') or {}
                return g.get('id'), d.get('duplicate')
        except Exception as e:
            last = e
            time.sleep(5 * (k + 1))
    raise RuntimeError(last)


CSV_FIELDS = ['結果', '開局大類', '開局', '主線步數', '判定依據', '主播執', '標題', '網址', '前20步', '終局FEN', '棋譜檔', '小棋聖ID']


def append_csv(path, rows):
    if not rows:
        return
    try:
        _write_csv(path, rows)
    except PermissionError:
        alt = path[:-4] + '_' + time.strftime('%m%d_%H%M%S') + '.csv'
        _write_csv(alt, rows)
        log('    ⚠ 總表.csv 被 Excel 開著無法寫入，這盤先寫到 %s（請關掉 Excel，之後的盤就會寫回總表）' % os.path.basename(alt))


def _write_csv(path, rows):
    new = not os.path.exists(path)
    with open(path, 'a', encoding='utf-8-sig' if new else 'utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def migrate_layout(out_dir):
    """舊版把棋譜放在 結果\\開局\\；改成 先手開局/後手開局\\結果\\開局\\，並更新總表。"""
    csv_path = os.path.join(out_dir, '總表.csv')
    if not os.path.exists(csv_path):
        return
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    moved = 0
    for r in rows:
        rel = r.get('棋譜檔') or ''
        parts = re.split(r'[\\/]', rel)
        if not parts or parts[0] not in RESULT_DIR.values():
            continue
        side = SIDE_DIR.get(r.get('主播執'), '先手開局')
        new_rel = os.path.join(side, *parts)
        src, dst = os.path.join(out_dir, *parts), os.path.join(out_dir, new_rel)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
        r['棋譜檔'] = new_rel
        moved += 1
    if not moved:
        return
    try:
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except PermissionError:
        log('⚠ 總表.csv 被 Excel 開著，棋譜已搬到新資料夾但總表路徑沒更新')
    for d in RESULT_DIR.values():
        for root, dirs, files in os.walk(os.path.join(out_dir, d), topdown=False):
            try:
                os.rmdir(root)
            except OSError:
                pass
    log('已把 %d 盤舊棋譜改放到「先手開局／後手開局」資料夾' % moved)


def process(video, path, out_dir, engine, args):
    dbg = os.path.join(out_dir, '_debug')
    os.makedirs(dbg, exist_ok=True)
    grid, tracker = scan_video(path, args.step, args.persist, dbg, video['id'])
    rows = []
    if grid is None:
        log('    ✗ 找不到開局完整盤面，略過')
        return 'no_start', rows
    branches = [ch for ch in tracker.root.children if len(main_line(ch)) + 1 >= args.min_moves]
    segs = [sg for sg in tracker.segments
            if sg.children and len(main_line(sg.children[0])) + 1 >= max(args.min_moves, 8)]
    if not branches and not segs:
        log('    ✗ 辨識到的步數太少（%d 個局面），略過' % tracker.nodes)
        return 'too_short', rows
    gi = 0
    for br in branches:
        write_game(out_dir, video, gi, br, grid, tracker, engine, args, rows)
        gi += 1
    for sg in segs:
        write_game(out_dir, video, gi, sg.children[0], grid, tracker, engine, args, rows, setup=sg)
        gi += 1
    return 'ok', rows


PRESET_FILE = os.path.join(BASE_DIR, 'playlists.json')
DEFAULT_PRESETS = [
    {'name': '後手', 'url': 'https://www.youtube.com/playlist?list=PLIoeW_p5UU0jDMbfjILRHWZUP8aRnwDMY'},
    {'name': '重點棋譜', 'url': 'https://www.youtube.com/playlist?list=PLIoeW_p5UU0hP8DtL-zoGEK45Q2iveh0U'},
]


def ask_url():
    if not os.path.exists(PRESET_FILE):
        json.dump(DEFAULT_PRESETS, open(PRESET_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    presets = json.load(open(PRESET_FILE, encoding='utf-8'))
    while True:
        log('\n要處理哪個播放清單？')
        for i, p in enumerate(presets, 1):
            log('  %d. %s' % (i, p['name']))
        log('  或直接貼上播放清單／影片網址')
        a = input('請輸入編號或網址：').strip().strip('"')
        if a.isdigit() and 1 <= int(a) <= len(presets):
            return presets[int(a) - 1]['url']
        if 'feed/playlists' in a:
            log('這是「所有播放清單」總覽頁，不是單一清單。請點進某個清單，'
                '複製網址列裡有 list= 的那一串。')
            continue
        if 'list=' in a or 'watch?v=' in a or 'youtu.be/' in a or '/shorts/' in a:
            if 'list=' in a:
                lid = re.search(r'list=([\w-]+)', a).group(1)
                if not any(lid in p['url'] for p in presets):
                    nm = input('要把這個清單存起來下次直接選嗎？輸入名稱（Enter 跳過）：').strip()
                    if nm:
                        presets.append({'name': nm, 'url': 'https://www.youtube.com/playlist?list=' + lid})
                        json.dump(presets, open(PRESET_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            return a
        log('看不懂這個輸入，請再試一次。')


def main():
    ap = argparse.ArgumentParser(description='象棋影片自動覆盤 v' + VERSION)
    ap.add_argument('url', nargs='?', help='YouTube 播放清單或影片網址')
    ap.add_argument('--local', help='改用本機影片（檔案或資料夾）')
    ap.add_argument('--browser', help='借用瀏覽器登入讀私人清單：edge / chrome / firefox')
    ap.add_argument('--cookies', help='cookies.txt 路徑')
    ap.add_argument('--out', default=os.path.join(BASE_DIR, 'video_results'))
    ap.add_argument('--cache', default=os.path.join(BASE_DIR, 'video_cache'))
    ap.add_argument('--keep-video', action='store_true', help='辨識完保留影片檔（預設刪除）')
    ap.add_argument('--limit', type=int, default=0, help='最多處理幾支')
    ap.add_argument('--step', type=float, default=0.25, help='取樣間隔秒數')
    ap.add_argument('--persist', type=int, default=2, help='同一局面連續出現幾次才算數')
    ap.add_argument('--min-moves', type=int, default=6, help='少於幾步不輸出')
    ap.add_argument('--depth', type=int, default=16, help='終局引擎深度')
    ap.add_argument('--margin', type=int, default=300, help='引擎評分超過多少分判勝')
    ap.add_argument('--no-engine', action='store_true')
    ap.add_argument('--upload', nargs='?', const=XQS_DEFAULT, default=None,
                    help='辨識完上傳到象棋小棋聖（預設 %s）' % XQS_DEFAULT)
    ap.add_argument('--no-upload', action='store_true')
    ap.add_argument('--redo', action='store_true', help='已處理過的也重做')
    args = ap.parse_args()
    if not args.url and not args.local:
        args.url = ask_url()
        if not args.upload and not args.no_upload:
            a = input('辨識完要自動上傳到「象棋小棋聖」棋譜倉庫嗎？(Y/n)：').strip().lower()
            if a in ('', 'y', 'yes', '是'):
                args.upload = XQS_DEFAULT
    if args.no_upload:
        args.upload = None
    if args.upload:
        log('辨識結果會上傳到：%s（第一次可能要等約 1 分鐘喚醒伺服器）' % args.upload)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    done_path = os.path.join(args.out, '已處理.json')
    done = json.load(open(done_path, encoding='utf-8')) if os.path.exists(done_path) else {}

    engine = None
    if not args.no_engine:
        try:
            engine = Engine()
            log('Pikafish 已啟動（%s），用來判斷終局勝負' % engine.dir)
        except Exception as e:
            log('（Pikafish 未啟動：%s；勝負只看絕殺與標題）' % e)

    if args.local:
        files = [args.local] if os.path.isfile(args.local) else sorted(
            glob.glob(os.path.join(args.local, '*.mp4')) + glob.glob(os.path.join(args.local, '*.webm')) +
            glob.glob(os.path.join(args.local, '*.mkv')))
        videos = [{'id': os.path.splitext(os.path.basename(p))[0], 'title': os.path.basename(p),
                   'url': p, 'path': p} for p in files]
        log('本機影片 %d 支' % len(videos))
    else:
        try:
            title, videos = list_videos(args.url, args)
        except Exception as e:
            if args.browser or args.cookies:
                raise
            log('讀不到清單（%s）' % e)
            br = input('如果是私人清單，輸入要借用登入的瀏覽器 firefox / edge / chrome（直接 Enter 結束）：').strip().lower()
            if not br:
                return
            args.browser = br
            title, videos = list_videos(args.url, args)
        log('「%s」共 %d 支影片' % (title, len(videos)))
    if args.limit:
        videos = videos[:args.limit]

    csv_path = os.path.join(args.out, '總表.csv')
    try:
        migrate_layout(args.out)
    except Exception as e:
        log('（舊棋譜搬資料夾失敗：%s）' % e)
    stats = {}
    for i, v in enumerate(videos, 1):
        if v['id'] in done and not args.redo and done[v['id']].get('status') == 'ok':
            log('[%d/%d] 已處理過，略過：%s' % (i, len(videos), v['title']))
            continue
        log('[%d/%d] %s' % (i, len(videos), v['title']))
        try:
            path = v.get('path') or download(v, args, args.cache)
            status, rows = process(v, path, args.out, engine, args)
            append_csv(csv_path, rows)
            if not v.get('path') and not args.keep_video:
                try:
                    os.remove(path)
                except OSError:
                    pass
        except KeyboardInterrupt:
            raise
        except Exception as e:
            status = 'error: %s' % e
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            where = ('（%s 第 %d 行）' % (os.path.basename(tb[-1].filename), tb[-1].lineno)) if tb else ''
            log('    ✗ 錯誤：%s%s' % (e, where))
        done[v['id']] = {'title': v['title'], 'status': status, 'time': time.strftime('%Y-%m-%d %H:%M')}
        json.dump(done, open(done_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        stats[status.split(':')[0]] = stats.get(status.split(':')[0], 0) + 1
    if engine:
        engine.close()
    log('\n完成：%s' % '，'.join('%s %d' % kv for kv in stats.items()))
    log('結果在：%s' % args.out)


if __name__ == '__main__':
    main()
