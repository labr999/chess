# 整盤辨識模型 vs 現行方法（Hough + 字樣）逐格比較；需要使用者上傳的截圖（僅供雲端沙箱）
# 用法：python tests/model_compare.py
import sys, os, glob, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import video_scan as V
from PIL import Image
U='/workspace/projects/afc5fecf-18e2-4c58-a648-481b8b0582ec/files/uploads/'
TESTS=[ # (name, file prefix, video id for own grid or None)
 ('4Xaz_s1','4XazHUpl30I_stall_1_raw-2d6e','4XazHUpl30I'),
 ('4Xaz_s2','4XazHUpl30I_stall_2_raw-0d30','4XazHUpl30I'),
 ('4Xaz_s3','4XazHUpl30I_stall_3_raw-12cc','4XazHUpl30I'),
 ('BAFH_s2','BA-FHVImzXE_stall_2_raw',None),
 ('BAFH_s3','BA-FHVImzXE_stall_3_raw',None),
 ('c8z_s1','c8z_isKk0p8_stall_1_raw','c8z_isKk0p8'),
 ('c8z_s2','c8z_isKk0p8_stall_2_raw','c8z_isKk0p8'),
 ('c8z_s3','c8z_isKk0p8_stall_3_raw','c8z_isKk0p8'),
 ('s1tK_1','s1tKyECd6Mk_nostart_1',None),
 ('s1tK_3','s1tKyECd6Mk_nostart_3',None),
 ('isIN_1','isIN8QSnRzQ_nostart_1',None),
 ('isIN_5','isIN8QSnRzQ_nostart_5-9b',None),
 ('3eMi_1','3eMivM5s-FE_nostart_1',None),
 ('YzLI_s1','YzLIcvT5tYQ_stall_1','YzLIcvT5tYQ'),
]
def path(prefix): return sorted(glob.glob(U+prefix+'*'))[0]

T={
'4Xaz_s1':['.rbaka...','......R..','..c.b....','p...p...p','..p......','.........','?...P.P.P','N..CB..r.','...nA....','RcBAKN...'],
'4Xaz_s2':['.rbakab..','.........','..c...n..','p...p.R.p','..p......','......B..','P...P.P.P','Nc.C....C','...nA....','R.BAKN.r.'],
'4Xaz_s3':['.rbakab..','.........','....cRn..','p...p.p.p','..p......','.........','P.n.P.P.P','NcC.B.NrC','....A....','.RBAK....'],
'BAFH_s2':['??b.kab..','??..a....','??.....c.','??pR....p','??....p..','??????...','P.......P','....n.NR.','..r.A.r..','...AK...C'],
'BAFH_s3':['.rbakab..','...r.....','..n....c.','p.p.p...p','.c....pn.','..PN.....','P...P.P.P','....C.N.C','....A....','.RB.KAB.R'],
'c8z_s1':['..bakab..','r........','.cn....cr','pCp.p...p','.....np..','..P......','P...P.P.P','..N.C.N..','.........','R.BAKABR.'],
'c8z_s2':['..bakab..','.c.......','........r','p.p..p..p','......p..','..P.C..R.','P...P.n.P','..N...N..','.........','R.BAKAB..'],
'c8z_s3':['..bakab..','.....r.c.','.?......r','..?.c.R.p','.........','..P...pn.','P...P.P.P','..N...N..','.........','.RBAKAB..'],
's1tK_1':['rnbakabr.','.........','.c....nc.','p.p.p..Rp','......p..','.........','P.P.P.P.P','.C..C.N..','.........','RNBAKAB..'],
's1tK_3':['r.b.kabr.','....a....','.cn...nc.','p...p..Rp','..p.P.p..','.........','P.P...P.P','.C..C.N..','.........','RNBAKAB..'],
'isIN_1':['rnbakabnr','.........','.c.....c.','p.p.p.p.p','.........','..P......','P...P.P.P','.C.....C.','.........','RNBAKABNR'],
'isIN_5':['...akabn.','...n.....','..c.b..c.','...rp.p..','r.......p','.......R.','....P.P.P','NC..C.N..','.........','.RBAKAB..'],
'3eMi_1':['rnbakab.r','.........','.c....nc.','p.p.p.p.p','.........','.........','P.P.P.P.P','.C..C....','.........','RNBAKABNR'],
'YzLI_s1':['r.bakabr.','.........','.cn...nc.','p...p.p.p','..p......','......P..','P.P.P...P','..C.C.N..','.........','RNBAKABR.'],
}


def load(p): return V.norm_frame(cv2.cvtColor(np.array(Image.open(p).convert('RGB')), cv2.COLOR_RGB2BGR))
def clean(im):
    b, g, r = [im[..., i].astype(int) for i in range(3)]
    m = ((g > 150) & (r < 120) & (b < 120) & (g - r > 60)) | ((r > 200) & (g < 60) & (b < 60))
    m = cv2.dilate(m.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    return cv2.inpaint(im, m, 3, cv2.INPAINT_TELEA)

def relaxed(rd, frame, grid, obs):
    P, S = rd.PAD, rd.SZ
    tpl = {pc: (m / rd.cnt[pc])[P:P + S, P:P + S].astype(np.float32) for pc, m in rd.sum.items()}
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV); out = [['.'] * 9 for _ in range(10)]
    for vr in range(10):
        for vc in range(9):
            nr, nc = grid.to_norm(vr, vc); o = obs[nr][nc]
            if o == 'E': continue
            red = o == 'R'; best = ('x', -9)
            for ang in (0, -15, 15, 180):
                m = rd._ink(hsv, grid, vr, vc, red, ang)
                if m is None: continue
                for pc, v in tpl.items():
                    if pc.isupper() != red: continue
                    sc = float(cv2.matchTemplate(m.astype(np.float32), v, cv2.TM_CCOEFF_NORMED).max())
                    if sc > best[1]: best = (pc, sc)
            out[nr][nc] = best[0]
    return [''.join(r) for r in out]

def current(name, pre, vid, lib):
    im = clean(load(path(pre))); grid = rd = None
    if vid:
        g0 = clean(load(glob.glob(U + vid + '_grid-*')[0])); grid = V.find_start_grid(g0)
        if grid is not None:
            rd = V.PieceReader(); rd.learn(g0, grid, V.START, 0, grid.observe(g0))
    if grid is None:
        r = V.find_mid_grid(im, lib)
        if r is None: return None
        grid, _, rd = r
    obs = grid.observe(im); pos = rd.read(im, grid, obs)
    return [''.join(c or '.' for c in r) for r in pos.b] if pos else relaxed(rd, im, grid, obs)

def score(pred, truth):
    if pred is None: return None
    return sum(1 for r in range(10) for c in range(9) if truth[r][c] != '?' and
               (pred[r][c] != truth[r][c]) and not (pred[r][c] == 'x' and truth[r][c] == '.'))

if __name__ == '__main__':
    bm = V.get_board_model(); lib = V.load_template_sets(os.path.join(V.BASE_DIR, 'piece_templates.npz'))
    tm = tc = n_all = 0
    for name, pre, vid in TESTS:
        t = T[name]; n = sum(ch not in '.?' for row in t for ch in row); n_all += n
        rows = bm.read(cv2.cvtColor(np.array(Image.open(path(pre)).convert('RGB')), cv2.COLOR_RGB2BGR))[0]
        me = score(rows, t); ce = score(current(name, pre, vid, lib), t)
        tm += me; tc += n if ce is None else ce
        print('%-8s %2d 子 | 模型錯 %d | 現行錯 %s' % (name, n, me, '讀不到盤' if ce is None else ce))
    print('合計 %d 子：模型錯 %d，現行錯 %d（讀不到盤以全錯計）' % (n_all, tm, tc))
