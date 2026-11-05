# 用上傳的截圖檢查「中局開始」的認盤（需要使用者上傳的圖片）
import sys,glob,os; sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import video_scan as V, cv2, numpy as np
from PIL import Image
U='/workspace/projects/afc5fecf-18e2-4c58-a648-481b8b0582ec/files/uploads/'
def load(p): return V.norm_frame(cv2.cvtColor(np.array(Image.open(p).convert('RGB')),cv2.COLOR_RGB2BGR))
def clean(im):
    b,g,r=[im[...,i].astype(int) for i in range(3)]
    m=((g>150)&(r<120)&(b<120)&(g-r>60))|((r>200)&(g<60)&(b<60))
    m=cv2.dilate(m.astype(np.uint8)*255,np.ones((5,5),np.uint8))
    return cv2.inpaint(im,m,3,cv2.INPAINT_TELEA)
readers=[]
for name in sys.argv[2:]:
    im=clean(load(glob.glob(U+name+'_grid-*')[0])); g=V.find_start_grid(im)
    if g is None: print('no start',name); continue
    rd=V.PieceReader(); rd.learn(im,g,V.START,0,g.observe(im)); readers.append(rd); print('lib',name,len(rd.sum))
truth=V.Position.from_fen(sys.argv[1]+' w')
for f in sorted(glob.glob(U+'s1tKyECd6Mk_nostart_*')):
    im=clean(load(f)); r=V.find_mid_grid(im,readers)
    if r is None: print(os.path.basename(f),'None'); continue
    g,pos,_=r
    diff=sum(1 for a in range(10) for b in range(9) if pos.b[a][b]!=truth.b[a][b])
    print(os.path.basename(f), pos.fen().split()[0], 'flip',g.flip,'diff vs frame1 truth',diff, round(_.last_score,2))
