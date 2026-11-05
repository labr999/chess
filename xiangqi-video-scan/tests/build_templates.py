# 用各影片開局截圖（_grid.jpg）建內附字樣 piece_templates.npz
import sys,glob,os; sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import video_scan as V, cv2, numpy as np
from PIL import Image
def load(p): return V.norm_frame(cv2.cvtColor(np.array(Image.open(p).convert('RGB')),cv2.COLOR_RGB2BGR))
def clean(im):  # 去掉 _grid.jpg 上畫的綠點
    b,g,r=[im[...,i].astype(int) for i in range(3)]
    m=((g>150)&(r<120)&(b<120)&(g-r>60))
    m=cv2.dilate(m.astype(np.uint8)*255,np.ones((5,5),np.uint8))
    return cv2.inpaint(im,m,3,cv2.INPAINT_TELEA)
out=sys.argv[1]; rds=[]
for f in sys.argv[2:]:
    im=clean(load(f)); g=V.find_start_grid(im)
    if g is None: print('skip',f); continue
    rd=V.PieceReader(); rd.learn(im,g,V.START,0,g.observe(im))
    if len(rd.sum)>=14: rds.append(rd); print('ok',os.path.basename(f))
V.save_template_sets(out,rds); print(len(V.load_template_sets(out)),'sets')
