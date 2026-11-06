# -*- coding: utf-8 -*-
"""整盤辨識模型（棋盤四角 + 10x9 每格分類）。

模型與前處理移植自 yolo12138/Chinese_Chess_Recognition（Hugging Face Space，MIT 授權，
作者 TheOne1006，https://github.com/TheOne1006/chinese-chess-recognition）。
ONNX 檔不放進本 repo，第一次使用時從 Hugging Face 下載到 models/ 並核對 sha256。
"""
import hashlib
import os
import urllib.request

import cv2
import numpy as np

HF_REV = '4bcaf91777effa645040ee76d008d45c2ffc03f3'
HF_BASE = 'https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition/resolve/%s/onnx/' % HF_REV
FILES = {
    'pose': ('pose/4_v6-0301.onnx', 'fbd895d40ab81b41b8188f881866485678b7871775821aa50cfe74004a1dc796'),
    'layout': ('layout_recognition/nano_v3-0319.onnx',
               'da66ba9809f15127f8ae729b1755e42ee61c100c4f9979ce0ef13602ac471298'),
}
LABELS = '.xKABNRCPkabnrcp'   # 模型 16 類的順序：空位、其他（手、遮擋）、紅子、黑子
MEAN = np.array([123.675, 116.28, 103.53])
STD = np.array([58.395, 57.12, 57.375])


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ensure_models(model_dir, log=print):
    os.makedirs(model_dir, exist_ok=True)
    out = {}
    for k, (rel, sha) in FILES.items():
        p = os.path.join(model_dir, os.path.basename(rel))
        if not (os.path.exists(p) and _sha256(p) == sha):
            log('    下載整盤辨識模型 %s（第一次約 40 MB）…' % os.path.basename(rel))
            tmp = p + '.part'
            urllib.request.urlretrieve(HF_BASE + rel, tmp)
            if _sha256(tmp) != sha:
                os.remove(tmp)
                raise RuntimeError('模型檔核對失敗：%s' % rel)
            os.replace(tmp, p)
        out[k] = p
    return out


def _affine(center, scale, out_size, inv=False):
    w, h = out_size
    sw, sh = scale
    if sw > sh * w / h:
        sh = sw * h / w
    else:
        sw = sh * w / h
    src = np.zeros((3, 2), np.float32)
    dst = np.zeros((3, 2), np.float32)
    src[0] = center
    src[1] = center + np.array([-0.5 * sw, 0.0])
    dst[0] = [w * 0.5, h * 0.5]
    dst[1] = [w * 0.5 - 0.5 * w, h * 0.5]

    def third(a, b):
        d = a - b
        return b + np.r_[-d[1], d[0]]
    src[2] = third(src[0], src[1])
    dst[2] = third(dst[0], dst[1])
    return cv2.getAffineTransform(dst, src) if inv else cv2.getAffineTransform(src, dst)


class BoardModel:
    POSE_IN = (256, 256)
    CLS_IN = (280, 315)
    WARP = (450, 500)
    CROP = (400, 450)

    def __init__(self, model_dir, log=print):
        import onnxruntime as ort
        paths = ensure_models(model_dir, log)
        opt = ort.SessionOptions()
        opt.log_severity_level = 3
        self.pose = ort.InferenceSession(paths['pose'], opt, providers=['CPUExecutionProvider'])
        self.cls = ort.InferenceSession(paths['layout'], opt, providers=['CPUExecutionProvider'])

    def keypoints(self, bgr):
        """回傳 A0, A8, J0, J8 四角（A = 黑方底線，J = 紅方底線）與分數。"""
        h, w = bgr.shape[:2]
        # 與原專案相同（它把寬高對調，但已用這個設定驗證過）
        center = np.array([h / 2.0, w / 2.0])
        scale = np.array([h, w], np.float64) * 1.25
        m = _affine(center, scale, self.POSE_IN)
        img = cv2.warpAffine(bgr, m, self.POSE_IN, flags=cv2.INTER_LINEAR)
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - MEAN) / STD
        x = np.transpose(img.astype(np.float32), (2, 0, 1))[None]
        sx, sy = self.pose.run(None, {self.pose.get_inputs()[0].name: x})
        xi, yi = np.argmax(sx[0], axis=1), np.argmax(sy[0], axis=1)
        sc = np.max(sx[0], axis=1) * np.max(sy[0], axis=1)
        pts = np.stack([xi / 2.0, yi / 2.0, np.ones(len(xi))], axis=1)
        kp = pts @ _affine(center, scale, self.POSE_IN, inv=True).T
        return kp.astype(np.float32), sc

    def classify(self, bgr, kp):
        """依四角拉正棋盤，回傳 10 列字串（黑方在上）與每格信心。"""
        pad = 50
        W, H = self.WARP
        dst = np.float32([[pad, pad], [W - pad, pad], [pad, H - pad], [W - pad, H - pad]])
        M = cv2.getPerspectiveTransform(np.float32(kp), dst)
        warp = cv2.cvtColor(cv2.warpPerspective(bgr, M, (W, H)), cv2.COLOR_BGR2RGB)
        cw, ch = self.CROP
        x0, y0 = W // 2 - cw // 2, H // 2 - ch // 2
        img = cv2.resize(warp[y0:y0 + ch, x0:x0 + cw], self.CLS_IN)
        img = (img - MEAN) / STD
        x = np.transpose(img.astype(np.float32), (2, 0, 1))[None]
        out = self.cls.run(None, {self.cls.get_inputs()[0].name: x})[0][0]   # (90, 16)
        idx = np.argmax(out, axis=-1)
        conf = out[np.arange(90), idx]
        rows = [''.join(LABELS[i] for i in idx[r * 9:(r + 1) * 9]) for r in range(10)]
        return rows, conf.reshape(10, 9)

    def read(self, bgr):
        kp, ksc = self.keypoints(bgr)
        rows, conf = self.classify(bgr, kp)
        return rows, conf, kp, ksc
