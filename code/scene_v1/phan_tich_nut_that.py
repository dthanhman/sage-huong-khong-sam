# Phân tích: vì sao loss trên Student (A2, A4...) không tăng mIoU, hai_nhanh thắng ở đâu.
_TL = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()   # thư mục code/
import sys, numpy as np, cv2, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
TL = Path(_TL); sys.path.insert(0, str(TL/'b3'))
from chung import duong as duong_dan_anh  # noqa
R = TL/'scene_v1/runs_b0_gop'; ANH = Path(__import__('os').environ.get('SAGE_ANH', f'{_TL}/b3/anh'))
man = pd.read_csv(TL/'scene_v1/kham_pha/dev.csv', dtype=str)
LOP10 = [0,1,2,4,5,6,7,8,9,13]  # road sidewalk building sign veg sky person car truck pole (chỉ số 0..13)

def miou_anh(b, seeds):
    out = []
    for s in seeds:
        m = np.load(R/f'{b}_b0_s{s}'/'dev_nham_last.npz'); c = m['conf'].astype(float)  # (N,14,14)
        tp = np.einsum('nii->ni', c); gt = c.sum(2); pr = c.sum(1)
        iou = np.where(gt > 0, tp/np.maximum(gt+pr-tp, 1), np.nan)[:, LOP10]
        out.append(np.nanmean(iou, 1)*100); ids = m['image_id']
    return np.mean(out, 0), ids

BT = {'hai_nhanh':[0,1,2], 'sage':[0,1,2], 'tb':[0,1,2], 'nosam_kp841':[0,1,2], 'a2l10':[0,1,2]}
M = {}
for b, s in BT.items(): M[b], ids = miou_anh(b, s)
ids = [str(i).zfill(5) for i in ids]

def grad(x):
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3); return gx, gy
rows = []
for i in ids:
    r = man[man.image_id == i].iloc[0]
    vis = cv2.imread(str(duong_dan_anh(r.vis_path))); y = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    ir = cv2.imread(str(duong_dan_anh(r.ir_path)), 0).astype(np.float32)
    vx, vy = grad(y); ix, iy = grad(ir); mv = np.hypot(vx, vy); mi = np.hypot(ix, iy)
    tv, ti = np.percentile(mv, 80), np.percentile(mi, 80)
    manh_ca2 = (mv > tv) & (mi > ti)
    xung = manh_ca2 & ((vx*ix + vy*iy) < 0)          # cạnh mạnh ở cả 2 nhưng NGƯỢC chiều sáng/tối
    rieng_ir = (mi > ti) & (mv < np.percentile(mv, 50))  # cấu trúc chỉ có ở IR
    d = dict(id=i, Y_vis=y.mean(), ir_std=ir.std(), ti_le_xung=xung.mean()/max(manh_ca2.mean(),1e-6),
             r_grad=np.corrcoef(mv.ravel(), mi.ravel())[0,1])
    for b in ['sage', 'tb', 'nosam_kp841', 'nosam_kp841_vung10']:
        f = cv2.imread(str(ANH/b/'train'/f'{i}.png')); fy = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
        fx, fyy = grad(fy); mf = np.hypot(fx, fyy)
        # phần cạnh IR-riêng còn giữ trong ảnh ghép (tỉ lệ độ lớn cạnh)
        d[f'giu_ir_{b}'] = mf[rieng_ir].mean()/max(mi[rieng_ir].mean(), 1e-6)
        # ở điểm xung đột: ảnh ghép theo chiều VIS hay IR
        s_v = np.sign(fx*vx + fyy*vy)[xung]; d[f'theo_vis_xung_{b}'] = (s_v > 0).mean() if xung.any() else np.nan
    rows.append(d)
F = pd.DataFrame(rows)
for b in BT: F[b] = M[b]
F['g_hn_sage'] = F.hai_nhanh - F.sage; F['g_hn_tb'] = F.hai_nhanh - F.tb; F['g_a2_nosam'] = F.a2l10 - F.nosam_kp841
print('Spearman (151 ảnh, mIoU 10 lớp/ảnh, TB 3 seed):')
for g in ['g_hn_sage', 'g_hn_tb', 'g_a2_nosam']:
    print(' ', g, ' '.join(f'{c}={spearmanr(F[g], F[c])[0]:+.2f}(p{spearmanr(F[g], F[c])[1]:.3f})' for c in ['Y_vis','ir_std','ti_le_xung','r_grad']))
q = pd.qcut(F.ti_le_xung, 3, labels=['xung_it', 'xung_vua', 'xung_nhieu'])
print(F.groupby(q, observed=True)[['g_hn_sage','g_hn_tb','g_a2_nosam','Y_vis']].mean().round(2))
print('\nGiữ cạnh riêng IR (độ lớn cạnh ghép / IR tại chỗ chỉ IR có cạnh), trung vị:')
print({b: round(F[f'giu_ir_{b}'].median(), 3) for b in ['sage','tb','nosam_kp841','nosam_kp841_vung10']})
print('Tại điểm xung đột chiều cạnh: % điểm ảnh ghép theo chiều VIS, trung vị:')
print({b: round(F[f'theo_vis_xung_{b}'].median(), 3) for b in ['sage','tb','nosam_kp841','nosam_kp841_vung10']})
print('Tỉ lệ xung đột trong cạnh mạnh-cả-hai, trung vị:', round(F.ti_le_xung.median(), 3))
F.to_csv(TL/'scene_v1/phan_tich_nosam/nut_that_1_kenh.csv', index=False)
