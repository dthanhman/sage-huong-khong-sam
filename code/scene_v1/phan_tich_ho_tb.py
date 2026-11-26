"""25/11: họ TB + phần sửa (tbsua 3 seed, a04 2 seed) thua/thắng TB Ở ĐÂU. Ma trận nhầm lẫn từng ảnh Dev228 (chẩn đoán)."""
import numpy as np, pandas as pd
_TL = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()   # thư mục code/
G=f'{_TL}/../ket_qua/dev228_b3/runs/'
T=["unl","road","sidewalk","building","light","sign","veg","sky","person","car","truck","bus","moto","bicycle","pole"]
L13=[1,2,3,4,5,6,7,8,9,10,11,12,14]; C8=[1,3,6,7,8,9,14,5]; C5=[2,4,10,11,12]
def nap(b,s):
    z=np.load(f'{G}{b}_b3_s{s}_15lop/dev_nham_last.npz'); a=z['image_id'].astype(int); o=np.argsort(a); c=z['conf'][o].astype(float); c[:,0]=0; return a[o],c
def iou(c): return 100*np.diag(c)/np.maximum(c.sum(0)+c.sum(1)-np.diag(c),1)
R={'tbsua_992':(0,1,2),'tbsua_a04':(0,1)}; ids=nap('tb',0)[0]
D={(b,s):nap(b,s)[1] for b in (*R,'tb','sage','nosam_992') for s in (0,1,2) if not(b=='tbsua_a04' and s==2)}
kh=(np.diff(ids,prepend=ids[0])>5).cumsum(); d=pd.read_csv(f'{_TL}/scene_v1/dai_dien_dev.csv'); y=d[d.tap=='dev'].set_index('id').y.loc[ids].values
for b,S in R.items():
    X=sum(D[b,s].sum(0) for s in S); Y=sum(D['tb',s].sum(0) for s in S)
    print(f'=== {b} - tb, gộp {len(S)} seed: IoU, recall, precision và nguồn thay đổi lớn nhất (triệu px/seed, +: nhiều hơn)')
    rx,px=100*np.diag(X)/X.sum(1).clip(1),100*np.diag(X)/X.sum(0).clip(1); ry,py=100*np.diag(Y)/Y.sum(1).clip(1),100*np.diag(Y)/Y.sum(0).clip(1)
    for i in L13:
        e=[f"{iou(D[b,s].sum(0))[i]-iou(D['tb',s].sum(0))[i]:+.1f}" for s in S]
        fn=(X-Y)[i].copy(); fn[i]=0; fp=(X-Y)[:,i].copy(); fp[i]=0; k=1000*len(S)
        j=np.argsort(-abs(fn))[:2]; m=np.argsort(-abs(fp))[:2]
        print(f"{T[i]:9s} {' / '.join(e):>22s} | R {rx[i]-ry[i]:+5.1f} P {px[i]-py[i]:+5.1f} | mất sang "+', '.join(f'{T[q]} {fn[q]/k:+.0f}k' for q in j)+' | nhận nhầm từ '+', '.join(f'{T[q]} {fp[q]/k:+.0f}k' for q in m))
    print('-- bỏ từng cụm: hiệu 13 lớp / 12 lớp (bỏ moto) / 8 lớp; dòng đầu là đủ')
    for q in [-1,*np.unique(kh)]:
        g=kh!=q; f=lambda b_,s:iou(D[b_,s][g].sum(0)); z=[(f(b,s),f('tb',s)) for s in S]
        h=lambda C:np.mean([(u[C]-v[C]).mean() for u,v in z])
        print(f"{'ĐỦ' if q<0 else f'{ids[kh==q].min()}-{ids[kh==q].max()}':10s} {h(L13):+.2f} / {h([c for c in L13 if c!=12]):+.2f} / {h(C8):+.2f}")
    print('-- theo độ sáng (IoU gộp lớp có mặt >2000px, hiệu TB seed)')
    for n,g in (('tối <98',y<97.9),('vừa',(y>=97.9)&(y<145.9)),('sáng',y>=145.9)):
        C=[c for c in L13 if D['tb',0][g][:,c].sum()>2000]
        print(f"{n:8s} {g.sum():3d} ảnh, {len(C)} lớp: {np.mean([(iou(D[b,s][g].sum(0))[C]-iou(D['tb',s][g].sum(0))[C]).mean() for s in S]):+.2f} | 8 lớp {np.mean([(iou(D[b,s][g].sum(0))[[c for c in C8 if c in C]]-iou(D['tb',s][g].sum(0))[[c for c in C8 if c in C]]).mean() for s in S]):+.2f}")
print('=== độ lệch giữa seed của TB và tbsua (SD IoU lớp, 3 seed)')
print(' '.join(f"{T[i]} {np.std([iou(D['tb',s].sum(0))[i] for s in range(3)],ddof=1):.1f}/{np.std([iou(D['tbsua_992',s].sum(0))[i] for s in range(3)],ddof=1):.1f}" for i in L13))
