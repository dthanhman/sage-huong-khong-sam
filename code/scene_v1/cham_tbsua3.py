"""24/11: tbsua_992 đủ 3 seed (Dev228, B3) + vhn3 s0. Tính từ ma trận nhầm lẫn, như cham_tbsua.py."""
import numpy as np
_TL = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()   # thư mục code/
G=f'{_TL}/../ket_qua/dev228_b3/runs/'
T=["0","road","sidewalk","building","light","sign","veg","sky","person","car","truck","bus","moto","bicycle","pole"]
C8=[1,3,6,7,8,9,14,5]; C5=[2,4,10,11,12]; S=(0,1,2)
def nap(b,s):
    z=np.load(f'{G}{b}_b3_s{s}_15lop/dev_nham_last.npz'); a=z['image_id'].astype(int); o=np.argsort(a); return a[o],z['conf'][o].astype(float)
def iou(c):
    c=c.copy(); c[0]=0; u=c.sum(0)+c.sum(1)-np.diag(c)
    return np.where(c.sum(1)>0,100*np.diag(c)/np.maximum(u,1),np.nan)
F={'13':lambda v:np.nanmean(v[1:]),'8':lambda v:v[C8].mean(),'5':lambda v:v[C5].mean()}
BT=['tbsua_992','tb','nosam_992','sage','vis']
D={(b,s):nap(b,s) for b in BT for s in S}; ids=D['tb',0][0]; assert all((v[0]==ids).all() for v in D.values()) and len(ids)==228
V={k:iou(v[1].sum(0)) for k,v in D.items()}
print('kiểm TB s0 13 lớp phải 60,29:',round(F['13'](V['tb',0]),2))
for n,f in F.items():
    print(f'--- {n} lớp')
    for b in BT:
        x=[f(V[b,s]) for s in S]; print(f'{b:10s}',' / '.join(f'{v:.2f}' for v in x),f'TB {np.mean(x):.2f}')
v3=iou(nap('tbsua_vhn3',0)[1].sum(0)); print('\nvhn3 s0:',{n:round(f(v3),2) for n,f in F.items()},'| tbsua s0:',{n:round(f(V['tbsua_992',0]),2) for n,f in F.items()})
print('\nIoU lớp TB 3 seed |',' '.join(BT),'| tbsua-tb s0/s1/s2 | vhn3-tbsua s0')
for i in range(1,15):
    if i==13: continue
    print(f'{T[i]:9s}',' '.join(f'{np.mean([V[b,s][i] for s in S]):6.1f}' for b in BT),' |',' / '.join(f"{V['tbsua_992',s][i]-V['tb',s][i]:+.1f}" for s in S),f" | {v3[i]-V['tbsua_992',0][i]:+.1f}")
N=len(ids); kh=np.concatenate([[0],(np.diff(ids)>5).cumsum()]); qk=np.unique(kh)
bl=[np.arange(i,min(i+10,N)) for i in range(0,N,10)]           # khối 10 ảnh như QUY_TAC_CHAM
def hieu(ii,b,f,ss): return np.mean([f(iou(D['tbsua_992',s][1][ii].sum(0)))-f(iou(D[b,s][1][ii].sum(0))) for s in ss])
rng=np.random.default_rng(0); B=2000
for b in ('sage','tb','nosam_992'):
    for n,f in F.items():
        d=[f(V['tbsua_992',s])-f(V[b,s]) for s in S]; X=[];Y=[];Z=[]
        for _ in range(B):
            ii=np.concatenate([bl[j] for j in rng.integers(0,len(bl),len(bl))])
            e=np.array([hieu(ii,b,f,(s,)) for s in S]); X.append(e.mean()); Y.append(e[rng.integers(0,3,3)].mean())
        if n!='5':
          for _ in range(500):
            jj=np.concatenate([np.flatnonzero(kh==j) for j in rng.choice(qk,len(qk))]); Z.append(hieu(jj,b,f,S))
        p=lambda x:np.nanpercentile(x,[2.5,97.5]).round(2) if len(x) else '-'
        a=p(X); m='HƠN CHẮC' if p(Y)[0]>0 else ('HƠN NHẤT QUÁN' if min(d)>0 and a[0]>0 else 'chưa phân biệt')
        print(f"tbsua-{b:9s} {n:>2s} lớp: {' / '.join(f'{x:+.2f}' for x in d)} TB {np.mean(d):+.2f} | CI ảnh {a} | ảnh+seed {p(Y)} | khối cảnh {p(Z)} -> {m}",flush=True)
allm=np.arange(N)
for b in ('sage','tb'):
    for n in ('13','8'):
        print('bỏ từng khối cảnh, tbsua -',b,n,':',' '.join(f"{ids[kh==q].min()}-{ids[kh==q].max()}:{hieu(allm[kh!=q],b,F[n],S):+.2f}" for q in qk))
