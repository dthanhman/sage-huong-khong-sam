"""26/11: a04_cvis 2 seed Dev228 B3. Luật: 13 lớp TB 2 seed > 60,69 VÀ cả 2 seed > TB."""
import numpy as np
_TL = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()   # thư mục code/
G=f'{_TL}/../ket_qua/dev228_b3/runs/'
T=["0","road","sidewalk","building","light","sign","veg","sky","person","car","truck","bus","moto","bicycle","pole"]
C8=[1,3,6,7,8,9,14,5]; C5=[2,4,10,11,12]; S=(0,1)
def nap(b,s):
    z=np.load(f'{G}{b}_b3_s{s}_15lop/dev_nham_last.npz'); a=z['image_id'].astype(int); o=np.argsort(a); return a[o],z['conf'][o].astype(float)
def iou(c):
    c=c.copy(); c[0]=0; u=c.sum(0)+c.sum(1)-np.diag(c)
    return np.where(c.sum(1)>0,100*np.diag(c)/np.maximum(u,1),np.nan)
F={'13':lambda v:np.nanmean(v[1:]),'8':lambda v:v[C8].mean(),'5':lambda v:v[C5].mean()}
BT=['a04_cvis','tbsua_a04','tbsua_992','tb','nosam_992','sage','vis']; A='a04_cvis'
D={(b,s):nap(b,s) for b in BT for s in S}; ids=D['tb',0][0]; assert all((v[0]==ids).all() for v in D.values()) and len(ids)==228
V={k:iou(v[1].sum(0)) for k,v in D.items()}
print('kiểm TB s0 13 phải 60,29:',round(F['13'](V['tb',0]),2),'TB s1 phải 60,08:',round(F['13'](V['tb',1]),2))
for n,f in F.items():
    print(f'--- {n} lớp')
    for b in BT:
        x=[f(V[b,s]) for s in S]; print(f'{b:10s}',' / '.join(f'{v:.2f}' for v in x),f'TB {np.mean(x):.2f}')
k8=np.mean([F['8'](V[A,s]) for s in S]); h5=np.mean([F['5'](V[A,s])-F['5'](V['tb',s]) for s in S])
x13=[F['13'](V[A,s]) for s in S]; t13=[F['13'](V['tb',s]) for s in S]; print('LUẬT: 13 lớp TB',round(np.mean(x13),3),'>60,69:',np.mean(x13)>60.69,'| từng seed > TB:',[round(a-b,2) for a,b in zip(x13,t13)],all(a>b for a,b in zip(x13,t13)))
print('lớp       ',' '.join(BT),'| cvis-tb s0/s1 | cvis-sage s0/s1')
for i in range(1,15):
    if i==13: continue
    g=lambda b:' / '.join(f"{V[A,s][i]-V[b,s][i]:+.1f}" for s in S)
    print(f'{T[i]:9s}',' '.join(f'{np.mean([V[b,s][i] for s in S]):6.1f}' for b in BT),' |',g('tb'),' |',g('sage'))
N=len(ids); bl=[np.arange(i,min(i+10,N)) for i in range(0,N,10)]; rng=np.random.default_rng(0)
for b in ('sage','tb','tbsua_a04'):
    for n,f in F.items():
        d=[f(V[A,s])-f(V[b,s]) for s in S]; X=[]
        for _ in range(500):
            ii=np.concatenate([bl[j] for j in rng.integers(0,len(bl),len(bl))])
            X.append(np.mean([f(iou(D[A,s][1][ii].sum(0)))-f(iou(D[b,s][1][ii].sum(0))) for s in S]))
        print(f"cvis-{b:9s} {n:>2s} lớp: {' / '.join(f'{x:+.2f}' for x in d)} TB {np.mean(d):+.2f} CI ảnh {np.nanpercentile(X,[2.5,97.5]).round(2)}")
