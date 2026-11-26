"""25/11 HẬU NGHIỆM: gộp a04-TB trên Dev228 (2 seed) + kc1 (3 seed); trung bình 2 hiệu, bootstrap khối 10 ảnh riêng từng tập."""
import numpy as np, pandas as pd
T13=[i for i in range(1,15) if i!=13]; N={'13':T13,'12':[i for i in T13 if i!=12],'8':[1,3,6,7,8,9,14,5],'5':[2,4,10,11,12]}
def iou(c): return 100*np.diag(c)/np.maximum(c.sum(0)+c.sum(1)-np.diag(c),1)
def nap(g,b,s):
    z=np.load(f'{g}{b}_b3_s{s}_15lop/dev_nham_last.npz'); a=z['image_id'].astype(int); o=np.argsort(a); c=z['conf'][o].astype(float); c[:,0]=0; return a[o],c
P={'kc1':('../ket_qua/kiem_cheo_kc1/runs/',(0,1,2)),'dev':('../ket_qua/dev228_b3/runs/',(0,1))}; D={};I={}
for p,(g,S) in P.items():
    for s in S:
        for b in('tb','tbsua_a04'): I[p],D[p,b,s]=nap(g,b,s)
print('ảnh chung:',len(set(I['kc1'])&set(I['dev'])))
def hi(p,w):
    return np.mean([iou(np.tensordot(w,D[p,'tbsua_a04',s],1))-iou(np.tensordot(w,D[p,'tb',s],1)) for s in P[p][1]],axis=0)
rng=np.random.default_rng(0); X={p:[] for p in P}; Bo={}
def boot(dv):
    for p in P:
        n=len(I[p]); kh=(np.diff(I[p],prepend=I[p][0])>20).cumsum(); bl=[]
        for k in np.unique(kh):
            c=np.where(kh==k)[0]; bl+=[c[i:i+10] for i in range(0,len(c),10)] if dv=='khoi' else [c]
        Bo[p]=np.array([hi(p,np.bincount(np.concatenate([bl[j] for j in rng.integers(0,len(bl),len(bl))]),minlength=n).astype(float)) for _ in range(1000)]); print(dv,p,'đơn vị',len(bl))
    for n,C in N.items():
        pt=[hi(p,np.ones(len(I[p])))[C].mean() for p in P]; z=(Bo['kc1'][:,C].mean(1)+Bo['dev'][:,C].mean(1))/2
        print(dv,n,'kc1 %+.2f dev %+.2f gộp %+.2f CI'%(pt[0],pt[1],np.mean(pt)),np.percentile(z,[2.5,97.5]).round(2))
boot('khoi'); boot('cum')
for ten,X0 in(('bỏ kc1 1261-1280',('kc1',1261,1280)),('bỏ dev 1383-1419',('dev',1383,1419))):
    q,a,b=X0; r=[hi(p,(~((I[p]>=a)&(I[p]<=b)) if p==q else np.ones(len(I[p]))).astype(float)) for p in P]
    print(ten,{n:round(np.mean([x[C].mean() for x in r]),2) for n,C in N.items()})
w=[hi(p,(~(((I[p]>=1261)&(I[p]<=1280)) if p=='kc1' else ((I[p]>=1383)&(I[p]<=1419)))).astype(float)) for p in P]
print('bỏ cả hai',{n:round(np.mean([x[C].mean() for x in w]),2) for n,C in N.items()})
