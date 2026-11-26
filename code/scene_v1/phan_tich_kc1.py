"""25/11: kiểm chéo phần 1 (kc1), 411 ảnh, B3 3 seed, a04 - TB. Ma trận nhầm lẫn từng ảnh, theo luật trong notebook kc1."""
import json, numpy as np, pandas as pd
G='../ket_qua/kiem_cheo_kc1/'; S=(0,1,2); M='tb','tbsua_a04'
T=["unl","road","sidewalk","building","light","sign","veg","sky","person","car","truck","bus","moto","bicycle","pole"]
L13=[i for i in range(1,15) if i!=13]; C8=[1,3,6,7,8,9,14,5]; C5=[2,4,10,11,12]
NH={'13':L13,'12':[i for i in L13 if i!=12],'8':C8,'5':C5}
def nap(b,s):
    z=np.load(f'{G}runs/{b}_b3_s{s}_15lop/dev_nham_last.npz'); a=z['image_id'].astype(int); o=np.argsort(a); c=z['conf'][o].astype(float); c[:,0]=0; return a[o],c
def iou(c): return 100*np.diag(c)/np.maximum(c.sum(0)+c.sum(1)-np.diag(c),1)
R={(b,s):nap(b,s) for b in M for s in S}; ids=R['tb',0][0]
p=pd.read_csv('scene_v1/kiem_cheo_p1.csv',dtype=str); assert set(ids)==set(p[p.phan=='kiem'].image_id.astype(int)) and len(ids)==411
assert all((v[0]==ids).all() for v in R.values())
D={k:v[1] for k,v in R.items()}
kq=json.load(open(G+'kq_kc1.json'))
assert all(abs(iou(D[b,s].sum(0))[L13].mean()-kq[f'{b}:{s}']['l13'])<1e-6 for b,s in D); print('khớp kq_kc1.json')
kc=[json.load(open(f'{G}runs/{b}_b3_s{s}_15lop/config.json',encoding='utf-8')) for b,s in D]
print('config khác nhau ở:',[k for k in kc[0] if len({str(x.get(k)) for x in kc})>1]); print('epoch cuối:',[len(pd.read_csv(f'{G}runs/{b}_b3_s{s}_15lop/log.csv')) for b,s in D])
V={k:iou(v.sum(0)) for k,v in D.items()}
for n,C in NH.items():
    d=[V['tbsua_a04',s][C].mean()-V['tb',s][C].mean() for s in S]
    print(f"{n:>2s} lớp: TB {np.mean([V['tb',s][C].mean() for s in S]):.2f} a04 {np.mean([V['tbsua_a04',s][C].mean() for s in S]):.2f} | hiệu",' / '.join(f'{x:+.2f}' for x in d),f'TB {np.mean(d):+.2f}')
def H(w,C):  # w: trọng số ảnh
    return np.mean([(iou(np.tensordot(w,D['tbsua_a04',s],1))[C]-iou(np.tensordot(w,D['tb',s],1))[C]).mean() for s in S])
kh=(np.diff(ids,prepend=ids[0])>20).cumsum(); cum=[np.where(kh==k)[0] for k in np.unique(kh)]
print('cụm',[(int(ids[c[0]]),int(ids[c[-1]]),len(c)) for c in cum])
bl=[]
for c in cum: bl+=[c[i:i+10] for i in range(0,len(c),10)]
rng=np.random.default_rng(0); X={'khối':{n:[] for n in NH},'cụm':{n:[] for n in NH}}; XL=[]
for ten,U in (('khối',bl),('cụm',cum)):
    for _ in range(1000):
        w=np.bincount(np.concatenate([U[j] for j in rng.integers(0,len(U),len(U))]),minlength=len(ids)).astype(float)
        di=np.mean([iou(np.tensordot(w,D['tbsua_a04',s],1))-iou(np.tensordot(w,D['tb',s],1)) for s in S],axis=0)
        for n,C in NH.items(): X[ten][n].append(di[C].mean())
        if ten=='khối': XL.append(di)
    for n in NH: print(f'CI {ten} {len(U)} đơn vị, {n} lớp',np.percentile(X[ten][n],[2.5,97.5]).round(2),'P(>0)',round(np.mean(np.array(X[ten][n])>0),3))
XL=np.array(XL)
gt=D['tb',0]
print('lớp: ảnh có lớp | TB a04 | hiệu s0/s1/s2 TB | CI khối | R P (gộp seed) ')
A=sum(D['tbsua_a04',s].sum(0) for s in S); B=sum(D['tb',s].sum(0) for s in S)
for i in L13:
    e=[V['tbsua_a04',s][i]-V['tb',s][i] for s in S]
    r=lambda c:100*c[i,i]/max(c[i].sum(),1); q=lambda c:100*c[i,i]/max(c[:,i].sum(),1)
    print(f"{T[i]:9s} {(gt[:,i].sum(1)>0).sum():3d} | {np.mean([V['tb',s][i] for s in S]):5.1f} {np.mean([V['tbsua_a04',s][i] for s in S]):5.1f} | "+' / '.join(f'{x:+.1f}' for x in e)+f' TB {np.mean(e):+.2f} (đóng góp 13 lớp {np.mean(e)/13:+.2f}) | {np.percentile(XL[:,i],[2.5,97.5]).round(1)} | R {r(A)-r(B):+.1f} P {q(A)-q(B):+.1f}')
o=np.ones(len(ids))
print('bỏ từng cụm 13/12/8/5:')
for c in [[],*cum]:
    w=o.copy(); w[c]=0; print('ĐỦ' if not len(c) else f'{ids[c[0]]}-{ids[c[-1]]}',' / '.join(f'{H(w,C):+.2f}' for C in NH.values()))
print('RIÊNG từng cụm (lớp có mặt >2000px GT) 13 | 8:')
for c in cum:
    w=o*0; w[c]=1; CC=[i for i in L13 if gt[c][:,i].sum()>2000]; print(f'{ids[c[0]]}-{ids[c[-1]]}',len(CC),f'{H(w,CC):+.2f} | {H(w,[i for i in C8 if i in CC]):+.2f}')
y=pd.read_csv('scene_v1/dai_dien_dev.csv'); y=y.drop_duplicates('id').set_index('id').y.reindex(ids).values; print('thiếu độ sáng',np.isnan(y).sum())
for n,g in (('tối <98',y<97.9),('vừa',(y>=97.9)&(y<145.9)),('sáng',y>=145.9)):
    if g.sum()<5: continue
    w=g.astype(float); CC=[i for i in L13 if gt[g][:,i].sum()>2000]; print(n,g.sum(),len(CC),f'{H(w,CC):+.2f} | 8 lớp {H(w,[i for i in C8 if i in CC]):+.2f}')
for b in M: print('SD seed 13 lớp',b,round(np.std([V[b,s][L13].mean() for s in S],ddof=1),2))
