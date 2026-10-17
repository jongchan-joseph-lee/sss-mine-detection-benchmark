"""Shadow depth per target (box-scale profile) and AUROC: objects vs seabed, MILCO vs NOMBO."""
import json, numpy as np
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score
exec(open('b2_signature.py').read().split('P={')[0])
u=np.linspace(-1.5,1.5,61); sw=(u>0.05)&(u<0.5); hw=(u>-0.5)&(u<-0.05)
rec=[]
for r in scenes:
    a=median_filter(load(r),3); W=a.shape[1]; nad,_,_=nadir_band(a)
    for c,x,y,w,h in r['labels']:
        if w*W<4: continue
        p=prof2(a,x*W,y*W,w*W,h*W,nad); rec.append(('MILCO' if c==0 else 'NOMBO',r['year'],p[sw].mean(),p[hw].mean()))
    if r['labels']:
        for _ in range(4):
            sd=rng.choice([-1,1]); cx=nad+sd*rng.uniform(0.1,0.45)*W; cy=rng.uniform(0.1,0.9)*W
            p=prof2(a,cx,cy,np.median([l[3] for l in r['labels']])*W,np.median([l[4] for l in r['labels']])*W,nad)
            rec.append(('BG',r['year'],p[sw].mean(),p[hw].mean()))
cls=np.array([q[0] for q in rec]); yr=np.array([q[1] for q in rec]); sh=np.array([q[2] for q in rec]); hi=np.array([q[3] for q in rec])
def auc(pos,neg,score):  # higher score = positive
    y=np.r_[np.ones(pos.sum()),np.zeros(neg.sum())]; s=np.r_[score[pos],score[neg]]; return float(roc_auc_score(y,s))
obj=(cls!='BG'); bg=cls=='BG'; mil=cls=='MILCO'; nom=cls=='NOMBO'
out={}
for name,mask in [('all',np.ones(len(cls),bool)),('test_years',np.isin(yr,['2010','2018']) | np.isin(yr,[2010,2018]))]:
    o=obj&mask; b=bg&mask; m=mil&mask; n=nom&mask
    out[name]=dict(n_obj=int(o.sum()),n_bg=int(b.sum()),n_milco=int(m.sum()),n_nombo=int(n.sum()),
      shadow_obj_vs_bg=auc(o,b,-sh), highlight_obj_vs_bg=auc(o,b,hi), contrast_obj_vs_bg=auc(o,b,hi-sh),
      shadow_milco_vs_nombo=auc(m,n,-sh), contrast_milco_vs_nombo=auc(m,n,hi-sh),
      med_shadow=dict(MILCO=float(np.median(sh[m])),NOMBO=float(np.median(sh[n])),BG=float(np.median(sh[b]))))
out['scores']=dict(cls=cls.tolist(),year=[str(y) for y in yr],shadow=np.round(sh,4).tolist(),highlight=np.round(hi,4).tolist())
json.dump(out,open('b2_shadow_auc.json','w')); print({k:{kk:vv for kk,vv in v.items() if 'auc' in kk or kk.startswith('shadow') or kk.startswith('contrast') or kk.startswith('highlight')} for k,v in out.items() if k!='scores'})
