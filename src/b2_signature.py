"""B2: box-scale highlight/shadow profile per class (median, IQR) -> b2_signature.json"""
import json, numpy as np
from scipy.ndimage import median_filter
exec(open('b2.py').read().split('# ---------------------------------------------------------------- 3.')[0])
def prof2(a,cx,cy,w,h,nad,L=1.5,npts=61):
    H,W=a.shape; sgn=1 if cx>nad else -1; u=np.linspace(-L,L,npts)
    xs=np.clip((cx+sgn*u*w).round().astype(int),0,W-1); y0,y1=int(max(0,cy-h/4)),int(min(H,cy+h/4+1))
    p=a[y0:y1][:,xs].mean(0)
    ys=np.r_[max(0,int(cy-5*h)):max(0,int(cy-3*h)), min(H,int(cy+3*h)):min(H,int(cy+5*h))]
    bg=np.median(a[ys][:,xs]) if len(ys) else np.median(p)
    return np.log10(p/(bg+1)+1e-3)*10
P={'MILCO':[],'NOMBO':[],'BG':[]}
for r in scenes:
    a=median_filter(load(r),3); W=a.shape[1]; nad,_,_=nadir_band(a)
    for c,x,y,w,h in r['labels']:
        if w*W<4: continue
        P['MILCO' if c==0 else 'NOMBO'].append(prof2(a,x*W,y*W,w*W,h*W,nad))
    if r['labels'] and len(P['BG'])<1200:
        for _ in range(4):
            sd=rng.choice([-1,1]); cx=nad+sd*rng.uniform(0.1,0.45)*W; cy=rng.uniform(0.1,0.9)*W
            P['BG'].append(prof2(a,cx,cy,np.median([l[3] for l in r['labels']])*W,np.median([l[4] for l in r['labels']])*W,nad))
u=np.linspace(-1.5,1.5,61); out={}
for k,v in P.items():
    v=np.array(v); out[k]=dict(med=np.median(v,0).round(3).tolist(),q25=np.percentile(v,25,0).round(3).tolist(),q75=np.percentile(v,75,0).round(3).tolist(),n=len(v),
        frac_shadow=float(((v[:,(u>0)&(u<0.6)].min(1))<-3).mean()), frac_high=float(((v[:,(u>-0.6)&(u<0.1)].max(1))>3).mean()))
json.dump(dict(u=u.round(3).tolist(),**out),open('b2_signature.json','w'))
