import json, numpy as np, sys
from scipy.ndimage import uniform_filter, median_filter, maximum_filter
from scipy.signal import fftconvolve
exec(open('b2.py').read().split('# ---------------------------------------------------------------- 3.')[0])
src=open('b2.py').read()
exec(src[src.index('def match('):src.index('det_res = {}')])
def detect2(a, scales=(5,8,12,18), k_top=40):
    H,W=a.shape; f=W/1024.
    x=np.log(median_filter(a,3)+4.0)
    mu=uniform_filter(x,int(96*f)|1); sd=np.sqrt(np.maximum(uniform_filter(x*x,int(96*f)|1)-mu**2,1e-4))
    z=(x-mu)/sd                                   # local contrast (z-score)
    nad,n0,n1=nadir_band(a); best=np.full_like(z,-np.inf)
    for s in scales:
        s=max(2,int(round(s*f))); box=np.ones((s,s))/(s*s)
        m=fftconvolve(z,box,mode='same')          # mean z in an s x s cell
        hi_r=np.roll(m, s//2, axis=1); sh_r=np.roll(m, -s//2-s//2, axis=1)   # starboard: highlight inner(-x), shadow outer(+x)
        hi_l=np.roll(m,-s//2, axis=1); sh_l=np.roll(m, s//2+s//2, axis=1)
        sc_r=np.minimum(hi_r,-sh_r); sc_l=np.minimum(hi_l,-sh_l)
        sc=np.where(np.arange(W)[None,:]>nad, sc_r, sc_l); best=np.maximum(best,sc)
    mg=int(0.02*W); best[:,max(0,n0-mg):min(W,n1+mg)]=-np.inf; best[:mg]=best[-mg:]=-np.inf; best[:,:mg]=best[:,-mg:]=-np.inf
    pk=(best==maximum_filter(best,size=int(24*f)|1))&np.isfinite(best); ys,xs=np.where(pk); sc=best[ys,xs]; o=np.argsort(-sc)[:k_top]
    return [(float(xs[i]/W),float(ys[i]/H),float(sc[i])) for i in o]
years=sys.argv[1].split(',')
rows=[];ng=0;ni=0
for r in [s for s in scenes if s['year'] in years]:
    m,g,_=match(detect2(load(r)),r['labels']); rows+=m; ng+=g; ni+=1
curve,raw=froc(rows,ng,ni); print(years,ni,ng,[(a,round(b,3)) for a,b in curve])
json.dump(dict(curve=curve,raw=raw,n_img=ni,n_gt=ng),open(f"mf2_{'_'.join(years)}.json",'w'))
