import glob, numpy as np, json
from PIL import Image
def feat(f, T=None):
    im=Image.open(f).convert('L').resize((64,64), Image.BILINEAR); a=np.asarray(im,float)
    return a
res={}
allpairs=[]
for y in ['2010','2015','2017','2018','2021']:
    fs=sorted(glob.glob(f'raw/{y}/{y}/*.jpg'))
    sizes=[Image.open(f).size[0] for f in fs]
    A=np.array([feat(f) for f in fs])
    def tf(a,k):  # 8 dihedral transforms
        b=np.rot90(a,k%4); return np.fliplr(b) if k>=4 else b
    big=[i for i,s in enumerate(sizes) if s==1024]; small=[i for i,s in enumerate(sizes) if s==416]
    def z(a): a=a-a.mean(); return a/ (np.linalg.norm(a)+1e-9)
    B=np.array([z(A[i]).ravel() for i in big])
    matches=[]
    for j in small:
        best=(-1,None,None)
        for k in range(8):
            v=z(tf(A[j],k)).ravel(); c=B@v; i=int(np.argmax(c))
            if c[i]>best[0]: best=(float(c[i]),big[i],k)
        matches.append((j,best[1],best[0],best[2]))
    cs=np.array([m[2] for m in matches]) if matches else np.array([])
    tcount={}
    for m in matches:
        if m[2]>0.9: tcount[m[3]]=tcount.get(m[3],0)+1
    print(y,'n',len(fs),'1024:',len(big),'416:',len(small),'416 with match>0.9:',int((cs>0.9).sum()) if len(cs) else 0,'transform k counts',tcount, 'corr pct',np.percentile(cs,[5,50]).round(3) if len(cs) else '')
    res[y]=dict(files=[f.split('/')[-1] for f in fs],sizes=sizes,matches=[(fs[j].split('/')[-1],fs[i].split('/')[-1],c,k) for j,i,c,k in matches])
json.dump(res,open('dups.json','w'))
