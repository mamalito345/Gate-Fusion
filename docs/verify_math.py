import numpy as np
rng=np.random.default_rng(0)
d,K=64,40
def sm(z): z=z-z.max(); e=np.exp(z); return e/e.sum()
def loss(A,B,f,g,W1,W2,b1,b2,y):
    fp=f+A@g; gp=g+B@f
    p1=sm(W1@fp+b1); p2=sm(W2@gp+b2)
    return -np.log(p1[y])-np.log(p2[y]),p1,p2,fp,gp
f=rng.normal(size=d);g=rng.normal(size=d)
W1=rng.normal(size=(K,d))/np.sqrt(d);W2=rng.normal(size=(K,d))/np.sqrt(d)
b1=rng.normal(size=K)*.1;b2=rng.normal(size=K)*.1;y=7
A=rng.normal(size=(d,d))*.05;B=rng.normal(size=(d,d))*.05
# 1 logit identity
_,p1,p2,fp,gp=loss(A,B,f,g,W1,W2,b1,b2,y)
z1=W1@fp+b1
alt=W1@f+(W1@A)@g+b1
print("logit identity err",np.abs(z1-alt).max())
# 2 gradients vs finite diff
def num_grad(fn,X,h=1e-6):
    G=np.zeros_like(X)
    for i in np.ndindex(*X.shape):
        X[i]+=h;a=fn();X[i]-=2*h;b=fn();X[i]+=h;G[i]=(a-b)/(2*h)
    return G
for name,A0,B0 in [("zero-init",np.zeros((d,d)),np.zeros((d,d))),("random",A,B)]:
    Ar=A0.copy();Br=B0.copy()
    L,p1,p2,fp,gp=loss(Ar,Br,f,g,W1,W2,b1,b2,y)
    e=np.zeros(K);e[y]=1
    dA=np.outer(W1.T@(p1-e),g)
    dB=np.outer(W2.T@(p2-e),f)
    nA=num_grad(lambda:loss(Ar,Br,f,g,W1,W2,b1,b2,y)[0],Ar)
    nB=num_grad(lambda:loss(Ar,Br,f,g,W1,W2,b1,b2,y)[0],Br)
    print(name,"dA err",np.abs(dA-nA).max(),"|dA|",np.abs(dA).max(),"dB err",np.abs(dB-nB).max())
    df=W1.T@(p1-e)+Br.T@(W2.T@(p2-e))
    dg=Ar.T@(W1.T@(p1-e))+W2.T@(p2-e)
    ff=f.copy();gg=g.copy()
    nf=num_grad(lambda:loss(Ar,Br,ff,g,W1,W2,b1,b2,y)[0],ff)
    ng=num_grad(lambda:loss(Ar,Br,f,gg,W1,W2,b1,b2,y)[0],gg)
    print(name,"df err",np.abs(df-nf).max(),"dg err",np.abs(dg-ng).max())
# at zero init df has no B-term
# 3 rank: concatenation head capacity: W1[I A] has rank K when W1 full row rank and d>=K
M=W1@np.hstack([np.eye(d),A]);print("rank [W1, W1A]",np.linalg.matrix_rank(M),"K",K)
# any V (K x 2d) = [V_f V_g] reachable if V_f=W1 -> V_g=W1 A: need W1 A = V_g solvable iff row space... W1 full row rank => A exists
Vg=rng.normal(size=(K,d)); A_sol=np.linalg.pinv(W1)@Vg
print("W1 A = Vg solvable err",np.abs(W1@A_sol-Vg).max())
# but V_f is fixed to W1 (not free) -> only g-block free
# 4 zero-init: first step on A decreases loss
A0=np.zeros((d,d));L0,p1,_,_,_=loss(A0,np.zeros((d,d)),f,g,W1,W2,b1,b2,y)
e=np.zeros(K);e[y]=1;dA=np.outer(W1.T@(p1-e),g)
L1=loss(A0-1e-3*dA,np.zeros((d,d)),f,g,W1,W2,b1,b2,y)[0]
print("loss step",L0,L1)
# 5 averaging softmax vs logit
