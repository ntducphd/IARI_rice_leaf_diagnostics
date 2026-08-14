import csv, itertools, random
import numpy as np
from scipy.stats import beta
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import os as _os
_D = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "derived"))
rows=list(csv.DictReader(open(_os.path.join(_D, "plants_30DAT.tsv")),delimiter="\t"))
G=sorted({r["genotype"] for r in rows})
y=np.array([1 if r["treatment"]=="NStress" else 0 for r in rows])
gid=np.array([r["genotype"] for r in rows])
X={k:np.array([[float(r[k])] for r in rows]) for k in ("rCCI_30","QY_BL_30")}
X["both"]=np.hstack([X["rCCI_30"],X["QY_BL_30"]])

def cp(k,n,a=.05):
    lo=0 if k==0 else beta.ppf(a/2,k,n-k+1)
    hi=1 if k==n else beta.ppf(1-a/2,k+1,n-k)
    return lo,hi

def logo(M,yy=y):
    pred=np.empty(len(yy),dtype=int)
    for g in G:
        te=gid==g; tr=~te
        m=LDA().fit(M[tr],yy[tr]); pred[te]=m.predict(M[te])
    return pred

print("LEAVE-ONE-GENOTYPE-OUT CV  (15 folds, 6 held-out plants per fold, 90 predictions pooled)\n")
print("%-10s %7s %8s %16s %10s %10s"%("model","correct","acc","95% CI (Clopper-Pearson)","sens","spec"))
res={}
for name in ("QY_BL_30","rCCI_30","both"):
    p=logo(X[name]); k=int((p==y).sum()); res[name]=(p,k)
    lo,hi=cp(k,90)
    sens=((p==1)&(y==1)).sum()/ (y==1).sum(); spec=((p==0)&(y==0)).sum()/(y==0).sum()
    print("%-10s %4d/90 %8.4f   [%.3f, %.3f]   %8.3f %10.3f"%(name,k,k/90,lo,hi,sens,spec))

print("\nBASELINES")
print("  majority class            45/90 = 0.500")
rng=random.Random(0)
worse=0; NP=2000
for _ in range(NP):
    yp=np.array(y); rng.shuffle(yp)
    if (logo(X["both"],yp)==yp).sum() >= res["both"][1]: worse+=1
print("  permutation null (n=%d): p = %.4f"%(NP,(worse+1)/(NP+1)))

print("\nPER-GENOTYPE (held-out), correct out of 6")
for g in G:
    m=gid==g
    line=" ".join("%-9s %d/6"%(n,int((res[n][0][m]==y[m]).sum())) for n in ("QY_BL_30","rCCI_30","both"))
    print("  %-20s %s"%(g,line))

print("\nGENOTYPE-LEVEL: folds in which ALL 6 plants were correct")
for n in ("QY_BL_30","rCCI_30","both"):
    k=sum(1 for g in G if (res[n][0][gid==g]==y[gid==g]).all())
    lo,hi=cp(k,15)
    print("  %-10s %2d/15  95%% CI [%.3f, %.3f]"%(n,k,lo,hi))

print("\nSINGLE-THRESHOLD RULE on QY_BL_30 (no model fitted)")
q=X["QY_BL_30"].ravel()
best=max(np.arange(.60,.85,.001), key=lambda t: (((q<t).astype(int))==y).sum())
k=int((((q<best).astype(int))==y).sum()); lo,hi=cp(k,90)
print("  threshold QY_BL_30 < %.3f  ->  %d/90 = %.4f  95%% CI [%.3f, %.3f]"%(best,k,k/90,lo,hi))
print("  Wu et al. 2024 published rCCI threshold 0.909 applied unchanged:")
r=X["rCCI_30"].ravel(); k2=int((((r<0.909).astype(int))==y).sum()); lo2,hi2=cp(k2,90)
print("    %d/90 = %.4f  95%% CI [%.3f, %.3f]"%(k2,k2/90,lo2,hi2))

# ---- export for figures and tables ----
import os, pandas as pd
_T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tables")
_T = os.path.abspath(_T); os.makedirs(_T, exist_ok=True)

pd.DataFrame([{"genotype": g,
               **{n: int((res[n][0][gid == g] == y[gid == g]).sum()) for n in ("QY_BL_30", "rCCI_30", "both")}}
              for g in G]).to_csv(os.path.join(_T, "logocv_per_genotype.csv"), index=False)

_rows = []
for n in ("rCCI_30", "QY_BL_30", "both"):
    p, k = res[n]
    lo, hi = cp(k, 90)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    folds = sum(1 for g in G if (res[n][0][gid == g] == y[gid == g]).all())
    flo, fhi = cp(folds, 15)
    _rows.append(dict(model=n, n_traits=2 if n == "both" else 1, correct=k, n=90,
                      accuracy=k / 90, acc_lo=lo, acc_hi=hi,
                      sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                      ppv=tp / (tp + fp) if tp + fp else float("nan"),
                      npv=tn / (tn + fn) if tn + fn else float("nan"),
                      TP=tp, TN=tn, FP=fp, FN=fn,
                      genotype_folds_perfect=folds, fold_lo=flo, fold_hi=fhi))
pd.DataFrame(_rows).to_csv(os.path.join(_T, "Table3_performance.csv"), index=False, float_format="%.6g")
print("\nexported: logocv_per_genotype.csv, Table3_performance.csv")
