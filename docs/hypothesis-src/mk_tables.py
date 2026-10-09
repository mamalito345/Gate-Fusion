import csv
rows=list(csv.DictReader(open('../../results/per_stage_accuracy.csv')))
by={(r['arch'],r['stage']):r for r in rows}
archs=list(dict.fromkeys(r['arch'] for r in rows))
tex=lambda a:a.replace('_','\\_')
f=lambda r,k:float(r[k])*100
# table 1: oracle & headroom & gap
out=[]
out.append(r"\begin{tabular}{@{}l rrr rrr rrr@{}}\toprule")
out.append(r" & \multicolumn{3}{c}{branch gap $|a_{\mathrm{IR}}-a_{\mathrm D}|$} & \multicolumn{3}{c}{oracle} & \multicolumn{3}{c}{headroom (oracle $-$ avg.)}\\")
out.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}")
out.append(r"backbone & S1 & S2 & S3 & S1 & S2 & S3 & S1 & S2 & S3\\\midrule")
sums=[[0]*9 for _ in range(1)][0]
for a in archs:
    vals=[]
    for s in ['stage1','stage2','stage3']: r=by[(a,s)]; vals.append(abs(f(r,'ir_branch_acc')-f(r,'depth_branch_acc')))
    for s in ['stage1','stage2','stage3']: vals.append(f(by[(a,s)],'oracle_acc'))
    for s in ['stage1','stage2','stage3']: r=by[(a,s)]; vals.append(f(r,'oracle_acc')-f(r,'avg_fusion_acc'))
    sums=[x+y for x,y in zip(sums,vals)]
    out.append(tex(a)+" & "+" & ".join(f"{v:.1f}" for v in vals)+r"\\")
out.append(r"\midrule")
out.append(r"mean & "+" & ".join(f"{v/6:.1f}" for v in sums)+r"\\")
out.append(r"\bottomrule\end{tabular}")
open('tab_gap.tex','w').write("\n".join(out))
print("\n".join(out[-3:]))
# oracle change
d2=[f(by[(a,'stage2')],'oracle_acc')-f(by[(a,'stage1')],'oracle_acc') for a in archs]
d3=[f(by[(a,'stage3')],'oracle_acc')-f(by[(a,'stage1')],'oracle_acc') for a in archs]
print([round(x,1) for x in d2],sum(d2)/6,[round(x,1) for x in d3],sum(d3)/6)
# table 2: deltas
out=[r"\begin{tabular}{@{}l rrr rr@{}}\toprule",
r"backbone & plain avg.\ S1 (\%) & S2 $\Delta$ & S3 $\Delta$ & best single S1 (\%) & S3 $-$ best single\\\midrule"]
for a in archs:
    s1=by[(a,'stage1')];s2=by[(a,'stage2')];s3=by[(a,'stage3')]
    best=max(f(s1,'ir_branch_acc'),f(s1,'depth_branch_acc'))
    out.append(f"{tex(a)} & {f(s1,'avg_fusion_acc'):.1f} & {f(s2,'avg_fusion_acc')-f(s1,'avg_fusion_acc'):+.1f} & {f(s3,'avg_fusion_acc')-f(s1,'avg_fusion_acc'):+.1f} & {best:.1f} & {f(s3,'avg_fusion_acc')-best:+.1f}\\\\")
out.append(r"\bottomrule\end{tabular}")
open('tab_delta.tex','w').write("\n".join(out))
print("\n".join(out))
