"""Figure 13 (tariff-design sweep) and results/tariff_sweep_summary.csv from results/tariff_sweep.csv."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","pdf.fonttype":42})
s=pd.read_csv("results/tariff_sweep.csv")
# add base 15.51 from main runs (Aug, Nov)
base=[]
for site in ["jpl","caltech"]:
    m=pd.concat([pd.read_csv(f"results/{site}_metrics.csv"),pd.read_csv(f"results/{site}_metrics_topup.csv")])
    b=m[m.month.isin(["2019-08","2019-11"])&((m.policy.isin(["uncontrolled","offline"]))|((m.policy=="mpc_topup")&m.q.isna()))].copy()
    b.insert(0,"demand_charge",15.51); base.append(b)
s=pd.concat([s]+base,ignore_index=True)
g=s.groupby(["site","demand_charge","policy"]).agg(bill=("bill","mean"),peak=("peak_all_kw","mean"),served=("served_ratio","mean"),en=("energy_cost","mean")).reset_index()
p=g.pivot_table(index=["site","demand_charge"],columns="policy",values=["bill","peak","served","en"])
p.columns=[f"{a}_{b}" for a,b in p.columns]; p=p.reset_index()
p["saving_topup_pct"]=100*(1-p.bill_mpc_topup/p.bill_uncontrolled)
p["saving_opt_pct"]=100*(1-p.bill_offline/p.bill_uncontrolled)
p["peakred_topup_pct"]=100*(1-p.peak_mpc_topup/p.peak_uncontrolled)
p["peakred_opt_pct"]=100*(1-p.peak_offline/p.peak_uncontrolled)
p["saving_topup_usd"]=p.bill_uncontrolled-p.bill_mpc_topup
p.to_csv("results/tariff_sweep_summary.csv",index=False)
print(p[["site","demand_charge","bill_uncontrolled","bill_mpc_topup","bill_offline","saving_topup_pct","saving_topup_usd","peak_uncontrolled","peak_mpc_topup","peak_offline","peakred_topup_pct","peakred_opt_pct","served_mpc_topup","en_uncontrolled","en_mpc_topup"]].round(2).to_string())
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0))
for site,ls,lab in [("jpl","-","JPL"),("caltech","--","Caltech")]:
    d=p[p.site==site]
    axs[0].plot(d.demand_charge,d.peakred_topup_pct,ls,marker="o",color="#9467bd",label=f"{lab}: MPC + top-up")
    axs[0].plot(d.demand_charge,d.peakred_opt_pct,ls,marker="*",color="#2ca02c",label=f"{lab}: hindsight optimum")
    axs[1].plot(d.demand_charge,d.saving_topup_usd,ls,marker="o",color="#9467bd",label=f"{lab}: MPC + top-up")
    axs[1].plot(d.demand_charge,d.bill_uncontrolled-d.bill_offline,ls,marker="*",color="#2ca02c",label=f"{lab}: hindsight optimum")
for a in axs: a.axvline(15.51,color="k",lw=0.6,ls=":"); a.set_xlabel("Demand charge (USD/kW-month)")
axs[0].set_ylabel("Peak reduction vs uncontrolled (%)"); axs[1].set_ylabel("Bill saving vs uncontrolled (USD/month)")
axs[0].legend(frameon=False,fontsize=6.5); fig.tight_layout()
for e in ("svg","pdf"): fig.savefig(f"paper/figures/Fig13_tariff_design.{e}",bbox_inches="tight")
# LaTeX table
t=p.copy(); t["Site"]=t.site.map({"jpl":"JPL","caltech":"Caltech"})
rows=[]
for _,r in t.sort_values(["site","demand_charge"],key=lambda x: x.map({"jpl":0,"caltech":1}) if x.name=="site" else x).iterrows():
    rows.append([r.Site,f"{r.demand_charge:.2f}",f"{r.peak_uncontrolled:.1f}",f"{r.peak_mpc_topup:.1f}",f"{r.peak_offline:.1f}",f"{r.peakred_topup_pct:.1f}",f"{r.bill_uncontrolled:,.0f}",f"{r.bill_mpc_topup:,.0f}",f"{r.saving_topup_pct:.1f}",f"{100*r.served_mpc_topup:.2f}"])
hdr=["Site","Demand charge (USD/kW-mo)","Peak unc. (kW)","Peak top-up (kW)","Peak opt. (kW)","Peak red. (\\%)","Bill unc. (USD)","Bill top-up (USD)","Saving (\\%)","Served top-up (\\%)"]
L=[r"\begin{table}[!htbp]",r"\centering",r"\caption{Tariff-design sweep (means over August and November 2019): how the demand-charge level changes the peak reduction and bill saving delivered by MPC + top-up. 15.51\,USD/kW-month is the TOU-EV-4 value.}",r"\label{tab:tariff_sweep}",r"\small",r"\resizebox{\textwidth}{!}{%",r"\begin{tabular}{lrrrrrrrrr}",r"\hline"," & ".join(hdr)+r" \\",r"\hline"]+[" & ".join(x)+r" \\" for x in rows]+[r"\hline",r"\end{tabular}}",r"\end{table}"]
open("paper/tables/Table8_tariff_sweep.tex","w").write("\n".join(L)+"\n")
fig,ax=plt.subplots(figsize=(7.2,0.45+0.24*(len(rows)+1)));ax.axis("off")
tb=ax.table(cellText=rows,colLabels=[h.replace("\\%","%") for h in hdr],loc="center",cellLoc="center");tb.auto_set_font_size(False);tb.set_fontsize(6);tb.scale(1,1.25)
for (i,j),c in tb.get_celld().items():
    c.set_linewidth(0.3)
    if i==0: c.set_facecolor("#e6e6e6"); c.set_text_props(weight="bold")
ax.set_title("Table 8. Tariff-design sweep (August and November 2019)",fontsize=8)
for e in ("svg","pdf"): fig.savefig(f"paper/figures/Table8_tariff_sweep.{e}",bbox_inches="tight")
