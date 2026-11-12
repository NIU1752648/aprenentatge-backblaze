import sys, numpy as np, pandas as pd
d = pd.read_parquet(sys.argv[1])
H = [7, 30, 60]; ERR = [5, 187, 188, 197, 198]
print("rows", len(d), "drives", d.serial_number.nunique(), "models", d.model.nunique())
for h in H:
    y = d[f"y_{h}"]
    print(f"t={h:2d} events {int(y.sum()):5d} unknown {int(y.isna().sum()):6d} usable {int(y.notna().sum()):8d} rate {y.mean():.5f}")
d["err_any"] = (d[[f"smart_{i}_raw" for i in ERR]].fillna(0) > 0).any(axis=1)
print("\nshare with any error counter > 0:", round(d.err_any.mean(), 4))
for h in H:
    k = d[d[f"y_{h}"].notna()]
    ct = pd.crosstab(k.err_any, k[f"y_{h}"])
    p = ct.loc[True, 1.0] / ct.loc[True].sum(); r = ct.loc[True, 1.0] / ct[1.0].sum()
    print(f"t={h}: rule 'any error>0' precision {p:.4f} recall {r:.3f} flagged {ct.loc[True].sum()}")
print("\nmissing share of SMART columns:")
print(d.filter(like="smart_").isna().mean().round(3).sort_values().to_string())
print("\ntop models\n", d.groupby("model").agg(n=("serial_number", "size"), fail60=("y_60", "mean")).sort_values("n", ascending=False).head(12).round(5))
print("\nper-model missing of smart_187_raw (top8)\n", d.groupby("model").smart_187_raw.apply(lambda s: s.isna().mean()).loc[d.model.value_counts().index[:8]].round(2))
