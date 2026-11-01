"""
HR Attrition Analysis: who leaves, when, and why
Topics: data type classification | derived metrics | group by | trend analysis | correlation (+ significance tests)
Stack: Python, pandas, NumPy, SciPy, Matplotlib, Seaborn
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
IMG = "images/"; os.makedirs(IMG, exist_ok=True); os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(8)
NAVY, TEAL, ORG, GREY, GOLD = "#264653", "#2A9D8F", "#E76F51", "#B0B8C4", "#E9C46A"
SNAP = pd.Timestamp("2026-09-30")

# ------------------------------------------------------------------
# 0. DATA (synthetic; replace with pd.read_csv for real data)
# ------------------------------------------------------------------
n = 4000
depts = {"Sales": (.28, .5, 52), "Customer Support": (.20, .4, 38), "Engineering": (.25, -.1, 85), "Finance": (.08, -.3, 70), "HR": (.05, -.2, 60), "Operations": (.14, .1, 48)}
dn = list(depts)
dept = rng.choice(dn, n, p=[depts[d][0] for d in dn])
hire_year_w = np.array([.02, .03, .04, .05, .06, .08, .09, .11, .13, .15, .16, .08]); hire_year_w = hire_year_w / hire_year_w.sum()   # 2015..2026
hire = pd.to_datetime([f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(rng.choice(np.arange(2015, 2027), n, p=hire_year_w), rng.integers(1, 13, n), rng.integers(1, 28, n))])
hire = pd.Series(np.minimum(hire.values, SNAP.to_datetime64() - np.timedelta64(30, "D")))
level = rng.choice([1, 2, 3, 4, 5], n, p=[.28, .3, .22, .14, .06])
salary = np.array([depts[d][2] for d in dept]) * (0.8 + 0.28 * level) * 1000 * rng.lognormal(0, .12, n)          # annual salary (thousands scaled)
salary = pd.Series(salary); compa = salary / salary.groupby([dept, level]).transform("median")
df = pd.DataFrame({"employee_id": [f"E{i:05d}" for i in range(1, n + 1)], "hire_date": hire, "department": dept, "job_level": level,
                   "gender": rng.choice(["Female", "Male"], n, p=[.46, .54]), "age": np.clip(rng.normal(36, 8, n), 21, 62).round().astype(int),
                   "annual_salary": salary.round(0), "overtime": rng.choice(["Yes", "No"], n, p=[.28, .72]),
                   "work_mode": rng.choice(["Onsite", "Hybrid", "Remote"], n, p=[.45, .38, .17]),
                   "satisfaction": rng.choice([1, 2, 3, 4, 5], n, p=[.08, .15, .32, .30, .15]), "manager_rating": rng.choice([1, 2, 3, 4, 5], n, p=[.05, .12, .33, .33, .17]),
                   "commute_km": np.clip(rng.gamma(2.2, 6, n), 1, 70).round(1), "training_hours": np.clip(rng.normal(28, 14, n), 0, 90).round(0).astype(int),
                   "performance": rng.choice([1, 2, 3, 4, 5], n, p=[.04, .12, .44, .30, .10])})
promo_raw = rng.exponential(2.4, n)
logit = (-1.2 + 1.0 * (df.overtime == "Yes") + 0.45 * (3 - df.satisfaction) + 0.16 * promo_raw - 3.0 * (compa - 1) + 0.022 * (df.commute_km - 12)
         + df.department.map({d: depts[d][1] for d in dn}) - 0.2 * (df.job_level - 3) - 0.3 * (df.work_mode == "Remote") - 0.2 * (df.manager_rating - 3))
leaver = rng.random(n) < 1 / (1 + np.exp(-logit))
life_days = np.maximum(90, rng.exponential(1.8 * 365, n)).astype(int)
exit_date = pd.Series(df.hire_date + pd.to_timedelta(life_days, unit="D")).where(leaver)
exit_date = exit_date.where(exit_date <= SNAP)
df["exit_date"] = exit_date; df["attrition"] = df.exit_date.notna().astype(int)
tenure_y = ((df.exit_date.fillna(SNAP) - df.hire_date).dt.days / 365.25)
df["tenure_years"] = tenure_y.round(2); df["years_since_promotion"] = np.minimum(promo_raw, tenure_y).round(1)
df["compa_ratio"] = compa.round(3)
raw = df.copy()
for c in ["hire_date", "exit_date"]: raw[c] = raw[c].dt.strftime("%Y-%m-%d")

# ------------------------------------------------------------------
# 1. DATA TYPE CLASSIFICATION
# ------------------------------------------------------------------
def classify(col, s):
    if col == "employee_id": return "Identifier"
    if "date" in col: return "Datetime"
    if col == "attrition": return "Binary target"
    if pd.api.types.is_numeric_dtype(s):
        if col in ("satisfaction", "manager_rating", "performance", "job_level"): return "Ordinal (rating scale)"
        return "Numeric - Continuous"
    return "Binary categorical" if s.nunique() == 2 else "Categorical"
types = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values, "classification": [classify(c, df[c]) for c in df.columns],
                      "missing": df.isna().sum().values, "unique_values": [df[c].nunique() for c in df.columns]})
print("=== 1. DATA TYPE CLASSIFICATION ===\n", types.to_string(index=False))
print(f"\nEmployees: {len(df):,} | Left: {df.attrition.sum():,} ({df.attrition.mean():.1%}) | Active: {(1 - df.attrition).sum():,.0f}")
print("(exit_date is empty for current employees by design, not a data error)")
fig, ax = plt.subplots(figsize=(8.5, 4.8)); cnt = types["classification"].value_counts(); b = ax.barh(cnt.index, cnt.values, color=sns.color_palette("deep", len(cnt)))
ax.bar_label(b, padding=3); ax.set_title("Columns by Data Type Classification", fontweight="bold"); ax.set_xlabel("Number of columns")
plt.tight_layout(); plt.savefig(IMG + "1_data_types.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 2. DERIVED METRICS
# ------------------------------------------------------------------
df["age_group"] = pd.cut(df.age, [0, 29, 39, 49, 100], labels=["<30", "30-39", "40-49", "50+"])
df["tenure_group"] = pd.cut(df.tenure_years, [0, 1, 2, 5, 100], labels=["<1 yr", "1-2 yrs", "2-5 yrs", "5+ yrs"])
df["pay_position"] = pd.cut(df.compa_ratio, [0, .9, 1.1, 10], labels=["Below market (<0.9)", "At market", "Above market (>1.1)"])
df["promo_gap"] = pd.cut(df.years_since_promotion, [-1, 1, 3, 100], labels=["<1 yr", "1-3 yrs", "3+ yrs"])
df["low_satisfaction"] = (df.satisfaction <= 2).astype(int)
df["risk_flags"] = (df.overtime.eq("Yes").astype(int) + df.satisfaction.le(2) + df.manager_rating.le(2) + df.compa_ratio.lt(.9) + df.commute_km.gt(25))
df["replacement_cost"] = df.annual_salary * 0.5 * df.attrition          # ASSUMPTION: replacing an employee costs 50% of annual salary
print("\n=== 2. DERIVED METRICS (sample) ===\n", df[["employee_id", "tenure_years", "tenure_group", "compa_ratio", "pay_position", "years_since_promotion", "risk_flags"]].head())

# ------------------------------------------------------------------
# 3. GROUP BY: attrition rate by dimension
# ------------------------------------------------------------------
rate = lambda col: df.groupby(col, observed=True)["attrition"].agg(["mean", "size"]).assign(mean=lambda d: d["mean"] * 100)
dims = ["department", "job_level", "overtime", "work_mode", "age_group", "pay_position", "gender", "risk_flags"]
R = {d: rate(d) for d in dims}
for d in ["department", "overtime", "pay_position"]: print(f"\nAttrition % by {d}:\n{R[d]['mean'].round(1)}")
overall = df.attrition.mean() * 100
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (d, c) in zip(axes.ravel(), [("department", ORG), ("job_level", NAVY), ("overtime", TEAL), ("pay_position", GOLD)]):
    r = R[d]["mean"]; bb = ax.bar(r.index.astype(str), r.values, color=c); ax.bar_label(bb, fmt="%.1f%%", padding=2)
    ax.axhline(overall, color="black", ls="--", lw=1, label=f"overall {overall:.1f}%"); ax.legend(); ax.set_title(f"Attrition Rate by {d.replace('_', ' ').title()}", fontweight="bold"); ax.tick_params(axis="x", rotation=18)
plt.tight_layout(); plt.savefig(IMG + "2_group_by.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 4. TREND ANALYSIS: hires, exits, headcount, attrition rate
# ------------------------------------------------------------------
years = np.arange(2015, 2027)
tr = pd.DataFrame(index=years)
tr["hires"] = df.hire_date.dt.year.value_counts().reindex(years).fillna(0).astype(int)
tr["exits"] = df.exit_date.dt.year.value_counts().reindex(years).fillna(0).astype(int)
tr["headcount_end"] = [((df.hire_date <= pd.Timestamp(f"{y}-12-31") if y < 2026 else df.hire_date <= SNAP) & ~(df.exit_date <= pd.Timestamp(f"{y}-12-31"))).sum() for y in years]
tr["headcount_start"] = tr.headcount_end.shift(1).fillna(0)
tr["attrition_rate_%"] = tr.exits / ((tr.headcount_start + tr.headcount_end) / 2) * 100
tr.loc[2026, "attrition_rate_%"] = tr.loc[2026, "attrition_rate_%"] * 12 / 9                                        # annualise the partial 2026 year
last12 = df[(df.exit_date > SNAP - pd.Timedelta(days=365)) & (df.exit_date <= SNAP)]
cost12 = last12.annual_salary.sum() * 0.5
print("\n=== 4. TREND ===\n", tr.round(1))
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
w = .38; x = np.arange(len(years)); ax[0].bar(x - w / 2, tr.hires, w, color=TEAL, label="Hires"); ax[0].bar(x + w / 2, tr.exits, w, color=ORG, label="Exits")
ax[0].plot(x, tr.headcount_end / 4, color=NAVY, marker="o", label="Headcount at year end (÷4)"); ax[0].set_xticks(x); ax[0].set_xticklabels(years, rotation=45); ax[0].legend()
ax[0].set_title("Hires, Exits and Headcount by Year", fontweight="bold")
a = tr.loc[2017:]; ax[1].plot(a.index, a["attrition_rate_%"], marker="o", color=ORG, lw=3); ax[1].set_title("Annual Attrition Rate % (2026 annualised from 9 months)", fontweight="bold"); ax[1].set_ylabel("%")
plt.tight_layout(); plt.savefig(IMG + "3_trend.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 5. DRIVERS: significance tests
# ------------------------------------------------------------------
cat_tests = []
for c in ["department", "overtime", "work_mode", "job_level", "gender", "age_group", "pay_position"]:
    tab = pd.crosstab(df[c], df.attrition); chi2, p, _, _ = stats.chi2_contingency(tab); v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    cat_tests.append((c, chi2, p, v))
cat_t = pd.DataFrame(cat_tests, columns=["feature", "chi2", "p_value", "cramers_v"]).sort_values("cramers_v", ascending=False)
num_tests = []
for c in ["satisfaction", "compa_ratio", "commute_km", "manager_rating", "training_hours", "performance", "age", "annual_salary"]:
    a, b = df.loc[df.attrition == 1, c], df.loc[df.attrition == 0, c]; u, p = stats.mannwhitneyu(a, b); auc = u / (len(a) * len(b))
    num_tests.append((c, a.median(), b.median(), p, abs(auc - .5) * 2))
num_t = pd.DataFrame(num_tests, columns=["feature", "median_left", "median_stayed", "p_value", "effect_size"]).sort_values("effect_size", ascending=False)
print("\n=== 5. SIGNIFICANCE TESTS ===\nCategorical (chi-square, Cramer's V):\n", cat_t.round(4).to_string(index=False), "\n\nNumeric (Mann-Whitney U, effect = |AUC-0.5|*2):\n", num_t.round(4).to_string(index=False))

# Kaplan-Meier retention curve (handles the fact that current employees have not left *yet*)
def km(dur, ev):
    dur, ev = np.asarray(dur), np.asarray(ev); t, sv, cur = [0.0], [1.0], 1.0
    for u in np.unique(dur[ev == 1]):
        cur *= 1 - ((dur == u) & (ev == 1)).sum() / (dur >= u).sum(); t.append(u); sv.append(cur)
    return np.array(t), np.array(sv)
def at(t, sv, x): return sv[np.searchsorted(t, x, side="right") - 1]
kt, ks = km(df.tenure_years, df.attrition); ret1, ret3 = at(kt, ks, 1), at(kt, ks, 3)
mature = df[df.tenure_years >= 3]; promo_m = mature.groupby("promo_gap", observed=True)["attrition"].mean() * 100
elig = df[df.hire_date <= SNAP - pd.Timedelta(days=365)]; first_year = ((elig.exit_date - elig.hire_date).dt.days <= 365).mean() * 100
print(f"\nKaplan-Meier retention: 1 year {ret1 * 100:.1f}% | 3 years {ret3 * 100:.1f}% | first-year attrition (cohorts with 12m of data): {first_year:.1f}%")
print("Attrition by promotion gap, employees with 3+ yrs tenure:\n", promo_m.round(1))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
r = rate("satisfaction")["mean"]; bb = axes[0, 0].bar(r.index.astype(str), r.values, color=TEAL); axes[0, 0].bar_label(bb, fmt="%.1f%%", padding=2)
axes[0, 0].axhline(overall, color="black", ls="--", lw=1); axes[0, 0].set_title("Attrition Rate by Satisfaction Score (1-5)", fontweight="bold")
bb = axes[0, 1].bar(promo_m.index.astype(str), promo_m.values, color=NAVY); axes[0, 1].bar_label(bb, fmt="%.1f%%", padding=2)
axes[0, 1].set_title("Attrition by Years Since Promotion (3+ yr tenure only)", fontweight="bold")
for grp, c in [("No", TEAL), ("Yes", ORG)]:
    g = df[df.overtime == grp]; t_, s_ = km(g.tenure_years, g.attrition); axes[1, 0].step(t_, s_, where="post", color=c, lw=2.5, label=f"Overtime: {grp}")
axes[1, 0].set_xlim(0, 8); axes[1, 0].set_ylim(0, 1.02); axes[1, 0].legend(); axes[1, 0].set_xlabel("Years since hire"); axes[1, 0].set_ylabel("Share still employed")
axes[1, 0].set_title("Retention Curve (Kaplan-Meier) by Overtime", fontweight="bold")
hm = df.pivot_table(index="overtime", columns="work_mode", values="attrition", aggfunc="mean") * 100
sns.heatmap(hm, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1, 1], cbar_kws={"label": "Attrition %"}); axes[1, 1].set_title("Attrition %: Overtime x Work Mode", fontweight="bold")
plt.tight_layout(); plt.savefig(IMG + "4_drivers.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 6. CORRELATION
# ------------------------------------------------------------------
dnum = df.assign(overtime_yes=(df.overtime == "Yes").astype(int))[["attrition", "satisfaction", "compa_ratio", "commute_km", "overtime_yes", "manager_rating", "job_level", "annual_salary", "training_hours", "performance", "age"]]
corr = dnum.corr(); cc = corr["attrition"].drop("attrition").sort_values()
print("\n=== 6. CORRELATION WITH ATTRITION ===\n", cc.round(2))
fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), width_ratios=[1.35, 1])
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool), 1), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[0], annot_kws={"size": 7}); ax[0].set_title("Correlation Matrix", fontweight="bold")
ax[1].barh(cc.index, cc.values, color=np.where(cc.values > 0, ORG, TEAL)); ax[1].axvline(0, color="black", lw=.8); ax[1].set_title("Correlation with Attrition", fontweight="bold")
plt.tight_layout(); plt.savefig(IMG + "5_correlation.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 7. LINKEDIN DASHBOARD
# ------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9)); fig.suptitle("HR Attrition Analysis  |  Python Data Analytics", fontsize=20, fontweight="bold", y=.98)
gs = fig.add_gridspec(2, 2, hspace=.42, wspace=.3)
a1 = fig.add_subplot(gs[0, 0]); r = R["department"]["mean"].sort_values(); bb = a1.barh(r.index, r.values, color=ORG); a1.bar_label(bb, fmt="%.1f%%", padding=3); a1.axvline(overall, color="black", ls="--", lw=1)
a1.set_title("Group By: % of Employees Who Left, by Department", fontweight="bold"); a1.set_xlim(0, r.max() * 1.2)
a2 = fig.add_subplot(gs[0, 1]); r = R["pay_position"]["mean"]; bb = a2.bar(["Below\nmarket", "At\nmarket", "Above\nmarket"], r.values, color=[ORG, GOLD, TEAL]); a2.bar_label(bb, fmt="%.1f%%", padding=2)
a2.set_title("Derived Metric: % Who Left by Pay Position (compa-ratio)", fontweight="bold")
a3 = fig.add_subplot(gs[1, 0]); x = np.arange(len(years)); a3.bar(x - w / 2, tr.hires, w, color=TEAL, label="Hires"); a3.bar(x + w / 2, tr.exits, w, color=ORG, label="Exits"); a3.set_xticks(x); a3.set_xticklabels(years, rotation=45, fontsize=8)
a3.legend(); a3.set_title("Trend: Hires vs Exits by Year", fontweight="bold")
a4 = fig.add_subplot(gs[1, 1]); a4.barh(cc.index, cc.values, color=np.where(cc.values > 0, ORG, TEAL)); a4.axvline(0, color="black", lw=.8); a4.set_title("Correlation: Features vs Attrition", fontweight="bold"); a4.tick_params(axis="y", labelsize=8)
plt.savefig(IMG + "0_linkedin_dashboard.png", dpi=200, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------
# INSIGHTS + EXPORT
# ------------------------------------------------------------------
print("\n=== KEY INSIGHTS ===")
print(f"- Overall attrition {overall:.1f}% ({df.attrition.sum():,} of {len(df):,}) | exits in last 12 months: {len(last12):,} | est. replacement cost {cost12 / 1e6:.2f}M (assumes 50% of salary)")
print(f"- Highest dept: {R['department']['mean'].idxmax()} {R['department']['mean'].max():.1f}% | lowest: {R['department']['mean'].idxmin()} {R['department']['mean'].min():.1f}%")
print(f"- Overtime Yes {R['overtime'].loc['Yes','mean']:.1f}% vs No {R['overtime'].loc['No','mean']:.1f}% | below-market pay {R['pay_position'].iloc[0]['mean']:.1f}% vs above {R['pay_position'].iloc[2]['mean']:.1f}%")
print(f"- Retention: {ret1 * 100:.1f}% still employed after 1 year, {ret3 * 100:.1f}% after 3 years | first-year attrition {first_year:.1f}%")
print(f"- Promo gap (3+ yr tenure): {promo_m.round(1).to_dict()}")
print(f"- Employees with 3+ risk flags: {(df.risk_flags >= 3).mean() * 100:.1f}% of staff, attrition {df[df.risk_flags >= 3].attrition.mean() * 100:.1f}% vs 0 flags {df[df.risk_flags == 0].attrition.mean() * 100:.1f}%")
print(f"- Gender attrition: {R['gender']['mean'].round(1).to_dict()} (chi-square p = {cat_t.set_index('feature').loc['gender','p_value']:.3f})")
out = df.copy()
for c in ["hire_date", "exit_date"]: out[c] = out[c].dt.date
out.to_csv("data/employees.csv", index=False); tr.round(2).to_csv("data/yearly_trend.csv"); num_t.round(4).to_csv("data/numeric_tests.csv", index=False); cat_t.round(4).to_csv("data/categorical_tests.csv", index=False)
print("\nSaved CSVs to data/")
