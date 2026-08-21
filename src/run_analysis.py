
# User and Network Behavior Analytics for Cybersecurity Risk Detection
# NOTE: The included CSV is SYNTHETIC/ILLUSTRATIVE. Do not present its results
# as empirical evidence from a real organization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_user_network_behavior.csv"
OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
print(df.head())
print(df.describe(include="all"))

features = ["login_count","file_access","web_requests","email_count","device_events","after_hours"]
X = df[features].fillna(0)

# User-level aggregation
user_df = df.groupby(["user_id","department"], as_index=False)[features + ["synthetic_anomaly"]].mean()
user_df["synthetic_anomaly"] = df.groupby(["user_id","department"])["synthetic_anomaly"].max().values

# Standardized behavioral features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_df[features])

# Isolation Forest
model = IsolationForest(n_estimators=200, contamination=0.10, random_state=42)
user_df["iforest_prediction"] = model.fit_predict(X_scaled)
user_df["anomaly_score_raw"] = -model.decision_function(X_scaled)

# Convert raw score to 0-100 relative risk score for this illustrative dataset
lo, hi = user_df["anomaly_score_raw"].min(), user_df["anomaly_score_raw"].max()
user_df["risk_score"] = 100 * (user_df["anomaly_score_raw"] - lo) / (hi - lo + 1e-9)
user_df["risk_level"] = pd.cut(
    user_df["risk_score"],
    bins=[-np.inf, 30, 60, np.inf],
    labels=["Low","Medium","High"]
)
user_df["decision"] = user_df["risk_level"].map({
    "Low":"Monitor",
    "Medium":"Review activity",
    "High":"Investigate"
}).astype(str)

print(user_df.sort_values("risk_score", ascending=False).head(10))

# Save table
user_df.sort_values("risk_score", ascending=False).to_csv(
    TABLES/"user_risk_scores.csv", index=False
)

# Figure 1: Distribution of file activity
plt.figure(figsize=(8,5))
plt.hist(df["file_access"], bins=25)
plt.title("Distribution of File Access Activity")
plt.xlabel("File access events")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUT/"figure1_file_activity_distribution.png", dpi=300)
plt.show()

# Figure 2: Behavioral relationship
plt.figure(figsize=(8,5))
plt.scatter(user_df["login_count"], user_df["file_access"],
            s=50, alpha=0.75)
plt.xlabel("Average login count")
plt.ylabel("Average file access")
plt.title("User Login Activity vs. File Access")
plt.tight_layout()
plt.savefig(OUT/"figure2_login_vs_file_access.png", dpi=300)
plt.show()

# Figure 3: Risk levels
risk_counts = user_df["risk_level"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
plt.figure(figsize=(7,5))
risk_counts.plot(kind="bar")
plt.title("Illustrative User Cybersecurity Risk Levels")
plt.xlabel("Risk level")
plt.ylabel("Number of users")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT/"figure3_risk_levels.png", dpi=300)
plt.show()

# Figure 4: Top 10 risk scores
top = user_df.sort_values("risk_score", ascending=False).head(10).sort_values("risk_score")
plt.figure(figsize=(8,5))
plt.barh(top["user_id"], top["risk_score"])
plt.xlabel("Risk score (0–100)")
plt.ylabel("User")
plt.title("Top 10 Illustrative User Risk Scores")
plt.tight_layout()
plt.savefig(OUT/"figure4_top_risk_users.png", dpi=300)
plt.show()

# Figure 5: Before/after-hours behavior
after = df.groupby("after_hours")["file_access"].mean()
plt.figure(figsize=(7,5))
plt.bar(["Business hours","After hours"], [after.get(0,0), after.get(1,0)])
plt.ylabel("Mean file access")
plt.title("File Access by Access-Time Category")
plt.tight_layout()
plt.savefig(OUT/"figure5_after_hours_file_access.png", dpi=300)
plt.show()

# Basic illustrative evaluation against the synthetic anomaly marker
pred = (user_df["iforest_prediction"] == -1).astype(int)
truth = user_df["synthetic_anomaly"].astype(int)
print("\nIllustrative anomaly counts:")
print(pd.crosstab(pd.Series(truth, name="synthetic_label"),
                  pd.Series(pred, name="model_anomaly")))

print("\nRisk distribution:")
print(user_df["risk_level"].value_counts())

print("\nIMPORTANT: All outputs are generated from the included synthetic sample.")
