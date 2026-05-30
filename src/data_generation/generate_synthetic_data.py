import pandas as pd
import numpy as np

np.random.seed(42)
N = 15000

# ════════════════════════════════════════════════════════
#  GÉNÉRATION DE BASE (même code qu'avant)
# ════════════════════════════════════════════════════════

age = np.random.randint(18, 75, N)
gender = np.random.choice(['Male', 'Female'], N)
senior_citizen = (age >= 65).astype(int)
has_partner = np.random.choice([0, 1], N, p=[0.46, 0.54])
has_dependents = np.where(
    has_partner == 1,
    np.random.choice([0, 1], N, p=[0.55, 0.45]),
    np.random.choice([0, 1], N, p=[0.85, 0.15])
)

contract = np.random.choice(
    ['Month-to-month', 'One year', 'Two year'],
    N, p=[0.55, 0.25, 0.20]
)
tenure = np.where(
    contract == 'Month-to-month',
    np.random.exponential(scale=14, size=N).clip(1, 72).astype(int),
    np.where(
        contract == 'One year',
        np.random.randint(12, 72, N),
        np.random.randint(24, 72, N)
    )
)

internet_service = np.random.choice(
    ['DSL', 'Fiber optic', 'No'],
    N, p=[0.35, 0.45, 0.20]
)
online_security = np.where(
    internet_service == 'No', 'No',
    np.random.choice(['Yes', 'No'], N, p=[0.38, 0.62])
)
tech_support = np.where(
    internet_service == 'No', 'No',
    np.random.choice(['Yes', 'No'], N, p=[0.35, 0.65])
)
streaming_tv = np.where(
    internet_service == 'No', 'No',
    np.random.choice(['Yes', 'No'], N, p=[0.44, 0.56])
)
num_services = (
    (online_security == 'Yes').astype(int) +
    (tech_support == 'Yes').astype(int) +
    (streaming_tv == 'Yes').astype(int)
)

base_charge = np.where(
    internet_service == 'Fiber optic',
    np.random.uniform(70, 110, N),
    np.where(
        internet_service == 'DSL',
        np.random.uniform(40, 70, N),
        np.random.uniform(20, 40, N)
    )
)
service_charge = num_services * np.random.uniform(8, 15, N)
monthly_charges = (base_charge + service_charge).round(2)
total_charges = (monthly_charges * tenure *
                 np.random.uniform(0.95, 1.05, N)).round(2)

payment_method = np.random.choice(
    ['Electronic check', 'Mailed check',
     'Bank transfer', 'Credit card'],
    N, p=[0.34, 0.23, 0.22, 0.21]
)
paperless_billing = np.random.choice([0, 1], N, p=[0.41, 0.59])

# Churn score
churn_score = np.zeros(N)
churn_score += np.where(contract == 'Month-to-month', 1.8, 0)
churn_score += np.where(contract == 'One year', 0.5, 0)
churn_score += np.where(tenure <= 6, 1.5, 0)
churn_score += np.where((tenure > 6) & (tenure <= 12), 0.8, 0)
churn_score += np.where(tenure > 36, -1.2, 0)
churn_score += np.where(
    (internet_service == 'Fiber optic') & (tech_support == 'No'), 1.2, 0)
churn_score += np.where(
    (internet_service == 'Fiber optic') & (tech_support == 'Yes'), -0.6, 0)
churn_score += np.where(monthly_charges > 85, 0.7, 0)
churn_score += np.where(monthly_charges < 35, -0.5, 0)
churn_score -= num_services * 0.4
churn_score += np.where(payment_method == 'Electronic check', 0.4, 0)
churn_score -= senior_citizen * 0.3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

churn_prob = sigmoid(churn_score - 1.5)
churn = (np.random.uniform(0, 1, N) < churn_prob).astype(int)

# ════════════════════════════════════════════════════════
#  ASSEMBLAGE PROPRE D'ABORD
# ════════════════════════════════════════════════════════

df = pd.DataFrame({
    'customer_id'      : [f'SYN-{i:05d}' for i in range(N)],
    'age'              : age.astype(float),
    'gender'           : gender,
    'senior_citizen'   : senior_citizen,
    'has_partner'      : has_partner,
    'has_dependents'   : has_dependents,
    'tenure_months'    : tenure.astype(float),
    'contract'         : contract,
    'internet_service' : internet_service,
    'online_security'  : online_security,
    'tech_support'     : tech_support,
    'streaming_tv'     : streaming_tv,
    'num_services'     : num_services,
    'monthly_charges'  : monthly_charges,
    'total_charges'    : total_charges,
    'payment_method'   : payment_method,
    'paperless_billing': paperless_billing,
    'churn'            : churn
})

print("Dataset propre généré.")
print(f"Shape : {df.shape}")
print(f"Churn rate : {df['churn'].mean():.1%}\n")


# ════════════════════════════════════════════════════════
#  INJECTION DE PROBLÈMES RÉALISTES
#  Chaque problème a une raison métier expliquée
# ════════════════════════════════════════════════════════

df_dirty = df.copy()
rng = np.random.default_rng(seed=99)


# ────────────────────────────────────────────────────────
#  1. VALEURS MANQUANTES
#  Raison : données réelles ont toujours des champs non remplis
# ────────────────────────────────────────────────────────

def inject_missing(series, pct, random_state=None):
    """Met NaN sur pct% des lignes de façon aléatoire."""
    mask = rng.random(len(series)) < pct
    result = series.copy().astype(object)
    result[mask] = np.nan
    return result

# age : 4% manquant — formulaire non rempli à la souscription
df_dirty['age'] = inject_missing(df_dirty['age'], 0.04)

# tenure_months : 3% manquant — migration de système legacy
df_dirty['tenure_months'] = inject_missing(df_dirty['tenure_months'], 0.03)

# monthly_charges : 2% manquant — erreur de facturation
df_dirty['monthly_charges'] = inject_missing(df_dirty['monthly_charges'], 0.02)

# total_charges : 5% manquant — clients très récents (tenure=0 ou 1)
# Manquant NON aléatoire : concentré sur les nouveaux clients
new_customer_idx = df_dirty[df_dirty['tenure_months'] <= 2].index
missing_tc_idx = rng.choice(
    new_customer_idx,
    size=int(len(new_customer_idx) * 0.60),
    replace=False
)
df_dirty.loc[missing_tc_idx, 'total_charges'] = np.nan

# payment_method : 2.5% manquant — champ optionnel en ligne
df_dirty['payment_method'] = inject_missing(df_dirty['payment_method'], 0.025)

# gender : 1.5% manquant — refus de déclarer
df_dirty['gender'] = inject_missing(df_dirty['gender'], 0.015)

print("── Valeurs manquantes injectées ──")
missing_report = df_dirty.isnull().sum()
missing_report = missing_report[missing_report > 0]
print(missing_report.to_string())
print()


# ────────────────────────────────────────────────────────
#  2. VALEURS ABERRANTES (OUTLIERS)
#  Raison : erreurs de saisie, bugs système, cas extrêmes réels
# ────────────────────────────────────────────────────────

n_outliers = int(N * 0.015)  # 1.5% du dataset = 225 lignes

# monthly_charges : valeurs impossibles (erreur de saisie × 10)
outlier_idx = rng.choice(N, size=n_outliers, replace=False)
df_dirty.loc[outlier_idx[:75], 'monthly_charges'] = (
    df_dirty.loc[outlier_idx[:75], 'monthly_charges'] * 10
)  # ex: 85.50 → 855.00

# monthly_charges : valeurs négatives (bug de crédit/remboursement)
df_dirty.loc[outlier_idx[75:110], 'monthly_charges'] = (
    -df_dirty.loc[outlier_idx[75:110], 'monthly_charges']
)

# age : valeurs impossibles
df_dirty.loc[outlier_idx[110:140], 'age'] = rng.choice(
    [0, 1, 150, 200, -5], size=30
)

# tenure_months : tenure > 72 mois (impossible dans notre contexte)
df_dirty.loc[outlier_idx[140:170], 'tenure_months'] = rng.integers(
    73, 300, size=30
)

# total_charges : zéro pour des clients avec tenure élevé (incohérence)
long_tenure_idx = df_dirty[df_dirty['tenure_months'] > 24].index
zero_tc_idx = rng.choice(long_tenure_idx, size=40, replace=False)
df_dirty.loc[zero_tc_idx, 'total_charges'] = 0.0

print("── Outliers injectés ──")
print(f"monthly_charges > 300  : "
      f"{(df_dirty['monthly_charges'] > 300).sum()} lignes")
print(f"monthly_charges < 0    : "
      f"{(df_dirty['monthly_charges'] < 0).sum()} lignes")
print(f"age hors [18, 100]     : "
      f"{((df_dirty['age'] < 18) | (df_dirty['age'] > 100)).sum()} lignes")
print(f"tenure > 72            : "
      f"{(df_dirty['tenure_months'] > 72).sum()} lignes")
print(f"total_charges = 0      : "
      f"{(df_dirty['total_charges'] == 0).sum()} lignes")
print()


# ────────────────────────────────────────────────────────
#  3. BRUIT DANS LES DONNÉES CATÉGORIELLES
#  Raison : fautes de frappe, majuscules incohérentes,
#           espaces, valeurs non standardisées
# ────────────────────────────────────────────────────────

# gender : variations de casse et fautes de frappe
noise_gender_idx = rng.choice(N, size=int(N * 0.03), replace=False)
noisy_gender_values = rng.choice(
    ['male', 'MALE', 'féminin', 'Femme', 'M', 'F',
     'female ', ' Male', 'FEMALE'],
    size=len(noise_gender_idx)
)
df_dirty.loc[noise_gender_idx, 'gender'] = noisy_gender_values

# contract : valeurs non standardisées
noise_contract_idx = rng.choice(N, size=int(N * 0.02), replace=False)
noisy_contract_values = rng.choice(
    ['month-to-month', 'MONTH TO MONTH', 'mensuel',
     'one year ', 'OneYear', '2 years', 'two-year'],
    size=len(noise_contract_idx)
)
df_dirty.loc[noise_contract_idx, 'contract'] = noisy_contract_values

# payment_method : espaces et casse
noise_pay_idx = rng.choice(N, size=int(N * 0.02), replace=False)
noisy_pay_values = rng.choice(
    ['electronic check', 'CREDIT CARD', 'bank_transfer',
     'mailed check ', 'CreditCard', 'Bank Transfer'],
    size=len(noise_pay_idx)
)
df_dirty.loc[noise_pay_idx, 'payment_method'] = noisy_pay_values

print("── Bruit catégoriel injecté ──")
print(f"Valeurs uniques gender          : "
      f"{df_dirty['gender'].nunique()} "
      f"(attendu : 2)")
print(f"Valeurs uniques contract        : "
      f"{df_dirty['contract'].nunique()} "
      f"(attendu : 3)")
print(f"Valeurs uniques payment_method  : "
      f"{df_dirty['payment_method'].nunique()} "
      f"(attendu : 4)")
print()


# ────────────────────────────────────────────────────────
#  4. INCOHÉRENCES LOGIQUES
#  Raison : jointures de tables mal faites, bugs de pipeline
# ────────────────────────────────────────────────────────

# Clients sans internet mais avec online_security = 'Yes'
incoherence_idx = rng.choice(
    df_dirty[df_dirty['internet_service'] == 'No'].index,
    size=80, replace=False
)
df_dirty.loc[incoherence_idx, 'online_security'] = 'Yes'

# has_dependents = 1 mais has_partner = 0 ET senior_citizen = 1
# (peu probable mais pas impossible — à documenter dans l'EDA)
senior_no_partner = df_dirty[
    (df_dirty['senior_citizen'] == 1) &
    (df_dirty['has_partner'] == 0)
].index
incoherence_idx2 = rng.choice(senior_no_partner,
                               size=min(50, len(senior_no_partner)),
                               replace=False)
df_dirty.loc[incoherence_idx2, 'has_dependents'] = 1

# Duplicates : 0.8% de doublons partiels (même client, deux entrées)
n_dupes = int(N * 0.008)
dupe_rows = df_dirty.sample(n=n_dupes, random_state=7).copy()
# Légèrement modifiés pour simuler une double saisie
dupe_rows['customer_id'] = [f'SYN-DUP-{i:04d}' for i in range(n_dupes)]
dupe_rows['monthly_charges'] = dupe_rows['monthly_charges'] * rng.uniform(
    0.98, 1.02, n_dupes
)
df_dirty = pd.concat([df_dirty, dupe_rows], ignore_index=True)

print("── Incohérences logiques injectées ──")
print(f"Sans internet mais security=Yes : "
      f"{((df_dirty['internet_service']=='No') & (df_dirty['online_security']=='Yes')).sum()}")
print(f"Doublons partiels ajoutés       : {n_dupes}")
print(f"Shape final du dataset dirty    : {df_dirty.shape}")
print()


# ════════════════════════════════════════════════════════
#  RAPPORT FINAL DE QUALITÉ
# ════════════════════════════════════════════════════════

print("═" * 50)
print("  RAPPORT QUALITÉ — DATASET BRUT")
print("═" * 50)
print(f"\nDimensions         : {df_dirty.shape}")
print(f"Churn rate         : {df_dirty['churn'].mean():.1%}")
print(f"\nValeurs manquantes :")
for col, count in df_dirty.isnull().sum().items():
    if count > 0:
        pct = count / len(df_dirty) * 100
        print(f"  {col:<22} {count:>5} ({pct:.1f}%)")

print(f"\nOutliers numériques :")
print(f"  monthly_charges > 300 : "
      f"{(df_dirty['monthly_charges'].dropna() > 300).sum()}")
print(f"  monthly_charges < 0   : "
      f"{(df_dirty['monthly_charges'].dropna() < 0).sum()}")
print(f"  age hors [18–100]     : "
      f"{((df_dirty['age'].dropna() < 18) | (df_dirty['age'].dropna() > 100)).sum()}")
print(f"  tenure > 72           : "
      f"{(df_dirty['tenure_months'].dropna() > 72).sum()}")

print(f"\nBruit catégoriel :")
print(f"  gender unique values    : {df_dirty['gender'].nunique()}")
print(f"  contract unique values  : {df_dirty['contract'].nunique()}")
print(f"  payment unique values   : {df_dirty['payment_method'].nunique()}")

print(f"\nDoublons partiels : {n_dupes}")
print(f"\nTotal problèmes à corriger en data cleaning : 4 catégories")
print("═" * 50)


# ════════════════════════════════════════════════════════
#  SAUVEGARDE
# ════════════════════════════════════════════════════════

# Dataset propre — référence de vérité terrain
df.to_parquet('data/raw/telecom_churn_CLEAN_REFERENCE.parquet',
              index=False)

# Dataset sale — c'est lui que tu utilises pour le projet
df_dirty.to_csv('data/raw/telecom_churn_raw.csv', index=False)
df_dirty.to_parquet('data/raw/telecom_churn_raw.parquet', index=False)

print("\nFichiers sauvegardés :")
print("  data/raw/telecom_churn_raw.csv           ← utilise celui-ci")
print("  data/raw/telecom_churn_raw.parquet       ← version optimisée")
print("  data/raw/telecom_churn_CLEAN_REFERENCE   ← vérité terrain")