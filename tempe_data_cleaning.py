# tempe_v2.py  — robust party/nightlife filter incl. 2024–2025
import pandas as pd
import numpy as np
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

INFILE  = "Calls_for_Service_(NIBRS_Reporting_Period_2022_-_Present).csv"
OUT_NEI = "tempe_party_calls_cleaned_neighborhood.csv"
OUT_RAD = "tempe_party_calls_cleaned_radius2mi.csv"

# -------------------- Helpers --------------------
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two points."""
    # handle NaNs safely
    lat1 = np.asarray(lat1, dtype="float64")
    lon1 = np.asarray(lon1, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")
    r = 3958.7613  # Earth radius (miles)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return r*c

def contains_any(df, cols, pattern):
    """Row-wise OR across string columns using regex pattern."""
    m = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns:
            m = m | df[c].astype(str).str.contains(pattern, case=False, na=False, regex=True)
    return m

# -------------------- Load --------------------
df = pd.read_csv(INFILE, low_memory=False)
print("Raw shape:", df.shape)

# -------------------- Time fields --------------------
df["OccurrenceDatetime"] = pd.to_datetime(df["OccurrenceDatetime"], errors="coerce")
df["Year"]    = df["OccurrenceDatetime"].dt.year
df["Month"]   = df["OccurrenceDatetime"].dt.month_name()
df["MonthNum"]= df["OccurrenceDatetime"].dt.month
df["Hour"]    = df["OccurrenceDatetime"].dt.hour
df["Weekday"] = df["OccurrenceDatetime"].dt.day_name()
df["IsWeekend"] = df["Weekday"].isin(["Friday","Saturday","Sunday"])

# Keep only 2022+ (safety)
df = df[df["Year"] >= 2022]

# -------------------- Party/Nightlife regex (broad & future-proof) --------------------
# Covers label drift like "LOUD PARTY CALL", "DISORDERLY NOISE", "PUBLIC INTOX", etc.
party_regex = r"(noise|loud|music|party|disturb(ance|ing)?|disorderly|intox|liquor|trespass|harass|fight)"

text_cols = [
    "CallType",
    "FinalCaseTypeTrans",
    "InitialCaseTypeTrans",
    "FinalCaseType",
    "InitialCaseType",
    "CallCategory"
]
party_mask = contains_any(df, text_cols, party_regex)

df_party = df[party_mask].copy()
print("After party keyword filter:", df_party.shape)

# -------------------- Geography: Neighborhoods OR Radius --------------------
# A) Neighborhood focus (keeps future years even if names exist/shift slightly via CharacterArea)
asu_neighborhoods = {
    "Rio Salado/DT/ASU/NW Neighborhood",
    "University Park",
    "Maple Ash",
    "Hudson Manor",
    "Holdeman",
    "Alameda",
    "Clark Park",
    "Mills/Emerald",
    "Apache",
}

nei_mask = pd.Series(False, index=df_party.index)
if "NeighborhoodName" in df_party.columns:
    nei_mask = nei_mask | df_party["NeighborhoodName"].isin(asu_neighborhoods)
if "CharacterArea" in df_party.columns:
    # also accept CharacterArea containing "ASU" or "Mill" corridor
    nei_mask = nei_mask | df_party["CharacterArea"].astype(str).str.contains(r"(ASU|Mill|University Park|Maple Ash)", case=False, na=False)

df_party_nei = df_party[nei_mask].copy()
print("Neighborhood-based shape:", df_party_nei.shape)

# B) Radius focus (2 miles from ASU Tempe Campus core)
ASU_LAT, ASU_LON = 33.424, -111.928  # near Palm Walk/Memorial Union
if {"Latitude","Longitude"}.issubset(df_party.columns):
    dist_mi = haversine(df_party["Latitude"], df_party["Longitude"], ASU_LAT, ASU_LON)
    df_party_rad = df_party[dist_mi <= 2.0].copy()
else:
    df_party_rad = df_party.iloc[0:0].copy()  # empty if no coords
print("Radius (<=2 mi) shape:", df_party_rad.shape)

# -------------------- Select output columns --------------------
keep_cols = [
    "PrimaryKey","OccurrenceDatetime","Year","Month","MonthNum","Hour","Weekday","IsWeekend",
    "OccurrenceYear","OccurrenceMonth","OccurrenceHour","OccurrenceWeekday",
    "NeighborhoodName","CharacterArea","PlaceName","ObfuscatedAddress",
    "Latitude","Longitude","PostalCode","CensusTractID",
    "CallType","CallCategory","CallReceivedType","CallReceivedGroup",
    "InitialCaseType","FinalCaseType","InitialCaseTypeTrans","FinalCaseTypeTrans",
    "Priority","CaseStatus","ClearedBy","ClearedByTrans"
]
keep_cols = [c for c in keep_cols if c in df_party.columns]

df_party_nei = df_party_nei[keep_cols].copy()
df_party_rad = df_party_rad[keep_cols].copy()

# -------------------- Sanity: year counts --------------------
def year_counts(tag, d):
    if d.empty:
        print(f"{tag} year counts: <empty>")
    else:
        print(f"{tag} year counts:")
        print(d["Year"].value_counts().sort_index())

year_counts("Neighborhood", df_party_nei)
year_counts("Radius2mi", df_party_rad)

# -------------------- Export --------------------
df_party_nei.to_csv(OUT_NEI, index=False)
df_party_rad.to_csv(OUT_RAD, index=False)
print(f"\n✅ Exported:\n - {Path(OUT_NEI).resolve()}\n - {Path(OUT_RAD).resolve()}")
