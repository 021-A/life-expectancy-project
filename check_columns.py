import pandas as pd

# Baca CSV
df = pd.read_csv("data/cleaned/Life_expectancy_clean.csv")

# Print semua nama kolom
print("=== NAMA KOLOM DATASET ===")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. '{col}'")

print(f"\nTotal kolom: {len(df.columns)}")
print(f"Total baris: {len(df)}")