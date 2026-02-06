import pandas as pd

df = pd.read_csv("data/used_cars_clean.csv")

# take random 5000 rows
sample_df = df.sample(5000, random_state=42)

sample_df.to_csv("data/used_cars_sample.csv", index=False)

print("Sample dataset created!")
