import pandas as pd
import numpy as np

# Load predictions
df = pd.read_parquet('artifacts/20251230_005546/4h/preds_test.parquet')

print("="*60)
print("4H MODEL PREDICTIONS ANALYSIS")
print("="*60)
print(f"\nTotal predictions: {len(df)}")
print(f"\nPrediction Statistics:")
print(f"  Mean:     {df['prediction'].mean():.8f}")
print(f"  Std:      {df['prediction'].std():.8f}")
print(f"  Min:      {df['prediction'].min():.8f}")
print(f"  Max:      {df['prediction'].max():.8f}")
print(f"  Median:   {df['prediction'].median():.8f}")

print(f"\nSignal Distribution:")
print(f"  Positive (Long):  {(df['prediction'] > 0).sum():4d} ({(df['prediction'] > 0).sum() / len(df) * 100:.1f}%)")
print(f"  Negative (Short): {(df['prediction'] < 0).sum():4d} ({(df['prediction'] < 0).sum() / len(df) * 100:.1f}%)")
print(f"  Zero (Neutral):   {(df['prediction'] == 0).sum():4d} ({(df['prediction'] == 0).sum() / len(df) * 100:.1f}%)")

print(f"\nWith threshold 0.0001:")
print(f"  Long signals:     {(df['prediction'] > 0.0001).sum():4d}")
print(f"  Short signals:    {(df['prediction'] < -0.0001).sum():4d}")
print(f"  Neutral:          {((df['prediction'] >= -0.0001) & (df['prediction'] <= 0.0001)).sum():4d}")

print(f"\nWith threshold 0.001:")
print(f"  Long signals:     {(df['prediction'] > 0.001).sum():4d}")
print(f"  Short signals:    {(df['prediction'] < -0.001).sum():4d}")
print(f"  Neutral:          {((df['prediction'] >= -0.001) & (df['prediction'] <= 0.001)).sum():4d}")

print(f"\nWith threshold 0.002:")
print(f"  Long signals:     {(df['prediction'] > 0.002).sum():4d}")
print(f"  Short signals:    {(df['prediction'] < -0.002).sum():4d}")
print(f"  Neutral:          {((df['prediction'] >= -0.002) & (df['prediction'] <= 0.002)).sum():4d}")

print(f"\nFirst 30 predictions:")
for i in range(min(30, len(df))):
    pred = df['prediction'].iloc[i]
    signal = "LONG" if pred > 0.0001 else ("SHORT" if pred < -0.0001 else "NEUTRAL")
    print(f"  {i:3d}: {pred:+.8f} -> {signal}")

print(f"\nLast 10 predictions:")
for i in range(max(0, len(df)-10), len(df)):
    pred = df['prediction'].iloc[i]
    signal = "LONG" if pred > 0.0001 else ("SHORT" if pred < -0.0001 else "NEUTRAL")
    print(f"  {i:3d}: {pred:+.8f} -> {signal}")

# Check for any negative predictions
neg_preds = df[df['prediction'] < 0]
if len(neg_preds) > 0:
    print(f"\n*** FOUND {len(neg_preds)} NEGATIVE PREDICTIONS ***")
    print("Sample negative predictions:")
    print(neg_preds.head(10))
else:
    print(f"\n*** WARNING: NO NEGATIVE PREDICTIONS! ***")
    print("*** MODEL HAS LONG BIAS - ALL PREDICTIONS POSITIVE ***")

