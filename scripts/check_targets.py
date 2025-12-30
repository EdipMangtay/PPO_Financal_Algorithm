import pandas as pd
import numpy as np

df = pd.read_parquet('artifacts/20251230_005546/4h/preds_test.parquet')

print("="*60)
print("TARGET (TRUE LOG RETURNS) ANALYSIS")
print("="*60)
print(f"\nTotal samples: {len(df)}")
print(f"\nTarget Statistics:")
print(f"  Mean:     {df['target'].mean():.8f}")
print(f"  Std:      {df['target'].std():.8f}")
print(f"  Min:      {df['target'].min():.8f}")
print(f"  Max:      {df['target'].max():.8f}")
print(f"  Median:   {df['target'].median():.8f}")

print(f"\nDirection Distribution:")
print(f"  Positive (Up):   {(df['target'] > 0).sum():4d} ({(df['target'] > 0).sum() / len(df) * 100:.1f}%)")
print(f"  Negative (Down): {(df['target'] < 0).sum():4d} ({(df['target'] < 0).sum() / len(df) * 100:.1f}%)")
print(f"  Zero:            {(df['target'] == 0).sum():4d} ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")

print(f"\nFirst 20 targets:")
for i in range(min(20, len(df))):
    tgt = df['target'].iloc[i]
    pred = df['prediction'].iloc[i]
    direction = "UP" if tgt > 0 else "DOWN"
    print(f"  {i:3d}: Target={tgt:+.6f} ({direction})  Pred={pred:+.6f} (LONG)")

print(f"\n*** COMPARISON ***")
print(f"Targets with UP movement:   {(df['target'] > 0).sum()} ({(df['target'] > 0).sum() / len(df) * 100:.1f}%)")
print(f"Targets with DOWN movement: {(df['target'] < 0).sum()} ({(df['target'] < 0).sum() / len(df) * 100:.1f}%)")
print(f"")
print(f"Predictions for LONG:  {(df['prediction'] > 0).sum()} ({(df['prediction'] > 0).sum() / len(df) * 100:.1f}%)")
print(f"Predictions for SHORT: {(df['prediction'] < 0).sum()} ({(df['prediction'] < 0).sum() / len(df) * 100:.1f}%)")

if (df['target'] < 0).sum() > 0 and (df['prediction'] < 0).sum() == 0:
    print(f"\n*** CRITICAL BUG DETECTED ***")
    print(f"Targets have both UP and DOWN movements ({(df['target'] < 0).sum()} down)")
    print(f"But predictions are ALL POSITIVE (100% LONG bias)")
    print(f"Model is NOT predicting negative returns at all!")

