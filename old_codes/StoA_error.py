import pandas as pd

def stochastic_to_analog(decimal_values, weights=[16,1,1], bitwidths=[8,16,16], vref=675):
    if len(decimal_values) != 3:
        raise ValueError("Expected 4 decimal values (one per chunk).")
    
    weighted_sum = sum(d * w for d, w in zip(decimal_values, weights))
    max_sum = sum(b * w for b, w in zip(bitwidths, weights))
    
    voltage = (weighted_sum / max_sum) * vref
    return weighted_sum,voltage  

# Store all rows here
result_list = []

for d1 in range(0, 9):      # 0 to 8
    for d2 in range(0, 17): # 0 to 16
        for d3 in range(0, 17):
            decimal_values = [d1, d2, d3]
            w, voltage = stochastic_to_analog(decimal_values)
            
            # Append result as dictionary
            result_list.append({
                'concat': f"{d1} {d2} {d3}",
                'weighted_sum': w,
                'V': voltage
            })

# Create DataFrame and save
df = pd.DataFrame(result_list)
df2 = pd.read_csv('processed.csv')

# print(f"Saved {len(df)} combinations to stochastic_to_analog_results.csv")

merged = pd.merge(df, df2, on='concat', suffixes=('_expected', '_actual'))

# Calculate absolute error
merged['Error'] = (merged['V_expected'] - merged['V_actual']+df2['V'][0]).abs()

# print(merged[['concat', 'V_expected', 'V_actual', 'Error']].head())

merged[['concat', 'weighted_sum_actual', 'V_expected', 'V_actual', 'Error']].to_csv('resources/S_to_A/s_to_a_error.csv', index=False)
mae = merged['Error'].mean()
max_error = merged['Error'].max()

print(f"Mean Absolute Numeric Error: {mae}")
print(f"Maximum Numeric Error: {max_error}")