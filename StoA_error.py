import pandas as pd

df = pd.read_csv('processed.csv')

df['V_ideal'] = df['weighted_sum']/ df['weighted_sum'].max() * df['V'].max()

df['Error'] = (df['V']-df['V_ideal']-df['V'][0]).abs()


df[['weighted_sum', 'V', 'V_ideal', 'Error']].to_csv('resources/S_to_A/s_to_a_error.csv', index=False)
mae = df['Error'].mean()
max_error = df['Error'].max()

print(f"Mean Absolute Numeric Error: {mae}")
print(f"Maximum Numeric Error: {max_error}")