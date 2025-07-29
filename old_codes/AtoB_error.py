import pandas as pd
import numpy as np

def voltage_to_binary(v_in, v_ref, n_bits=8):
    max_digital = (2 ** n_bits) - 1
    v_in_clipped = np.clip(v_in, 0, v_ref)
    digital_code = np.floor((v_in_clipped / v_ref) * max_digital).astype(int)
    binary_code = [format(code, f'0{n_bits}b') for code in digital_code]
    return binary_code, digital_code

def analyze_binary_conversion_error(input_csv, output_csv='binary_conversion_error.csv', v_ref=675, n_bits=8):
    df = pd.read_csv(input_csv)

    if 'V' not in df.columns:
        raise ValueError("CSV must contain a 'V' column.")

    df['V'] = df['V']-df['V'][0]
    # Convert analog voltage back to binary
    bin_strs, digital_vals = voltage_to_binary(df['V'].values, v_ref=v_ref, n_bits=n_bits)

    # Add columns
    df['Recovered_Binary'] = bin_strs
    df['Recovered_Value'] = digital_vals

    # Expected ideal value is scaled based on `weighted_sum`
    max_weighted_sum = df['weighted_sum'].max()
    ideal_vals = np.floor((df['weighted_sum'] / max_weighted_sum) * ((2**n_bits) - 1)).astype(int)

    df['Ideal_Digital'] = ideal_vals
    df['Error'] = abs(df['Recovered_Value'] - df['Ideal_Digital'])

    # Save
    df.to_csv("resources/A_to_B/AtoBerror.csv", index=False)
    print(f"Saved binary conversion error analysis to {output_csv}")
    print(f"Mean Absolute Error: {(df['Error'].mean())/255}")
    print(f"Max Error: {df['Error'].max()/255}")

    return df

# Example usage:
analyze_binary_conversion_error('processed.csv')
