import pandas as pd
import numpy as np

#return the binary and digital value of an input voltage
def voltage_to_binary(v_in, v_ref, n_bits=8):
    max_digital = (2 ** n_bits) - 1
    v_in_clipped = np.clip(v_in, 0, v_ref)
    digital_code = np.floor((v_in_clipped / v_ref) * max_digital).astype(int)
    binary_code = [format(code, f'0{n_bits}b') for code in digital_code]
    return binary_code, digital_code

def analyze_binary_conversion_error(input_csv, output_csv='resources/A_to_B/AtoBerror.csv', v_ref=675, n_bits=8, total_bitlines=40, accumulations=30):
    df = pd.read_csv(input_csv)

    if 'V' not in df.columns:
        raise ValueError("CSV must contain a 'V' column.")

    #remove initial bias value
    df['V'] = df['V']-df['V'][0]
    #update v_ref to maximum saturated value as it can be lower than ideal maximum
    v_ref = df['V'].max()
    
    # Convert analog voltage back to binary
    bin_strs, digital_vals = voltage_to_binary(df['V'].values, v_ref, n_bits=n_bits)

    # Add columns
    df['Recovered_Binary'] = bin_strs
    df['Recovered_Value'] = digital_vals

    df['V_ideal'] = (df['weighted_sum'] * accumulations / (total_bitlines * accumulations)) * v_ref
    
    max_digital = (2 ** n_bits) - 1
    df['Ideal_Digital'] = np.floor((df['V_ideal'] / v_ref) * max_digital).astype(int)

    df['Error'] = abs(df['Recovered_Value'] - df['Ideal_Digital'])

    # Save
    df.to_csv(output_csv, index=False)
    print(f"Saved binary conversion error analysis to {output_csv}")
    print(f"Mean Absolute Error: {(df['Error'].mean())/max_digital}")
    print(f"Max Error: {df['Error'].max()/max_digital}")

    return df

# Usage
analyze_binary_conversion_error('processed.csv')
