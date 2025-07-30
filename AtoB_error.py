import pandas as pd
import numpy as np

#return the binary and digital value of an input voltage
def voltage_to_binary(v_in, v_ref, n_bits):
    max_digital = (2 ** n_bits) - 1
    # clip between range if needed (not exactly required since we preprocess it but just to be safe)
    v_in_clipped = np.clip(v_in, 0, v_ref)
    # ADC formula: vin/vref * full_value
    digital_code = np.floor((v_in_clipped / v_ref) * max_digital).astype(int)
    #convert to binary
    binary_code = [format(code, f'0{n_bits}b') for code in digital_code]
    return binary_code, digital_code

def analyze_binary_conversion_error(input_csv, output_csv='resources/A_to_B/AtoBerror.csv', n_bits=8, accumulations=16):
    df = pd.read_csv(input_csv)

    if 'V' not in df.columns:
        raise ValueError("CSV must contain a 'V' column.")

    #remove initial bias value
    df['V'] = df['V']-df['V'][0]

    #update v_ref to maximum saturated value as it can be lower than ideal maximum
    # h/w scaling required
    v_ref = df['V'].max()
    
    # Convert analog voltage back to binary
    bin_strs, digital_vals = voltage_to_binary(df['V'].values, v_ref, n_bits=n_bits)

    # Add columns for the binary and digital value of spice readings
    df['Recovered_Binary'] = bin_strs
    df['Recovered_Value'] = digital_vals

    # weighted sum/maximum gives a scale. Multiply by accumulations to denote multiple accumulated data but it cancels out
    # max of weighted sum is 128 for STAR but because of scaling, it will be 160 for PROSTAR
    df['V_ideal'] = (df['weighted_sum'] * accumulations / (df['weighted_sum'].max() * accumulations)) * v_ref
    
    max_digital = (2 ** n_bits) - 1

    # get ideal digital based on the ideal analog value
    df['Ideal_Digital'] = np.floor((df['V_ideal'] / v_ref) * max_digital).astype(int)

    #calculate error in digital form
    df['Error'] = abs(df['Recovered_Value'] - df['Ideal_Digital'])

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved binary conversion error analysis to {output_csv}")
    print(f"Mean Absolute Error: {(df['Error'].mean())/max_digital}")
    print(f"Max Error: {df['Error'].max()/max_digital}")

    return df

# Usage
analyze_binary_conversion_error('processed.csv')
