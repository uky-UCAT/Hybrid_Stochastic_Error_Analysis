import pandas as pd

def calculate_errors(file_path):
    # Read CSV into pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Ensure 'Error' column exists
    if 'Error' not in df.columns:
        raise ValueError("CSV file does not contain an 'Error' column.")
    
    # Calculate Mean Absolute Error
    mae = df['Error'].mean()
    
    # Calculate Maximum Error
    max_error = df['Error'].max()
    
    return mae, max_error

# Example usage
file_path = "resources/StochasticMULV2/8_bit_MUL.csv"
mae,max_error = calculate_errors(file_path)

print("Stochastic Multiplication Error:")
print(f"Mean Absolute Error: {mae}")
print(f"Maximum Error: {max_error}")


file_path = "resources/S_to_A/s_to_a_error.csv"  # replace with your file
mae,max_error = calculate_errors(file_path)
print("Stochastic to Analog Error:")
print(f"Mean Absolute Error: {mae}")
print(f"Maximum Error: {max_error}")

file_path = "resources/A_to_B/AtoBerror.csv"  # replace with your file
mae,max_error = calculate_errors(file_path)
print("Analog to Binary Error:")
print(f"Mean Absolute Error: {mae/255}")
print(f"Maximum Error: {max_error/255}")
