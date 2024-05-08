import pandas as pd

# Load the CSV file
df = pd.read_csv('./test.csv')

# Get the first 10,000 records
first_10k = df.head(100)

# Save the first 10,000 records to a new CSV file
first_10k.to_csv('./test_100.csv', index=False)