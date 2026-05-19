# Small amount of code to open a validation dataframe and get the index of your specified stellar parameters. 

import pandas as pd
import numpy as np

# specify the file you want to open (relates to the order you have chosen)
filename = "SPIRou45_val.df"

df = pd.read_pickle(f"../data/validation_data/{filename}")
print(f"Opening file: {filename}" )

# specify the parameters you want 
teff = 3100
logg =  5.0
metal = -0.5
alpha = 0.0

# find a specific star with your parameters. 
# print the index you need

row_number = np.where(
    (df.teff == teff) &
    (df.logg == logg) &
    (df.metal == metal) &
    (df.alpha == alpha)
)[0]

print("Row number (Use as Index):", row_number)
print(df.iloc[row_number])