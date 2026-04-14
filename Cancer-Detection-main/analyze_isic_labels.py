import pandas as pd

try:
    # Analyzing train_concat.csv for BCC and SCC
    df_isic = pd.read_csv(r"c:\Users\Usama\Downloads\isic_cleaned\train_concat.csv")
    
    # We need to know what 'target' values correspond to BCC and SCC
    # Usually in the original ISIC 2019 GroundTruth:
    # MEL: 0, NV: 1, BCC: 2, AK: 3, BKL: 4, DF: 5, VASC: 6, SCC: 7, UNK: 8
    # However, 'target' in train_concat.csv was 0/1 (benign/malignant) in my previous check.
    # Let me check if there are other columns that specify the exact diagnosis.
    
    print("Columns in train_concat.csv:")
    print(df_isic.columns.tolist())
    print("\nFirst 5 rows:")
    print(df_isic.head())
    
    # Also check ISIC_2019_Training_Metadata.csv
    df_meta = pd.read_csv(r"c:\Users\Usama\Downloads\ISIC_2019_Training_Input\ISIC_2019_Training_Input\ISIC_2019_Training_Metadata.csv")
    print("\nColumns in ISIC_2019_Training_Metadata.csv:")
    print(df_meta.columns.tolist())
    
except Exception as e:
    print(f"Error: {e}")
