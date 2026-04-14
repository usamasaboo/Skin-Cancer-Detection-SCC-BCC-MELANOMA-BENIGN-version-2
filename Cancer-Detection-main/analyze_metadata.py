import pandas as pd

try:
    df_isic = pd.read_csv(r"c:\Users\Usama\Downloads\isic_cleaned\train_concat.csv")
    
    def get_source(name):
        try:
            parts = name.split('_')
            if len(parts) < 2: return 'Other'
            num_str = parts[1]
            if not num_str.isdigit(): return 'Other'
            num = int(num_str)
            if 0 <= num <= 11600: return 'MSK'
            elif 24306 <= num <= 34320: return 'HAM10000'
            elif 34321 <= num <= 73222: return 'BCN_20000'
            else: return 'ISIC_2020' 
        except: return 'Other'

    df_isic['source'] = df_isic['image_name'].apply(get_source)
    
    isic_2020_benign = df_isic[(df_isic['source'] == 'ISIC_2020') & (df_isic['target'] == 0)]
    print(f"ISIC 2020 Benign Count: {len(isic_2020_benign)}")
    print("Example ISIC 2020 Benign names:")
    print(isic_2020_benign['image_name'].head(10).tolist())
    
except Exception as e:
    print(f"Error: {e}")
