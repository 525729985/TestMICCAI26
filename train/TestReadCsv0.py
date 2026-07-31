import pandas as pd

file = "F:/MICCAI/pred/Train-Labeled250_250/size.csv"
# file = "F:/MICCAI/pred/Train-Labeled500_250/size.csv"

names = []
data = pd.read_csv(file)
data = data.iloc[: -1]
for idx, row in data.iterrows():
    name = row["name"]
    error_num = sum(1 for i in range(1, 65) for t in ["d", "h", "w"] if row[f"size_{i}_{t}"] > 100)
    error_tooth_num = sum(1 for i in range(1, 33) for t in ["d", "h", "w"] if row[f"tooth_size_{i}_{t}"] > 100)
    error_block_w_num = sum(1 for i in range(1, 5) if row[f"block_size_{i}_w"] > 200)
    error_block_z_num = sum(1 for i in range(1, 5) if row[f"block_size_{i}_d"] > 150)
    error_block_num = error_block_w_num + error_block_z_num
    print(f"{name}: error_num={error_num}, error_tooth_num={error_tooth_num}, error_block_num={error_block_num}")
    if error_num == 0 and error_tooth_num == 0 and error_block_num == 0:
        names.append(name)
print(names)
print(len(names))