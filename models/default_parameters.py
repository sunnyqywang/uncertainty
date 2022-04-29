import torch

device = torch.device("cpu")
lookback = 2
difference=True
predict_hzn = 1
time_size = 4
max_lookback = 6
adj_type = ["func","euc","con","net"]
train_extent='downtown'
