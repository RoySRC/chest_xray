import time
import torch
from YOLOv3Head import YOLOv3Head

bs = 8
model = YOLOv3Head([1024, 1024, 768], 15).cuda()
model.train()
v1 = torch.ones((bs, 1024, 13, 13), dtype=torch.float).cuda()
v2 = torch.ones((bs, 512, 26, 26), dtype=torch.float).cuda()
v3 = torch.ones((bs, 256, 52, 52), dtype=torch.float).cuda()
l, m, s = model(v1, v2, v3)
print(l.shape)
print(m.shape)
print(s.shape)
time.sleep(10)
