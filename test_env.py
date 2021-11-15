import torch
wbn = torch.randn(64,64)
wconv = torch.randn(64,108)
w = torch.randn(64,12,3,3)
w.copy_(torch.mm(wbn, wconv).view(w.size()))
