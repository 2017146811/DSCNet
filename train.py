import torch
from mmrotate.models import PKINet

# 初始化模型
model = PKINet(depths=[2, 2, 9, 2], embed_dim=96)
model_dict = model.state_dict()

# 加载预训练权重
pretrained = torch.load('./checkpoint/pkinet_s_pretrain.pth', map_location='cpu')
pretrained_dict = pretrained['state_dict'] if 'state_dict' in pretrained else pretrained

# 打印差异
print("====== Missing keys in pretrained ======")
print([k for k in model_dict if k not in pretrained_dict])

print("====== Extra keys in pretrained ======")
print([k for k in pretrained_dict if k not in model_dict])