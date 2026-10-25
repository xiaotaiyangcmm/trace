import torch.nn as nn
import torch.nn.functional as F
use_bn = False
if use_bn:
    from modelss.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction
else:
    from modelss.pointnet_util_no_bn import PointNetSetAbstractionMsg, PointNetSetAbstraction

class PointNetV2(nn.Module):
    def __init__(self): # remove the num_class premeter
        super(PointNetV2, self).__init__()
        normal_channel = False
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel # default: false
        self.sa1 = PointNetSetAbstractionMsg(256, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(256//16, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 320 + 3, [256, 512, 1024], True)
        self.linear = nn.Linear(640, 1024)
    def forward(self, xyz):
        # import pdb; pdb.set_trace()
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        xyz = xyz.permute(0, 2, 1)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        x = l2_points.view(B, -1, 640)
        x = self.linear(x)
        x = x.view(B, 1024, -1)
        # print(l2_xyz.shape)
        # print(x.shape)
        return  l2_xyz, x
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # x = l3_points.view(B, 1024) # change 1024 to 2048
        # print(l3_xyz.shape)
        # return  l3_xyz, x
        #return x, l3_points, (l2_xyz, l2_points)

if __name__ == "__main__":
    import torch
    encoder = PointNetV2()
    encoder.cuda()
    random_tensor = torch.rand(4, 256, 3).cuda()
    
    
    # 前向传播
    x, l3_points, (l2_xyz, l2_points) = encoder(random_tensor)
    # import pdb; pdb.set_trace()
    pass