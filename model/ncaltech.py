import torch
import torch.nn as nn

BN_MOMENTUM = 0.1
from torchvision import models



class FPN_Multi_ResNet34_Moe(nn.Module):
    def __init__(self,num_classes):
        super(FPN_Multi_ResNet34_Moe,self).__init__()

        self.resnet = models.resnet34(pretrained=True)


        self.convx_1 = nn.Conv2d(64, 64, kernel_size=1)
        self.convx_2 = nn.Conv2d(128, 64, kernel_size=1)
        self.convx_3 = nn.Conv2d(256, 64, kernel_size=1)
        self.convx_4 = nn.Conv2d(512, 64, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.coarse_fine_fusion = nn.Sequential(nn.Conv2d(64*3, 64, kernel_size=3, stride=2, padding=1, bias=True),
                                                nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)
                                                )

        self.conv1_1 = nn.Sequential(nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                     nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),

                                     )
        self.relu = nn.ReLU(inplace=True)



        self.raw_decay = nn.Parameter(torch.randn(2))
        self.scale=100
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.resnet.fc = nn.Linear(64*4, num_classes)


    def forward(self,x):
        x_1=x[:,:2,:,:]
        x_2=x[:,2:4,:,:]
        x_3=x[:,4:,:,:]

        x_1conv = self.conv1_1(x_1)  # 112,112,64
        x_2conv = self.conv1_1(x_2)
        x_3conv = self.conv1_1(x_3)

        decay = torch.sigmoid(self.raw_decay)
        x_1_2conv = decay[0] * x_1conv + x_2conv
        x_1_2_3conv = decay[1] * x_1_2conv + x_3conv


        multi1_2_3 = torch.cat((x_1conv, x_1_2conv, x_1_2_3conv), dim=1)

        multi1_2_3_conv = self.coarse_fine_fusion(multi1_2_3)

        layer1=self.resnet.layer1(multi1_2_3_conv)
        layer2=self.resnet.layer2(layer1)
        layer3=self.resnet.layer3(layer2)
        layer4=self.resnet.layer4(layer3)



        layer1=self.convx_1(layer1)
        layer2 = self.convx_2(layer2) #28,28,256
        layer3 = self.convx_3(layer3) #14,14,256
        layer4=self.convx_4(layer4)


        x_4_up_conv = nn.functional.interpolate(layer4, [14, 14])
        x_4_3_conv=x_4_up_conv+layer3


        x_3_up_conv=nn.functional.interpolate(x_4_3_conv, [28,28])  #和2融合

        x_3_2_conv=x_3_up_conv+layer2
        x_3_2_1conv=nn.functional.interpolate(x_3_2_conv, [56,56])


        x_3_2_1conv=x_3_2_1conv+layer1


        x_4conv = nn.functional.interpolate(layer4, [56, 56])
        x_3conv=nn.functional.interpolate( x_4_3_conv, [56,56])
        x_3_2_conv=nn.functional.interpolate(x_3_2_conv, [56,56])


        multi1_2_3=torch.cat((x_4conv,x_3conv, x_3_2_conv,x_3_2_1conv), dim=1)

        avg = self.avgpool(multi1_2_3)

        avg=torch.flatten(avg,start_dim=1)

        fc = self.resnet.fc(avg)

        return fc


