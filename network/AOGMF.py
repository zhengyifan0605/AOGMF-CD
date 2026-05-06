#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class FEC(nn.Module):
    #convolutional block
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(FEC, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes,out_planes, kernel_size = kernel_size, padding=padding,dilation=dilation,bias=False)
        # self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=3, dilation=3, bias=True)  #加入dilation和padding！！
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        # print('x的大小:',x.shape)
        x = self.conv1(x)
        # print('x的大小2:', x.shape)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class ChangeGuideModule(nn.Module):
    def __init__(self, in_dim):
        super(ChangeGuideModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()

        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)

        guiding_map = F.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out
    
class ChangeGuideModule_baseline(nn.Module):
    def __init__(self, in_dim):
        super(ChangeGuideModule_baseline, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()

        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)

        guiding_map = F.sigmoid(guiding_map0)

        out = x * (1 + guiding_map)

        out = self.gamma * out + x

        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)

    
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
    
class aggregation_init(nn.Module):

    def __init__(self, channel):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class aggregation_init_baseline(nn.Module):

    def __init__(self):
        super(aggregation_init_baseline, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(96,1,3,1,1)


    def forward(self, x1, x2, x3):
        x1_1 = self.upsample(self.upsample(x1)) 
        x2_1 = self.upsample(x2)
        x3_1 = x3

        x = torch.cat((x1_1,x2_1,x3_1),dim=1)
        x = self.conv(x)

        return x




#增加decoder解码器，不concat多尺度特征生成guide map,也concat多尺度特征生成最后输出
class AOGMFBase(nn.Module):
    def __init__(self,):
        super(AOGMFBase, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

       


        self.conv_reduce_1 = BasicConv2d(128*2,128,3,1,1)
        self.conv_reduce_2 = BasicConv2d(256*2,256,3,1,1)
        self.conv_reduce_3 = BasicConv2d(512*2,512,3,1,1)
        self.conv_reduce_4 = BasicConv2d(512*2,512,3,1,1)
        #self.conv_reduce_4 = BasicConv2d(1024*2,1024,3,1,1)



        #self.up_layer4 = BasicConv2d(512,512,3,1,1)
        self.up_layer4 = BasicConv2d(1024,1024,3,1,1)
        self.up_layer3 = BasicConv2d(512,512,3,1,1)
        self.up_layer2 = BasicConv2d(256,256,3,1,1)

        self.atten_A_channel_1=ChannelAttention(128)
        self.atten_A_channel_2=ChannelAttention(256)
        self.atten_A_channel_3=ChannelAttention(512)
        self.atten_A_channel_4=ChannelAttention(512)

        self.atten_A_spatial_1=SpatialAttention()
        self.atten_A_spatial_2=SpatialAttention()
        self.atten_A_spatial_3=SpatialAttention()
        self.atten_A_spatial_4=SpatialAttention()

        self.atten_B_channel_1=ChannelAttention(128)
        self.atten_B_channel_2=ChannelAttention(256)
        self.atten_B_channel_3=ChannelAttention(512)
        self.atten_B_channel_4=ChannelAttention(512)

        self.atten_B_spatial_1=SpatialAttention()
        self.atten_B_spatial_2=SpatialAttention()
        self.atten_B_spatial_3=SpatialAttention()
        self.atten_B_spatial_4=SpatialAttention()

        #self.decoder_single_layer = nn.Sequential(BasicConv2d(512,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        #self.decoder_single_layer = nn.Sequential(FEC(512,256,3,1,1),FEC(512,256,3,1,1),FEC(256,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        self.decoder = nn.Sequential(FEC(1408,512,3,1,1),FEC(512,256,3,1,1),FEC(256,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        
        self.rfb2 = GCM(256, 32)
        self.rfb3 = GCM(512, 32)
        self.rfb4 = GCM(512, 32)
        self.agg1 = aggregation_init(32)



        #self.decoder_final = nn.Sequential(BasicConv2d(128,64,3,1,1),nn.Conv2d(64,1,1))
        self.decoder_final = nn.Sequential(FEC(128,64,3,1,1),nn.Conv2d(64,1,1))

       


        self.cgm_2 = ChangeGuideModule(256)
        self.cgm_3 = ChangeGuideModule(512)
        self.cgm_4 = ChangeGuideModule(512)
      

        #相比v2 额外的模块
        self.upsample2x=nn.UpsamplingBilinear2d(scale_factor=2)
        # self.decoder_module4 = BasicConv2d(1024,512,3,1,1)
        # self.decoder_module3 = BasicConv2d(768,256,3,1,1)
        # self.decoder_module2 = BasicConv2d(384,128,3,1,1)

        self.decoder_module4 = FEC(1024,512,3,1,1)
        self.decoder_module3 = FEC(768,256,3,1,1)
        self.decoder_module2 = FEC(384,128,3,1,1)

    # def forward(self, A,B=None):
    #     if B == None:
    #         B = A
    def forward(self,A,B):
        # print(self)
        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        # layer1 = torch.cat((layer1_B,layer1_A),dim=1)

        # layer2 = torch.cat((layer2_B,layer2_A),dim=1)

        # layer3 = torch.cat((layer3_B,layer3_A),dim=1)

        # layer4 = torch.cat((layer4_B,layer4_A),dim=1)

        # layer1 = self.conv_reduce_1(layer1)
        # layer2 = self.conv_reduce_2(layer2)
        # layer3 = self.conv_reduce_3(layer3)
        # layer4 = self.conv_reduce_4(layer4)

        #layer1
        layer1_A = layer1_A.mul(self.atten_A_channel_1(layer1_A))
        layer1_A = layer1_A.mul(self.atten_A_spatial_1(layer1_A))

        layer1_B = layer1_B.mul(self.atten_B_channel_1(layer1_B))
        layer1_B = layer1_B.mul(self.atten_B_spatial_1(layer1_B))
        layer1 = layer1_A + layer1_B
        
        #layer2
        layer2_A = layer2_A.mul(self.atten_A_channel_2(layer2_A))
        layer2_A = layer2_A.mul(self.atten_A_spatial_2(layer2_A))

        layer2_B = layer2_B.mul(self.atten_B_channel_2(layer2_B))
        layer2_B = layer2_B.mul(self.atten_B_spatial_2(layer2_B))
        layer2 = layer2_A + layer2_B

        #layer3
        layer3_A = layer3_A.mul(self.atten_A_channel_3(layer3_A))
        layer3_A = layer3_A.mul(self.atten_A_spatial_3(layer3_A))

        layer3_B = layer3_B.mul(self.atten_B_channel_3(layer3_B))
        layer3_B = layer3_B.mul(self.atten_B_spatial_3(layer3_B))
        layer3 = layer3_A + layer3_B

        #layer4
        layer4_A = layer4_A.mul(self.atten_A_channel_4(layer4_A))
        layer4_A = layer4_A.mul(self.atten_A_spatial_4(layer4_A))

        layer4_B = layer4_B.mul(self.atten_B_channel_4(layer4_B))
        layer4_B = layer4_B.mul(self.atten_B_spatial_4(layer4_B))
        layer4 = layer4_A + layer4_B


        

        layer2_1 = self.rfb2(layer2)
        layer3_1 = self.rfb3(layer3)
        layer4_1 = self.rfb4(layer4)
        change_map = self.agg1(layer4_1,layer3_1,layer2_1)
        # change_map = F.interpolate(feature_fuse, layer1.size()[2:], mode='bilinear', align_corners=True)

        # layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        # feature_fuse=layer4_1 #需要注释！


        # # layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        # # layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        # # layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)
        # # feature_fuse = torch.cat((layer1,layer2_1,layer3_1,layer4_1),dim=1)

        # change_map = self.decoder_single_layer(feature_fuse) #需要注释！

 


        # ---------------注释这两句------------------------
        # if not self.training:
        #     feature_fuse = layer4_1
        #     feature_fuse = feature_fuse.cpu().detach().numpy()
        #     for num in range(0, 511):
        #         display = feature_fuse[0, num, :, :]  # 第几张影像，第几层特征0-511
        #         plt.figure()
        #         plt.imshow(display)  # [B, C, H,W]
        #         plt.savefig('./test_result/feature_fuse-v6-LEVIR1419/' + 'v6-fuse-' + str(num) + '.png')
        # # change_map = self.decoder(torch.cat((layer1, layer2_1, layer3_1, layer4_1), dim=1))
        # change_map = self.decoder(feature_fuse)
        # ---------------注释这两句------------------------

        layer4 = self.cgm_4(layer4, change_map)
        feature4=self.decoder_module4(torch.cat([self.upsample2x(layer4),layer3],1))

        layer3 = self.cgm_3(feature4, change_map)
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(layer3),layer2],1))

       
        layer2 = self.cgm_2(feature3, change_map)
        layer1 = self.decoder_module2(torch.cat([self.upsample2x(layer2), layer1], 1))

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        final_map = self.decoder_final(layer1)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return change_map, final_map

class AOGMF(nn.Module):
    def __init__(self,):
        super(AOGMF, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.conv_reduce_1 = BasicConv2d(128*2,128,3,1,1)
        self.conv_reduce_2 = BasicConv2d(256*2,256,3,1,1)
        self.conv_reduce_3 = BasicConv2d(512*2,512,3,1,1)
        self.conv_reduce_4 = BasicConv2d(512*2,512,3,1,1)

        self.up_layer4 = BasicConv2d(512,512,3,1,1)
        self.up_layer3 = BasicConv2d(512,512,3,1,1)
        self.up_layer2 = BasicConv2d(256,256,3,1,1)

        self.atten_A_channel_1=ChannelAttention(128)
        self.atten_A_channel_2=ChannelAttention(256)
        self.atten_A_channel_3=ChannelAttention(512)
        self.atten_A_channel_4=ChannelAttention(512)

        self.atten_A_spatial_1=SpatialAttention()
        self.atten_A_spatial_2=SpatialAttention()
        self.atten_A_spatial_3=SpatialAttention()
        self.atten_A_spatial_4=SpatialAttention()

        self.atten_B_channel_1=ChannelAttention(128)
        self.atten_B_channel_2=ChannelAttention(256)
        self.atten_B_channel_3=ChannelAttention(512)
        self.atten_B_channel_4=ChannelAttention(512)

        self.atten_B_spatial_1=SpatialAttention()
        self.atten_B_spatial_2=SpatialAttention()
        self.atten_B_spatial_3=SpatialAttention()
        self.atten_B_spatial_4=SpatialAttention()

        self.rfb2_baseline = FEC(256,32,3,1,1)
        self.rfb3_baseline = FEC(512,32,3,1,1)
        self.rfb4_baseline = FEC(512,32,3,1,1)
        self.agg1_baseline = aggregation_init_baseline()

        self.rfb2 = GCM(256, 32)
        self.rfb3 = GCM(512, 32)
        self.rfb4 = GCM(512, 32)
        self.agg1 = aggregation_init(32)


        self.decoder = nn.Sequential(BasicConv2d(512,64,3,1,1),nn.Conv2d(64,1,3,1,1))

        self.decoder_final = nn.Sequential(FEC(128,64,3,1,1),nn.Conv2d(64,1,1))

        self.cgm_2_baseline = ChangeGuideModule_baseline(256)
        self.cgm_3_baseline = ChangeGuideModule_baseline(512)
        self.cgm_4_baseline = ChangeGuideModule_baseline(512)


        self.cgm_2 = ChangeGuideModule(256)
        self.cgm_3 = ChangeGuideModule(512)
        self.cgm_4 = ChangeGuideModule(512)

        #相比v2 额外的模块
        self.upsample2x=nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = FEC(1024,512,3,1,1)
        self.decoder_module3 = FEC(768,256,3,1,1)
        self.decoder_module2 = FEC(384,128,3,1,1)


    # def forward(self, A,B=None):
    #     if B == None:
    #         B = A
    def forward(self,A,B):

        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        # layer1 = torch.cat((layer1_B,layer1_A),dim=1)

        # layer2 = torch.cat((layer2_B,layer2_A),dim=1)

        # layer3 = torch.cat((layer3_B,layer3_A),dim=1)

        # layer4 = torch.cat((layer4_B,layer4_A),dim=1)

        # layer1 = self.conv_reduce_1(layer1)
        # layer2 = self.conv_reduce_2(layer2)
        # layer3 = self.conv_reduce_3(layer3)
        # layer4 = self.conv_reduce_4(layer4)


        # ---------注意力机制的消融实验-----------------


        # layer1 = layer1_A + layer1_B
        # layer2 = layer2_A + layer2_B
        # layer3 = layer3_A + layer3_B
        # layer4 = layer4_A + layer4_B

        layer1_A = layer1_A.mul(self.atten_A_channel_1(layer1_A))
        layer1_A = layer1_A.mul(self.atten_A_spatial_1(layer1_A))

        layer1_B = layer1_B.mul(self.atten_B_channel_1(layer1_B))
        layer1_B = layer1_B.mul(self.atten_B_spatial_1(layer1_B))
        layer1 = layer1_A + layer1_B
        
        #layer2
        layer2_A = layer2_A.mul(self.atten_A_channel_2(layer2_A))
        layer2_A = layer2_A.mul(self.atten_A_spatial_2(layer2_A))

        layer2_B = layer2_B.mul(self.atten_B_channel_2(layer2_B))
        layer2_B = layer2_B.mul(self.atten_B_spatial_2(layer2_B))
        layer2 = layer2_A + layer2_B

        #layer3
        layer3_A = layer3_A.mul(self.atten_A_channel_3(layer3_A))
        layer3_A = layer3_A.mul(self.atten_A_spatial_3(layer3_A))

        layer3_B = layer3_B.mul(self.atten_B_channel_3(layer3_B))
        layer3_B = layer3_B.mul(self.atten_B_spatial_3(layer3_B))
        layer3 = layer3_A + layer3_B

        #layer4
        layer4_A = layer4_A.mul(self.atten_A_channel_4(layer4_A))
        layer4_A = layer4_A.mul(self.atten_A_spatial_4(layer4_A))

        layer4_B = layer4_B.mul(self.atten_B_channel_4(layer4_B))
        layer4_B = layer4_B.mul(self.atten_B_spatial_4(layer4_B))
        layer4 = layer4_A + layer4_B

        # ---------注意力机制的消融实验-----------------

        # --------空洞卷积的消融实验-------------------
        # layer2_1 = self.rfb2_baseline(layer2)
        # layer3_1 = self.rfb3_baseline(layer3)
        # layer4_1 = self.rfb4_baseline(layer4)

        layer2_1 = self.rfb2(layer2)
        layer3_1 = self.rfb3(layer3)
        layer4_1 = self.rfb4(layer4)


        # --------空洞卷积的消融实验-------------------

        # --------逐步融合模块的消融实验-------------------
        # change_map = self.agg1_baseline(layer4_1,layer3_1,layer2_1)
        change_map = self.agg1(layer4_1,layer3_1,layer2_1)
        # --------逐步融合模块的消融实验-------------------

        

        # layer4 = self.up_layer4(layer4)
        #
        # layer3 = self.up_layer3(layer3)
        #
        # layer2 = self.up_layer2(layer2)

        # layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        # feature_fuse=layer4_1 #需要注释！


        # # layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        # # layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        # # layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)
        # # feature_fuse = torch.cat((layer1,layer2_1,layer3_1,layer4_1),dim=1)

        # change_map = self.decoder(feature_fuse) #需要注释！

        # ---------------注释这两句------------------------
        # if not self.training:
        #     feature_fuse = layer4_1
        #     feature_fuse = feature_fuse.cpu().detach().numpy()
        #     for num in range(0, 511):
        #         display = feature_fuse[0, num, :, :]  # 第几张影像，第几层特征0-511
        #         plt.figure()
        #         plt.imshow(display)  # [B, C, H,W]
        #         plt.savefig('./test_result/feature_fuse-v6-LEVIR1419/' + 'v6-fuse-' + str(num) + '.png')
        # # change_map = self.decoder(torch.cat((layer1, layer2_1, layer3_1, layer4_1), dim=1))
        # change_map = self.decoder(feature_fuse)
        # ---------------注释这两句------------------------

        #--------------去掉cgm的消融实验-----------------
        # layer4 = self.cgm_4_baseline(layer4, change_map)
        # feature4=self.decoder_module4(torch.cat([self.upsample2x(layer4),layer3],1))
        
        # layer3 = self.cgm_3_baseline(feature4, change_map)
        # feature3 = self.decoder_module3(torch.cat([self.upsample2x(feature4),layer2],1))
        
        # layer2 = self.cgm_2_baseline(feature3, change_map)
        # layer1 = self.decoder_module2(torch.cat([self.upsample2x(layer2), layer1], 1))

        layer4 = self.cgm_4(layer4, change_map)
        feature4=self.decoder_module4(torch.cat([self.upsample2x(layer4),layer3],1))

        layer3 = self.cgm_3(feature4, change_map)
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(layer3),layer2],1))

       
        layer2 = self.cgm_2(feature3, change_map)
        layer1 = self.decoder_module2(torch.cat([self.upsample2x(layer2), layer1], 1))
         #--------------去掉cgm的消融实验-----------------

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        final_map = self.decoder_final(layer1)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return change_map, final_map

class HCGMNet(nn.Module):
    def __init__(self,):
        super(HCGMNet, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.conv_reduce_1 = BasicConv2d(128*2,128,3,1,1)
        self.conv_reduce_2 = BasicConv2d(256*2,256,3,1,1)
        self.conv_reduce_3 = BasicConv2d(512*2,512,3,1,1)
        self.conv_reduce_4 = BasicConv2d(512*2,512,3,1,1)

        self.up_layer4 = BasicConv2d(512,512,3,1,1)
        self.up_layer3 = BasicConv2d(512,512,3,1,1)
        self.up_layer2 = BasicConv2d(256,256,3,1,1)

        # self.decoder = nn.Sequential(BasicConv2d(1408,512,3,1,1),BasicConv2d(512,256,3,1,1),BasicConv2d(256,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        self.deocde = nn.Sequential(BasicConv2d(1408, 512, 3, 1, 1), BasicConv2d(512, 256, 3, 1, 1),BasicConv2d(256, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))

        # self.decoder_final = nn.Sequential(BasicConv2d(1408,512,3,1,1),BasicConv2d(512,256,3,1,1),BasicConv2d(256,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        self.deocde_final = nn.Sequential(BasicConv2d(1408, 512, 3, 1, 1), BasicConv2d(512, 256, 3, 1, 1),BasicConv2d(256, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))

        self.cgm_2 = ChangeGuideModule(256)
        self.cgm_3 = ChangeGuideModule(512)
        self.cgm_4 = ChangeGuideModule(512)

    # def forward(self, A,B=None):
    #     if B == None:
    #         B = A
    def forward(self,A,B):
        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        layer1 = torch.cat((layer1_B,layer1_A),dim=1)

        layer2 = torch.cat((layer2_B,layer2_A),dim=1)

        layer3 = torch.cat((layer3_B,layer3_A),dim=1)

        layer4 = torch.cat((layer4_B,layer4_A),dim=1)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        layer4 = self.up_layer4(layer4)

        layer3 = self.up_layer3(layer3)

        layer2 = self.up_layer2(layer2)

        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)

        feature_fuse = torch.cat((layer1,layer2_1,layer3_1,layer4_1),dim=1)
        change_map = self.deocde(feature_fuse)
        # ---------------注释这两句------------------------
        # if not self.training:
        #     feature_fuse = torch.cat((layer1,layer2_1,layer3_1,layer4_1), dim=1)
        #     feature_fuse = feature_fuse.cpu().detach().numpy()
        #     for num in range(0, 511):
        #         display = feature_fuse[0, num, :, :]  # 第几张影像，第几层特征0-511
        #         plt.figure()
        #         plt.imshow(display)  # [B, C, H,W]
        #         plt.savefig('./test_result/feature_fuse-v2/' + 'v2-fuse-' + str(num) + '.png')
        # change_map = self.decoder(torch.cat((layer1,layer2_1,layer3_1,layer4_1), dim=1))
        # ---------------注释这两句------------------------

        layer2 = self.cgm_2(layer2, change_map)
        layer3 = self.cgm_3(layer3, change_map)
        layer4 = self.cgm_4(layer4, change_map)

        layer4_2 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_2 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_2 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)

        new_feature_fuse = torch.cat((layer1,layer2_2,layer3_2,layer4_2),dim=1)

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        final_map = self.deocde_final(new_feature_fuse)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return change_map, final_map



if __name__=='__main__':
    #测试热图
    # net=AOGMFBase().cuda()
    # out=net(torch.rand((2,3,256,256)).cuda(),torch.rand((2,3,256,256)).cuda())

    #测试模型大小
    print('Hi~')
    input_size = 256
    model_restoration = HCGMNet()
    # model_restoration = AOGMFBase()
    from ptflops import get_model_complexity_info
    from torchstat import stat

    # input = torch.rand((3, input_size, input_size))
    # output = model_restoration(input)
    macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    stat(model_restoration, (3, input_size, input_size))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))