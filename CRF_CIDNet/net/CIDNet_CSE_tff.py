import torch
import torch.nn as nn
try:
    from net.HVI_transform import RGB_HVI
    from net.transformer_utils import *
    from net.LCA import *
except:
    from HVI_transform import RGB_HVI
    from transformer_utils import *
    from LCA import *
from huggingface_hub import PyTorchModelHubMixin

class CSE_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(CSE_LCA, self).__init__()
        self.hv_lca = HV_LCA(dim, num_heads, bias)
        self.i_lca = I_LCA(dim, num_heads, bias)

    def forward(self, hv, i):
        hv_out = self.hv_lca(hv, i)
        i_out = self.i_lca(i, hv)
        return hv_out, i_out

class CSE_CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[ 36, 72, 144],
                 heads=[2, 4, 8],
                 norm=False
        ):
        super(CSE_CIDNet, self).__init__()
        
        
        [ch2, ch3, ch4] = channels
        [head2, head3, head4] = heads
        
        self.LCA_stage1_level1 = CSE_LCA(ch2, head2)
        self.LCA_stage1_level1_prime = CSE_LCA(ch2, head2)

        self.LCA_stage1_level2 = CSE_LCA(ch3, head3)
        self.LCA_stage1_level2_prime = CSE_LCA(ch3, head3)
        self.LCA_stage1_level2_second = CSE_LCA(ch3, head3)
        self.LCA_stage1_level2_third = CSE_LCA(ch3, head3)

        self.LCA_stage1_level3 = CSE_LCA(ch4, head4)
        self.LCA_stage1_level3_prime = CSE_LCA(ch4, head4)
        self.LCA_stage1_level3_second = CSE_LCA(ch4, head4)
        self.LCA_stage1_level3_third = CSE_LCA(ch4, head4)

        self.LCA_stage2_level1 = CSE_LCA(ch2, head2)
        self.LCA_stage2_level1_prime = CSE_LCA(ch2, head2)

        self.LCA_stage2_level2 = CSE_LCA(ch3, head3)
        self.LCA_stage2_level2_prime = CSE_LCA(ch3, head3)
        self.LCA_stage2_level2_second = CSE_LCA(ch3, head3)
        self.LCA_stage2_level2_third = CSE_LCA(ch3, head3)

        self.LCA_stage2_level3 = CSE_LCA(ch4, head4)
        self.LCA_stage2_level3_prime = CSE_LCA(ch4, head4)
        self.LCA_stage2_level3_second = CSE_LCA(ch4, head4)
        self.LCA_stage2_level3_third = CSE_LCA(ch4, head4)

        self.LCA_stage3_level1 = CSE_LCA(ch2, head2)
        self.LCA_stage3_level1_prime = CSE_LCA(ch2, head2)

        self.LCA_stage3_level2 = CSE_LCA(ch3, head3)
        self.LCA_stage3_level2_prime = CSE_LCA(ch3, head3)
        self.LCA_stage3_level2_second = CSE_LCA(ch3, head3)
        self.LCA_stage3_level2_third = CSE_LCA(ch3, head3)

        self.LCA_stage3_level3 = CSE_LCA(ch4, head4)
        self.LCA_stage3_level3_prime = CSE_LCA(ch4, head4)
        self.LCA_stage3_level3_second = CSE_LCA(ch4, head4)
        self.LCA_stage3_level3_third = CSE_LCA(ch4, head4)
        
        self.trans = RGB_HVI()

        # in proj

        self.hvi_proj_in_level1 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch2, 3, stride=1, padding=0,bias=False)
            )
        self.i_proj_in_level1 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch2, 3, stride=1, padding=0,bias=False)
            )

        self.hvi_proj_in_level2 =  NormDownsample(ch2, ch3, use_norm = norm)
        self.i_proj_in_level2 = NormDownsample(ch2, ch3, use_norm = norm)

        self.hvi_proj_in_level3 =  NormDownsample(ch3, ch4, use_norm = norm)
        self.i_proj_in_level3 = NormDownsample(ch3, ch4, use_norm = norm)

        # intermediary up/down sampling
        # stage 2
        # for level 1 to 2
        self.hvi_down_stage2_level1to2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.i_down_stage2_level1to2 = NormDownsample(ch2, ch3, use_norm = norm)
        # for level 2 to 3
        # for level 1 to 3
        self.hvi_down_stage2_level1to2_prime = NormDownsample(ch2, ch3, use_norm = norm)
        self.i_down_stage2_level1to2_prime = NormDownsample(ch2, ch3, use_norm = norm)
        self.hvi_down_stage2_level2to3_prime = NormDownsample(ch3, ch4, use_norm = norm)
        self.i_down_stage2_level2to3_prime = NormDownsample(ch3, ch4, use_norm = norm)

        self.hvi_down_stage2_level2to3_secondary = NormDownsample(ch3, ch4, use_norm = norm)
        self.i_down_stage2_level2to3_secondary = NormDownsample(ch3, ch4, use_norm = norm)
        
        # stage 3
        # for level 3 to 2
        self.hvi_up_stage3_level3to2 = NormUpsample(ch4, ch3, use_norm = norm)
        self.i_up_stage3_level3to2 = NormUpsample(ch4, ch3, use_norm = norm)
        # for level 3 to 1
        self.hvi_up_stage3_level3to2_prime = NormUpsample(ch4, ch3, use_norm = norm)
        self.i_up_stage3_level3to2_prime = NormUpsample(ch4, ch3, use_norm = norm)

        self.hvi_up_stage3_level2to1_prime = NormUpsample(ch3, ch2, use_norm = norm)
        self.i_up_stage3_level2to1_prime = NormUpsample(ch3, ch2, use_norm = norm)

        self.hvi_up_stage3_level2to1_secondary = NormUpsample(ch3, ch2, use_norm = norm)
        self.i_up_stage3_level2to1_secondary = NormUpsample(ch3, ch2, use_norm = norm)

        # final processing
        self.hvi_up_final_level2to1 = NormUpsample(ch3, ch2, use_norm = norm)
        self.hvi_up_final_level3to2_prime = NormUpsample(ch4, ch3, use_norm = norm)
        self.hvi_up_final_level2to1_prime = NormUpsample(ch3, ch2, use_norm = norm)

        self.i_up_final_level2to1 = NormUpsample(ch3, ch2, use_norm = norm)
        self.i_up_final_level3to2_prime = NormUpsample(ch4, ch3, use_norm = norm)
        self.i_up_final_level2to1_prime = NormUpsample(ch3, ch2, use_norm = norm)


        # out proj
        self.hv_proj_out =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch2, 2, 3, stride=1, padding=0,bias=False),
            )
        self.i_proj_out =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch2, 1, 3, stride=1, padding=0,bias=False),
            )
        
    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)

        # input prep

        hvi_level1 = self.hvi_proj_in_level1(hvi)
        i_level1 = self.i_proj_in_level1(i)

        hvi_level2 = self.hvi_proj_in_level2(hvi_level1)
        i_level2 = self.i_proj_in_level2(i_level1)

        hvi_level3 = self.hvi_proj_in_level3(hvi_level2)
        i_level3 = self.i_proj_in_level3(i_level2) 

        # stage 1
        input_hvi_level1_stage1 = hvi_level1
        input_i_level1_stage1 = i_level1
        output_hvi_level1_stage1, output_i_level1_stage1 = self.LCA_stage1_level1(input_hvi_level1_stage1, input_i_level1_stage1)
        output_hvi_level1_stage1, output_i_level1_stage1 = self.LCA_stage1_level1_prime(output_hvi_level1_stage1, output_i_level1_stage1)

        input_hvi_level2_stage1 = hvi_level2
        input_i_level2_stage1 = i_level2
        output_hvi_level2_stage1, output_i_level2_stage1 = self.LCA_stage1_level2(input_hvi_level2_stage1, input_i_level2_stage1)
        output_hvi_level2_stage1, output_i_level2_stage1 = self.LCA_stage1_level2_prime(output_hvi_level2_stage1, output_i_level2_stage1)
        output_hvi_level2_stage1, output_i_level2_stage1 = self.LCA_stage1_level2_second(output_hvi_level2_stage1, output_i_level2_stage1)
        output_hvi_level2_stage1, output_i_level2_stage1 = self.LCA_stage1_level2_third(output_hvi_level2_stage1, output_i_level2_stage1)

        input_hvi_level3_stage1 = hvi_level3
        input_i_level3_stage1 = i_level3
        output_hvi_level3_stage1, output_i_level3_stage1 = self.LCA_stage1_level3(input_hvi_level3_stage1, input_i_level3_stage1)
        output_hvi_level3_stage1, output_i_level3_stage1 = self.LCA_stage1_level3_prime(output_hvi_level3_stage1, output_i_level3_stage1)
        output_hvi_level3_stage1, output_i_level3_stage1 = self.LCA_stage1_level3_second(output_hvi_level3_stage1, output_i_level3_stage1)
        output_hvi_level3_stage1, output_i_level3_stage1 = self.LCA_stage1_level3_third(output_hvi_level3_stage1, output_i_level3_stage1)

        # stage 2
        input_hvi_level1_stage2 = output_hvi_level1_stage1
        input_i_level1_stage2 = output_i_level1_stage1
        output_hvi_level1_stage2, output_i_level1_stage2 = self.LCA_stage2_level1(input_hvi_level1_stage2, input_i_level1_stage2)
        output_hvi_level1_stage2, output_i_level1_stage2 = self.LCA_stage2_level1_prime(output_hvi_level1_stage2, output_i_level1_stage2)

        input_hvi_level2_stage2 = output_hvi_level2_stage1 + self.hvi_down_stage2_level1to2(output_hvi_level1_stage1)
        input_i_level2_stage2 = output_i_level2_stage1 + self.i_down_stage2_level1to2(output_i_level1_stage1)
        output_hvi_level2_stage2, output_i_level2_stage2 = self.LCA_stage2_level2(input_hvi_level2_stage2, input_i_level2_stage2)
        output_hvi_level2_stage2, output_i_level2_stage2 = self.LCA_stage2_level2_prime(output_hvi_level2_stage2, output_i_level2_stage2)
        output_hvi_level2_stage2, output_i_level2_stage2 = self.LCA_stage2_level2_second(output_hvi_level2_stage2, output_i_level2_stage2)
        output_hvi_level2_stage2, output_i_level2_stage2 = self.LCA_stage2_level2_third(output_hvi_level2_stage2, output_i_level2_stage2)

        input_hvi_level3_stage2 = output_hvi_level3_stage1 + self.hvi_down_stage2_level2to3_secondary(output_hvi_level2_stage1) + self.hvi_down_stage2_level2to3_prime(self.hvi_down_stage2_level1to2_prime(output_hvi_level1_stage1))
        input_i_level3_stage2 = output_i_level3_stage1 + self.i_down_stage2_level2to3_secondary(output_i_level2_stage1) + self.i_down_stage2_level2to3_prime(self.i_down_stage2_level1to2_prime(output_i_level1_stage1))
        output_hvi_level3_stage2, output_i_level3_stage2 = self.LCA_stage2_level3(input_hvi_level3_stage2, input_i_level3_stage2)
        output_hvi_level3_stage2, output_i_level3_stage2 = self.LCA_stage2_level3_prime(output_hvi_level3_stage2, output_i_level3_stage2)
        output_hvi_level3_stage2, output_i_level3_stage2 = self.LCA_stage2_level3_second(output_hvi_level3_stage2, output_i_level3_stage2)
        output_hvi_level3_stage2, output_i_level3_stage2 = self.LCA_stage2_level3_third(output_hvi_level3_stage2, output_i_level3_stage2)


        # stage 3
        input_hvi_level1_stage3 = output_hvi_level1_stage2 + self.hvi_up_stage3_level2to1_secondary(output_hvi_level2_stage2) + self.hvi_up_stage3_level2to1_prime(self.hvi_up_stage3_level3to2_prime(output_hvi_level3_stage2))
        input_i_level1_stage3 = output_i_level1_stage2 + self.i_up_stage3_level2to1_secondary(output_i_level2_stage2) + self.i_up_stage3_level2to1_prime(self.i_up_stage3_level3to2_prime(output_i_level3_stage2))
        output_hvi_level1_stage3, output_i_level1_stage3 = self.LCA_stage3_level1(input_hvi_level1_stage3, input_i_level1_stage3)
        output_hvi_level1_stage3, output_i_level1_stage3 = self.LCA_stage3_level1_prime(output_hvi_level1_stage3, output_i_level1_stage3)

        input_hvi_level2_stage3 = output_hvi_level2_stage2 + self.hvi_up_stage3_level3to2(output_hvi_level3_stage2)
        input_i_level2_stage3 = output_i_level2_stage2 + self.i_up_stage3_level3to2(output_i_level3_stage2)
        output_hvi_level2_stage3, output_i_level2_stage3 = self.LCA_stage3_level2(input_hvi_level2_stage3, input_i_level2_stage3)
        output_hvi_level2_stage3, output_i_level2_stage3 = self.LCA_stage3_level2_prime(output_hvi_level2_stage3, output_i_level2_stage3)
        output_hvi_level2_stage3, output_i_level2_stage3 = self.LCA_stage3_level2_second(output_hvi_level2_stage3, output_i_level2_stage3)
        output_hvi_level2_stage3, output_i_level2_stage3 = self.LCA_stage3_level2_third(output_hvi_level2_stage3, output_i_level2_stage3)

        input_hvi_level3_stage3 = output_hvi_level3_stage2
        input_i_level3_stage3 = output_i_level3_stage2
        output_hvi_level3_stage3, output_i_level3_stage3 = self.LCA_stage3_level3(input_hvi_level3_stage3, input_i_level3_stage3)
        output_hvi_level3_stage3, output_i_level3_stage3 = self.LCA_stage3_level3_prime(output_hvi_level3_stage3, output_i_level3_stage3)
        output_hvi_level3_stage3, output_i_level3_stage3 = self.LCA_stage3_level3_second(output_hvi_level3_stage3, output_i_level3_stage3)
        output_hvi_level3_stage3, output_i_level3_stage3 = self.LCA_stage3_level3_third(output_hvi_level3_stage3, output_i_level3_stage3)

        #output
        output_hv = output_hvi_level1_stage3 + self.hvi_up_final_level2to1(output_hvi_level2_stage3) + self.hvi_up_final_level2to1_prime(self.hvi_up_final_level3to2_prime(output_hvi_level3_stage3))
        output_i = output_i_level1_stage3 + self.i_up_final_level2to1(output_i_level2_stage3) + self.i_up_final_level2to1_prime(self.i_up_final_level3to2_prime(output_i_level3_stage3))
        # hv_1 = self.HVD_block1(hv_1, hv_jump0)
        # hv_0 = self.HVD_block0(hv_1)

        hv_0 = self.hv_proj_out(output_hv)
        i_0 = self.i_proj_out(output_i)
        
        output_hvi = torch.cat([hv_0, i_0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)
        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()
        
    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        # low
        # --- downsample normal
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        # ---
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        # --- t block
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        # ---
        v_jump1 = i_enc2
        hv_jump1 = hv_2

        # --- downsample
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        # ---
        
        # --- t block
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        # ---
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        # --- - downsample
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        # ---
        
        # --- t block
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        # ---
        
        # --- t block
        i_dec4 = self.I_LCA4(i_enc4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        # ---

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        # ---t block
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        # ---

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        # --- t block
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        # ---
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)
        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi