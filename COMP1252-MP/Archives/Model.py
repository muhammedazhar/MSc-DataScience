class ConvBlock(nn.Module):
   def __init__(self, in_ch, out_ch):
       super().__init__()
       self.conv = nn.Sequential(
           nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
           nn.BatchNorm2d(out_ch),
           nn.ReLU(inplace=True),
           nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
           nn.BatchNorm2d(out_ch),
           nn.ReLU(inplace=True)
       )

   def forward(self, x):
       return self.conv(x)

class UNetCH(nn.Module):
   def __init__(self, in_channels=27): # 9 channels each for img1, img2, diff
       super().__init__()
       
       # Encoder
       self.enc1 = ConvBlock(in_channels, 64)
       self.enc2 = ConvBlock(64, 128)
       self.enc3 = ConvBlock(128, 256)
       
       # Decoder
       self.dec3 = ConvBlock(256 + 128, 128)
       self.dec2 = ConvBlock(128 + 64, 64)
       self.dec1 = ConvBlock(64, 32)
       
       # Classification Head
       self.cls_head = nn.Sequential(
           nn.AdaptiveMaxPool2d(1),
           nn.Flatten(),
           nn.Linear(256, 1),
           nn.Sigmoid()
       )
       
       # Final Conv
       self.final = nn.Conv2d(32, 1, kernel_size=1)
       
       # Pooling and Upsampling
       self.pool = nn.MaxPool2d(2)
       self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
       
   def forward(self, x):
       # Encoder
       e1 = self.enc1(x)
       e2 = self.enc2(self.pool(e1))
       e3 = self.enc3(self.pool(e2))
       
       # Classification Branch
       cls_output = self.cls_head(e3)
       
       # Decoder with Skip Connections
       d3 = self.dec3(torch.cat([self.up(e3), e2], dim=1))
       d2 = self.dec2(torch.cat([self.up(d3), e1], dim=1))
       d1 = self.dec1(d2)
       
       # Final Output
       seg_output = torch.sigmoid(self.final(d1))
       
       return seg_output, cls_output