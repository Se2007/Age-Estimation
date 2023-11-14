import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
# from timm.models.volo import VOLO
import volo
import time

class DWConv(nn.Module):
    def __init__(self, nin, nout, kernel=3, pad=1, stride=1, dropout=0.25):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel,stride=stride, padding=pad, groups=nin)
        self.depthwise_dropout = nn.Dropout2d(p=dropout)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.pointwise_dropout = nn.Dropout2d(p=dropout)


    def forward(self, x):
        out = self.depthwise_dropout(self.depthwise(x))
        out = self.pointwise_dropout(self.pointwise(out))
        return out
    

    
class Head(nn.Module):
  def __init__(self, in_channel, num_head_node, embedding_size=512, dw_kernel=3):
    super(Head, self).__init__()

    self.dw_conv = nn.Sequential(
       DWConv(in_channel, in_channel, kernel=dw_kernel, pad=0),
       nn.BatchNorm2d(in_channel),
       nn.Conv2d(in_channel, embedding_size, kernel_size=1)
       )
    
    self.embedding = nn.Sequential(
       nn.AdaptiveAvgPool2d(output_size = (1,1)),
       nn.Flatten(),
       nn.BatchNorm1d(embedding_size),
       nn.Linear(embedding_size, embedding_size),
       nn.ReLU(),
       nn.Linear(embedding_size, num_head_node)
       )

  def forward(self, x):

    y = self.dw_conv(x)
    y = self.embedding(y)
    return y
  


class Resnet(nn.Module):
    def __init__(self, num_head_node, reset=True):
        super(Resnet, self).__init__()
        if reset == True :
          self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
          self.model = models.resnet50()

        self.model = nn.Sequential(*(list(self.model.children())[:-2]))
        self.head = Head(in_channel=2048, num_head_node=num_head_node)

    def forward(self, x):
        out = self.model(x)
        out = self.head(out)
        return out
    
#########################################

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, stem_conv=True, stem_stride=1, patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384, dropout=0.25):
        super().__init__()
        assert patch_size in [4, 8, 16]
        assert in_chans in [3]

        if stem_conv:
            self.conv = self.create_stem(stem_stride, in_chans, hidden_dim, dropout)
            
        else:
            self.conv = None

        
        self.proj = nn.Conv2d(
            hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride
            )

        self.patch_dim = img_size // patch_size
        self.num_patches = self.patch_dim**2

    def create_stem(self, stem_stride, in_chans, hidden_dim, dropout):
        return nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3, bias=False),  # 112x112
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),  # 112x112
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),  # 112x112
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
            x = self.proj(x)  # B, C, H, W

        return x
    


class MultiHeadAttention(nn.Module):
    def __init__(self, channel, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert channel % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channel // num_heads

        self.W_q = nn.Conv2d(channel, channel, kernel_size=1)
        self.W_k = nn.Conv2d(channel, channel, kernel_size=1)
        self.W_v = nn.Conv2d(channel, channel, kernel_size=1)
        self.W_o = nn.Conv2d(channel, channel, kernel_size=1)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        d_k = q.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def split_heads(self, x):
        Batch, Channel, Hight, Weight = x.size()
        x = x.view(Batch, self.num_heads, self.head_dim, Hight, Weight)
        return x

    def combine_heads(self, x):
        Batch, num_heads, head_dim, Hight, Weight = x.size()
        x = x.contiguous().view(Batch, num_heads*head_dim, Hight, Weight)
        return x

    def forward(self, q, k, v, mask=None):

        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))

        output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        output = self.combine_heads(output)
        output = self.W_o(output)

        return output, attention_weights



    
class FeatureEnhancer(nn.Module):
    def __init__(self, in_channel, num_heads):
        super().__init__()

        self.face2body_cross_attention = MultiHeadAttention(in_channel, num_heads=num_heads)
        self.body2face_cross_attention = MultiHeadAttention(in_channel, num_heads=num_heads)

        self.cov = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel, kernel_size=1, bias=False),  
            nn.BatchNorm2d(in_channel),#####                                      -->    ????!!!
            nn.ReLU())
        

    def forward(self, face, body):
        face_out, _= self.face2body_cross_attention(face, body, body)
        body_out, _= self.body2face_cross_attention(body, face, face)


        out = torch.cat([face_out, body_out], dim=1)
        out = self.cov(out)

        return out
    
    
##################################################################################################################
##################################################################################################################


class GhostModule(nn.Module):
  def __init__(self, in_channel, out_channel, activation=True, dropout=0.25):
    super(GhostModule, self).__init__()
    self._activition = activation
    out_channel = int(in_channel // (2 / (out_channel / in_channel)))

    self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1)#, groups=out_channel)
    self.conv1_dropout = nn.Dropout2d(p=dropout)
    self.BN1 = nn.BatchNorm2d(out_channel)

    self.dw = DWConv(out_channel,out_channel, dropout=dropout)
    self.BN_dw = nn.BatchNorm2d(out_channel)

    if self._activition == True :
        self.func1 = nn.PReLU()
        self.func_dw = nn.PReLU()
    else :
        self.func1 = lambda x: x
        self.func_dw = lambda x: x

  def forward(self, x):
    half_1 = self.conv1_dropout(self.conv1(x))
    half_1 = self.func1(self.BN1(half_1))

    half_2 = self.dw(half_1)
    half_2 = self.func_dw(self.BN_dw(half_2))

    return torch.cat((half_1, half_2), dim=1)
  
class SE(nn.Module):
  def __init__(self,channel, reduction=4, dropout=0.25):
    super(SE, self).__init__()
    self.global_pool = nn.AdaptiveAvgPool2d((1,1))

    self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False,),
            nn.Dropout1d(p=dropout),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Dropout1d(p=dropout),
            nn.Sigmoid()
        )

  def forward(self,x1):

    b, c, _, _ = x1.size()
    x2 = self.global_pool(x1).view(b, c)
    x2 = self.fc(x2).view(b, c, 1, 1)

    return x1 * x2.expand_as(x1)

class DFC(nn.Module):
  def __init__(self, in_channel, out_channel, dropout=0.25):
    super(DFC, self).__init__()
    self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    self.single_conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1),
        nn.Dropout2d(p=dropout),
        nn.BatchNorm2d(out_channel)
    )

    self.horizontal_fc = nn.Sequential(
        nn.Conv2d(out_channel, out_channel, kernel_size=(1,5), stride=1, padding=(0,2), groups=out_channel,bias=False),
        nn.Dropout2d(p=dropout),
        nn.BatchNorm2d(out_channel),
    )

    self.vertical_fc = nn.Sequential(
        nn.Conv2d(out_channel, out_channel, kernel_size=(5,1), stride=1, padding=(2,0), groups=out_channel,bias=False),
        nn.Dropout2d(p=dropout),
        nn.BatchNorm2d(out_channel),
    )

    self.func = nn.Sigmoid()

  def forward(self, x):
    y = self.avg_pool(x)
    y = self.single_conv(y)

    y = self.horizontal_fc(y)
    y = self.vertical_fc(y)

    return self.func(y)

class DwShortcut(nn.Module):
  def __init__(self, in_channel, out_channel, dropout=0.25):
    super(DwShortcut, self).__init__()

    self.fc = nn.Sequential(
        DWConv(nin=in_channel, nout=out_channel, kernel=5, pad=2, stride=2, dropout=dropout),
        nn.BatchNorm2d(out_channel),
        nn.Conv2d(out_channel, out_channel, kernel_size=1),
        nn.Dropout2d(p=dropout),
        nn.BatchNorm2d(out_channel)
    )

  def forward(self, x):
    return self.fc(x)

class GhostBlock(nn.Module):
  def __init__(self, in_channel, out_channel, dw_kernel=3, stride=1, se=False, dropout=0.25):
    super(GhostBlock, self).__init__()
    self.stride = stride

    self.ghost1 = GhostModule(in_channel, out_channel, activation=True, dropout=dropout)
    self.dfc = DFC(in_channel, out_channel, dropout=dropout)

    if self.stride > 1:
        self.conv_dw = nn.Sequential(nn.Conv2d(out_channel, out_channel, dw_kernel, stride=stride, padding=(dw_kernel-1)//2,groups=out_channel, bias=False),
                                     nn.Dropout2d(p=dropout),
                                   nn.BatchNorm2d(out_channel)
                                   )

        self.shortcut = DwShortcut(in_channel, out_channel, dropout=dropout)

    if se:
        self.se = SE(out_channel, dropout=dropout)
    else:
        self.se = None

    self.ghost2 = GhostModule(out_channel, out_channel, activation=False, dropout=dropout)

  def forward(self, x):
    residual = x

    x1 = self.ghost1(x)
    x2 = self.dfc(x)

    y = x1 * F.interpolate(x2, size=(x1.shape[-2], x1.shape[-1]), mode='nearest') #out[:,:self.oup,:,:]

    if self.stride > 1 :
        y = self.conv_dw(y)
        residual = self.shortcut(residual)


    if self.se is not None :
        y = self.se(y)

    y = self.ghost2(y)

    return y + residual


######################################################################################################################
######################################################################################################################

class MultiInputModel(nn.Module):
    def __init__(
            self, num_head_node, in_channel=3, embed_dim=384, patch_size=8, img_size=224, num_head=8, gb_dropout=0.25, pe_dropout=0.25
            ):
        super().__init__()
        self.embedding_dim = embed_dim
        self.in_chans = in_channel
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_head = num_head

        self.face_patch_embed = PatchEmbed(img_size=self.img_size, stem_conv=True, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embedding_dim, dropout=pe_dropout)
        self.body_patch_embed = PatchEmbed(img_size=self.img_size, stem_conv=True, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embedding_dim, dropout=pe_dropout)

        self.feature_enhancer = FeatureEnhancer(in_channel=self.embedding_dim, num_heads=self.num_head)

        self.block1 = GhostBlock(self.embedding_dim, self.embedding_dim*2, stride=2, se=True, dropout=gb_dropout)
        self.block2 = GhostBlock(self.embedding_dim*2, self.embedding_dim*4, stride=2, se=True, dropout=gb_dropout)
        self.block3 = GhostBlock(self.embedding_dim*4, self.embedding_dim*8, stride=2, se=True, dropout=gb_dropout)

        self.head = Head(self.embedding_dim*8, num_head_node=num_head_node)



    def forward(self, face, body):
        face = self.face_patch_embed(face)
        body = self.body_patch_embed(body)

        out = self.feature_enhancer(face, body)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.head(out)

        return out
    


#########################################

class InputModel(nn.Module):
    def __init__(
            self, num_head_node, in_channel=3, embed_dim=384, patch_size=8, img_size=224, num_head=8
            ):
        super().__init__()
        self.embedding_dim = embed_dim
        self.in_chans = in_channel
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_head = num_head

        self.face_patch_embed = PatchEmbed(img_size=self.img_size, stem_conv=True, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embedding_dim)

        self.block1 = GhostBlock(self.embedding_dim, self.embedding_dim*2, stride=2, se=True)
        self.block2 = GhostBlock(self.embedding_dim*2, self.embedding_dim*4, stride=2, se=True)
        self.block3 = GhostBlock(self.embedding_dim*4, self.embedding_dim*8, stride=2, se=True)

        self.head = Head(self.embedding_dim*8, num_head_node=num_head_node)



    def forward(self, face):
        face = self.face_patch_embed(face)

        out = self.block1(face)
        out = self.block2(out)
        out = self.block3(out)

        out = self.head(out)

        return out

##################################################################################




##################################################################################

class MIVolo(nn.Module):#VOLO
    def __init__(
            self, layers, outlook_attention=None, embed_dims=None, downsamples=None, num_heads=None, mlp_ratios=None, qk_scale=None, post_layers=None, num_classes=1,  
            out_kernel=3, out_stride=2, out_padding=1, qkv_bias=False,  attn_drop_rate=0., norm_layer=nn.LayerNorm, drop_path_rate=0., pooling_scale=2, drop_rate=0.,
            in_channel=3, embed_dim=192, patch_size=8, img_size=224, num_head=8, pe_dropout=0.25
            ):
        super().__init__()

        self.embedding_dim = 192
        self.in_chans = 3
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_head = 8

        self.face_patch_embed = PatchEmbed(img_size=self.img_size, stem_conv=True, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embedding_dim, dropout=0.25)
        self.body_patch_embed = PatchEmbed(img_size=self.img_size, stem_conv=True, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embedding_dim, dropout=0.25)

        self.feature_enhancer = FeatureEnhancer(in_channel=self.embedding_dim, num_heads=self.num_head)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size // pooling_scale,
                        img_size // patch_size // pooling_scale,
                        embed_dims[-1]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))

        # set the main block in network
        network = []
        for i in range(len(layers)):
            if outlook_attention[i]:
                # stage 1
                stage = volo.outlooker_blocks(volo.Outlooker, i, embed_dims[i], layers,
                                         downsample=downsamples[i], num_heads=num_heads[i],
                                         kernel_size=out_kernel, stride=out_stride,
                                         padding=out_padding, mlp_ratio=mlp_ratios[i],
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop_rate, norm_layer=norm_layer)
                network.append(stage)
            else:
                # stage 2
                stage = volo.transformer_blocks(volo.Transformer, i, embed_dims[i], layers,
                                           num_heads[i], mlp_ratio=mlp_ratios[i],
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop_path_rate=drop_path_rate,
                                           attn_drop=attn_drop_rate,
                                           norm_layer=norm_layer)
                network.append(stage)

            if downsamples[i]:
                # downsampling between two stages
                network.append(volo.Downsample(embed_dims[i], embed_dims[i + 1], 2))


        self.network = nn.ModuleList(network)

        self.post_network = None
        if post_layers is not None:
            self.post_network = nn.ModuleList([
                volo.get_block(post_layers[i],
                          dim=embed_dims[-1],
                          num_heads=num_heads[-1],
                          mlp_ratio=mlp_ratios[-1],
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          attn_drop=attn_drop_rate,
                          drop_path=0.,
                          norm_layer=norm_layer)
                for i in range(len(post_layers))
            ])

        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        

        # self.volo = volo.volo_d1()

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):  
            if idx == 2:  # add positional encoding after outlooker blocks
                x = x + self.pos_embed  
                x = self.pos_drop(x)  
            x = block(x)

        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward_cls(self, x):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network: ###
            x = block(x)
        return x


    def forward(self, face, body):
        
        face = self.face_patch_embed(face)
        body = self.body_patch_embed(body)

        out = self.feature_enhancer(face, body)

        out = self.forward_tokens(out.permute(0, 2, 3, 1))

        out = self.forward_cls(out)
        out = self.norm(out)

        out = self.head(out[:,0])  ###   if you want to change the head refer to volo.py

        return out

##################################################################################
    

def test(model, device = 'cpu'):
#   with torch.no_grad():
    fr = model.to(device)
    print(fr(torch.randn((12,3,128,128),device=device)).shape)
#   fr(next(iter(train_loader))[0].to((device))).shape

def get_parameters(model, device = 'cpu'):
  fr = model.to(device)
  param_values = [param.data for param in fr.parameters()]

  total_params = sum([param.numel() for param in param_values])

  print(f"\nTotal number of parameters: {total_params}")


  

if __name__ == '__main__':
    model = Resnet(120)
    s = time.time()
    test(model, device='cuda')
    e = time.time()
    print('time : ' ,e-s)
    get_parameters(model)

    '''
    v = MultiHeadAttention(4, 4)
    print(v(torch.randn(1,4,224,224), torch.randn(1,4,224,224), torch.randn(1,4,224,224))[0].shape)

    re = FeatureEnhancer(3, 3)
    print(re(torch.randn(1,3,224,224), torch.randn(1,3,224,224)).shape)
    '''
    layers = [4, 4, 8, 2]  # num of layers in the four blocks
    embed_dims = [192, 384, 384, 384]
    num_heads = [6, 12, 12, 12]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False] # do downsampling after first block
    outlook_attention = [True, False, False, False ]
    
    miv = MIVolo(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 )
    s = time.time()
    print(miv(torch.randn(32,3,224,224), torch.randn(32,3,224,224)).shape)
    # get_parameters(miv)
    e = time.time()
    print('time : ' ,e-s)
    
    # print(miv)

    s = time.time()
    mi = MultiInputModel(1)
    get_parameters(mi)
    mi.eval()
    print(mi(torch.randn(32,3,224,224), torch.randn(32,3,224,224)).shape)
    e = time.time()
    print('time : ' ,e-s)

    