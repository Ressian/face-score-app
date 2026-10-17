import torch
import torch.nn as nn
import math
import numpy as np

from collections import OrderedDict
# 第9行，如果你要用数据增强：
import albumentations as A

# ===================== 原有代码 完全保留，一字不改 =====================
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.group1(x)+residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block,128, layers[1], stride=2)
        self.layer3 = self._make_layer(block,256, layers[2], stride=2)
        self.layer4 = self._make_layer(block,512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.group2 = nn.Sequential(OrderedDict([
            ('fullyconnected', nn.Linear(512 * block.expansion, num_classes))
        ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, val=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.group2(x)
        return x

# ===================== 新增工具函数（全部放在Nets内部） =====================

def face_detect_and_verify(img_rgb:np.ndarray, conf_thresh=0.8):
    """人脸校验：检测人脸、置信度、yaw偏头角度，返回是否有效人脸
    【注意：仅推理阶段调用；训练不会执行本函数，内部import仅调用时才触发】
    """
    # 延迟导入：只有调用该函数才加载retinaface
    from retinaface import RetinaFace

    resp = RetinaFace.detect_faces(img_rgb)
    if not resp:
        return False, {}
    max_score = -1
    best_face = None
    for k, face in resp.items():
        if face["score"]>max_score:
            max_score = face["score"]
            best_face = face
    if max_score < conf_thresh:
        return False, {}
    lms5 = np.array(best_face["landmarks"])
    left_eye, right_eye, nose = lms5[0], lms5[1], lms5[2]
    eye_mid = (left_eye+right_eye)/2
    dx = nose[0] - eye_mid[0]
    eye_dist = np.sqrt(np.sum((left_eye-right_eye)**2))
    yaw = np.arctan2(dx, eye_dist)*180/np.pi
    face_info = {
        "score":max_score,
        "bbox":best_face["facial_area"],
        "landmarks5":lms5,
        "landmarks68":None,
        "yaw":yaw
    }
    return True, face_info

def calc_facial_geometric_feature(landmark86:np.ndarray):
    """
    输入SCUT‑FBP5500‑v2官方86个人工标注关键点 (86,2)
    输出：14维人脸美学几何特征，维度与ResNetFusion geo_feat_dim=14严格匹配
    索引定义(数据集原生86点)
        0‑16: 眉毛；17‑31:眼睛；32‑45:鼻子；46‑67:嘴唇；68‑85:人脸外轮廓
    """
    lm = landmark86

    # --------三庭关键点选取--------
    brow_center = (lm[8] + lm[9]) / 2.0      # 眉毛中间
    nose_tip = lm[45]                        # 鼻尖
    lip_top = lm[46]                         # 上唇上沿
    chin = lm[84]                            # 下巴最低点

    up_face  = np.linalg.norm(brow_center - nose_tip)
    mid_face = np.linalg.norm(nose_tip - lip_top)
    low_face = np.linalg.norm(lip_top - chin)
    three_1 = up_face / (mid_face + 1e-6)
    three_2 = mid_face / (low_face + 1e-6)

    # --------五眼相关 眼睛轮廓17‑31--------
    left_eye_outer  = lm[17]
    left_eye_inner  = lm[24]
    right_eye_inner = lm[25]
    right_eye_outer = lm[31]

    eye_width_left  = np.linalg.norm(left_eye_outer - left_eye_inner)
    eye_width_right = np.linalg.norm(right_eye_inner - right_eye_outer)
    inter_eye_dist  = np.linalg.norm(left_eye_inner - right_eye_inner)

    face_left  = lm[68]
    face_right = lm[77]
    total_face_width = np.linalg.norm(face_left - face_right)

    five_1 = eye_width_left  / (total_face_width + 1e-6)
    five_2 = inter_eye_dist / (total_face_width + 1e-6)

    # --------面部对称性：左右轮廓成对采样--------
    sym_err = 0.0
    left_contour_idx  = [68, 69, 70, 71, 72]
    right_contour_idx = [77, 76, 75, 74, 73]
    for li, ri in zip(left_contour_idx, right_contour_idx):
        mid_x = (lm[li,0] + lm[ri,0]) / 2.0
        sym_err += abs(lm[li,0] - mid_x) + abs(lm[ri,0] - mid_x)
    sym_err = sym_err / len(left_contour_idx)
    sym_feature = sym_err / (total_face_width + 1e-6)

    # --------鼻子、嘴巴、下颌比例--------
    nose_left  = lm[33]
    nose_right = lm[44]
    nose_w = np.linalg.norm(nose_left - nose_right)

    mouth_left  = lm[48]
    mouth_right = lm[60]
    mouth_w = np.linalg.norm(mouth_left - mouth_right)

    jaw_h = np.linalg.norm(chin - brow_center)

    nose_ratio        = nose_w / (total_face_width + 1e-6)
    mouth_nose_ratio  = mouth_w / (nose_w + 1e-6)
    jaw_ratio         = jaw_h / (total_face_width + 1e-6)

    # --------眼睛高宽比--------
    left_eye_top  = lm[19]
    left_eye_bot  = lm[21]
    right_eye_top = lm[27]
    right_eye_bot = lm[29]

    left_eye_h  = np.linalg.norm(left_eye_top  - left_eye_bot)
    right_eye_h = np.linalg.norm(right_eye_top - right_eye_bot)

    eye_aspect_left  = left_eye_h  / (eye_width_left + 1e-6)
    eye_aspect_right = right_eye_h / (eye_width_right + 1e-6)

    feat = np.array([
        three_1, three_2,
        five_1, five_2,
        sym_feature,
        nose_ratio, mouth_nose_ratio, jaw_ratio,
        eye_aspect_left, eye_aspect_right,
        up_face/(total_face_width+1e-6),
        mid_face/(total_face_width+1e-6),
        low_face/(total_face_width+1e-6),
        total_face_width/(jaw_h+1e-6)
    ], dtype=np.float32)
    return feat


def get_train_aug():
    """训练数据增强，模拟美颜/远景/轻微侧脸，支持关键点同步变换"""
    aug_pipeline = A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.7,1.0), p=0.6),
        A.RandomRotate(limit=12, p=0.4),
        A.GaussianBlur(blur_limit=(1,5), p=0.3),
        A.Downscale(scale_min=0.4, scale_max=0.8, p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.3),
        A.HorizontalFlip(p=0.5),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    return aug_pipeline

# ===================== 新增融合模型 ResNetFusion（放在Nets内部） =====================
class ResNetFusion(nn.Module):
    def __init__(self, block, layers, geo_feat_dim=14):
        super().__init__()
        self.inplanes = 64
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block,128, layers[1], stride=2)
        self.layer3 = self._make_layer(block,256, layers[2], stride=2)
        self.layer4 = self._make_layer(block,512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.img_proj = nn.Linear(512 * block.expansion,128)

        self.fusion_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2)
        self.score_head = nn.Linear(128, 1)
        self.conf_head = nn.Sequential(
            nn.Linear(128,1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, val=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, img_tensor, geo_feat_tensor):
        x = self.group1(img_tensor)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        img_feat = self.img_proj(x)

        geo_feat = self.geo_mlp(geo_feat_tensor)
        fuse = torch.cat([img_feat, geo_feat], dim=-1)
        fuse = self.fusion_head(fuse)
        fuse = self.dropout(fuse)
        score = self.score_head(fuse)
        conf = self.conf_head(fuse)
        return score, conf

if __name__ == "__main__":
    """本地自测：验证特征计算 + ResNetFusion网络前向传播"""
    import numpy as np
    import torch

    # 1. 测试几何特征提取函数
    cache_path = r"D:\颜值打分\landmark68_cache.npy"
    cache_dict = np.load(cache_path, allow_pickle=True).item()
    sample_img_name = list(cache_dict.keys())[0]
    lm86 = cache_dict[sample_img_name]
    print(f"[测试] 输入关键点shape: {lm86.shape}")

    geo14 = calc_facial_geometric_feature(lm86)
    print(f"[测试] 输出14维几何特征shape: {geo14.shape}")
    assert geo14.shape == (14,), f"特征维度错误，期待(14,)，得到{geo14.shape}"
    print(f"[测试] geo14前5个值: {geo14[:5]}")

    # 2. 构造模拟batch数据，测试ResNetFusion网络前向
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH = 2
    # 模拟图片输入: [B,3,224,224]
    dummy_img = torch.randn(BATCH, 3, 224, 224).to(device)
    # 模拟几何特征输入 [B,14]
    dummy_geo = torch.from_numpy(np.stack([geo14]*BATCH)).to(device)

    model = ResNetFusion(block=BasicBlock, layers=[2,2,2,2], geo_feat_dim=14).to(device)
    model.eval()
    with torch.no_grad():
        pred_score, pred_conf = model(dummy_img, dummy_geo)

    print(f"\n[测试] ResNetFusion输出：")
    print(f"pred_score shape: {pred_score.shape}")
    print(f"pred_conf  shape: {pred_conf.shape}")
    print(f"score sample: {pred_score.squeeze().cpu().numpy()}")
    print(f"conf sample:  {pred_conf.squeeze().cpu().numpy()}")

    assert pred_score.shape == (BATCH,1)
    assert pred_conf.shape == (BATCH,1)
    print("\n✅全部自测通过！可以运行New‑Train.py进行训练。")
