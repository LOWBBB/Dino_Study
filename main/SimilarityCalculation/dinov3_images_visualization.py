import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model.to(device)

img = Image.open("../data/test_imgs/oxfordMuseum.jpeg")
inputs = processor(images=img, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    patch_flat = outputs.last_hidden_state[:, 1 + model.config.num_register_tokens:, :]
    B, N, C = patch_flat.shape
    H = W = int(np.sqrt(N))
    feat_map = patch_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # 转为特征图形式

# PCA 可视化
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_feat = pca.fit_transform(patch_flat.squeeze(0).cpu().numpy())
pca_img = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min())
plt.imshow(pca_img.reshape(H, W, 3))
plt.show()

# 余弦相似度热力图
def cosine_heatmap(feat_map, target_patch):
    target_feat = feat_map[:, :, target_patch[0], target_patch[1]]
    similarities = torch.nn.functional.cosine_similarity(
        target_feat.unsqueeze(2).unsqueeze(3), feat_map, dim=1
    )
    return similarities.squeeze(0).cpu().numpy()

heatmap = cosine_heatmap(feat_map, (H//2, W//2))
plt.imshow(heatmap, cmap='viridis')
plt.show()

# 特征图可视化
for ch in range(16):  # 可视化前 16 通道
    ch_map = feat_map[0, ch].cpu().numpy()
    ch_map = (ch_map - ch_map.min()) / (ch_map.max() - ch_map.min())
    plt.imshow(ch_map, cmap='gray')
    plt.show()