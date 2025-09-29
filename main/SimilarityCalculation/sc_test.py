from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import matplotlib.pyplot as plt
import torch
plt.rcParams['font.sans-serif'] = ['SimHei']
# 生成图像特征
def gen_image_features(processor, model, device, image):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        image_features = outputs.last_hidden_state
        image_features = image_features.mean(dim=1)
        return image_features[0]

# 计算两个图像的相似度
def similarity_image(processor, model, device, image1, image2):
    features1 = gen_image_features(processor, model, device, image1)
    features2 = gen_image_features(processor, model, device, image2)
    cos_sim = torch.cosine_similarity(features1, features2, dim=0)
    cos_sim = (cos_sim + 1) / 2
    return cos_sim.item()

def main():
    model_dir = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    image1 = Image.open("../data/img/image_00345.jpg")
    image2 = Image.open("../data/img/image_00348.jpg")
    similarity = similarity_image(processor, model, device, image1, image2)
    plt.figure()
    plt.axis('off')
    plt.title(f"相似度: {similarity}")
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show()

if __name__ == '__main__':
    main()
