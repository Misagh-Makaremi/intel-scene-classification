"""
app.py
------
Gradio interface to compare CNN Baseline vs ResNet18 Transfer Learning models
on the Intel Image Classification dataset (6 classes).
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gradio as gr
import time
from pathlib import Path

# =====================
# Define SceneCNN architecture (must match training)
# =====================

class ConvBlock(nn.Module):
    """دو کانولوشن 3x3 (Conv-BN-ReLU) و سپس max pooling با اندازه 2x2."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class SceneCNN(nn.Module):
    """مدل پایه‌ی آموزش‌دیده از صفر روی دیتاست Intel."""
    def __init__(self, num_classes=6, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),    # 224 -> 112
            ConvBlock(32,  64),    # 112 -> 56
            ConvBlock(64,  128),   # 56  -> 28
            ConvBlock(128, 256),   # 28  -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

# =====================
# Class names (must match training order)
# =====================
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# =====================
# Load models
# =====================
# ریشه پروژه را مستقل از محل اجرا پیدا می‌کنیم
PROJECT_ROOT = Path(r"C:\Users\ARTA PC\Desktop\Deep Learning\dl-project")
MODELS_DIR = PROJECT_ROOT / "models"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- CNN Baseline (چک‌پوینت به‌صورت dict ذخیره شده) ---
cnn_ckpt = torch.load(MODELS_DIR / "scenecnn_scratch_best.pt",
                      map_location=device, weights_only=False)
cnn_model = SceneCNN(num_classes=6)
cnn_model.load_state_dict(cnn_ckpt["model_state"])     # ← کلید صحیح
cnn_model.to(device).eval()

# کلاس‌ها و اندازه تصویر را از خود چک‌پوینت می‌خوانیم (مطمئن‌تر از هاردکد کردن)
CLASS_NAMES = cnn_ckpt["class_names"]
CNN_IMG_SIZE = cnn_ckpt["img_size"]

# --- ResNet18 (state_dict خام ذخیره شده) ---
resnet_model = models.resnet18(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, len(CLASS_NAMES))
resnet_model.load_state_dict(
    torch.load(MODELS_DIR / "best_resnet18_finetuned.pth", map_location=device)
)
resnet_model.to(device).eval()

print(f"CLASS_NAMES: {CLASS_NAMES} | CNN img_size: {CNN_IMG_SIZE}")

# =====================
# Transform (ImageNet normalization)
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# =====================
# Prediction function
# =====================
def predict(image):
    """
    Returns comparison results for both models.
    """
    if image is None:
        return "لطفاً تصویری آپلود کنید", "", ""
    
    # Preprocess
    img_pil = Image.fromarray(image).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    results = {}
    
    # CNN Baseline prediction
    with torch.no_grad():
        start = time.time()
        cnn_output = cnn_model(img_tensor)
        cnn_time = (time.time() - start) * 1000
        cnn_probs = torch.softmax(cnn_output, dim=1)[0]
        cnn_top3_prob, cnn_top3_idx = torch.topk(cnn_probs, 3)
    
    cnn_result = "**CNN Baseline**\n\n"
    cnn_result += f"⏱️ زمان استنتاج: {cnn_time:.2f} ms\n\n"
    cnn_result += "پیش‌بینی‌های برتر:\n"
    for i in range(3):
        cnn_result += f"{i+1}. {CLASS_NAMES[cnn_top3_idx[i]]}: {cnn_top3_prob[i]*100:.2f}%\n"
    
    results['cnn'] = {
        'class': CLASS_NAMES[cnn_top3_idx[0]],
        'conf': cnn_top3_prob[0].item(),
        'time': cnn_time,
        'text': cnn_result
    }
    
    # ResNet18 Transfer Learning prediction
    with torch.no_grad():
        start = time.time()
        resnet_output = resnet_model(img_tensor)
        resnet_time = (time.time() - start) * 1000
        resnet_probs = torch.softmax(resnet_output, dim=1)[0]
        resnet_top3_prob, resnet_top3_idx = torch.topk(resnet_probs, 3)
    
    resnet_result = "**ResNet18 (Transfer Learning)**\n\n"
    resnet_result += f"⏱️ زمان استنتاج: {resnet_time:.2f} ms\n\n"
    resnet_result += "پیش‌بینی‌های برتر:\n"
    for i in range(3):
        resnet_result += f"{i+1}. {CLASS_NAMES[resnet_top3_idx[i]]}: {resnet_top3_prob[i]*100:.2f}%\n"
    
    results['resnet'] = {
        'class': CLASS_NAMES[resnet_top3_idx[0]],
        'conf': resnet_top3_prob[0].item(),
        'time': resnet_time,
        'text': resnet_result
    }
    
    # Comparison summary
    best_model = 'ResNet18' if results['resnet']['conf'] > results['cnn']['conf'] else 'CNN'
    comparison = f"### 📊 مقایسه\n\n"
    comparison += f"✅ **بهترین پیش‌بینی:** {best_model}\n\n"
    
    if results['cnn']['class'] == results['resnet']['class']:
        comparison += f"✔️ هر دو مدل کلاس **{results['cnn']['class']}** را پیش‌بینی کرده‌اند\n\n"
    else:
        comparison += f"⚠️ پیش‌بینی‌ها متفاوت:\n"
        comparison += f"- CNN: **{results['cnn']['class']}** ({results['cnn']['conf']*100:.1f}%)\n"
        comparison += f"- ResNet18: **{results['resnet']['class']}** ({results['resnet']['conf']*100:.1f}%)\n\n"
    
    speedup = results['cnn']['time'] / results['resnet']['time']
    comparison += f"⚡ سرعت نسبی: ResNet18 {'سریع‌تر' if speedup > 1 else 'کندتر'} است ({speedup:.1f}x)"
    
    return results['cnn']['text'], results['resnet']['text'], comparison


# =====================
# Gradio Interface
# =====================
with gr.Blocks(title="مقایسه مدل‌های طبقه‌بندی تصویر", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🖼️ مقایسه CNN Baseline و ResNet18 Transfer Learning
    
    این اپلیکیشن دو مدل آموزش‌دیده را برای طبقه‌بندی تصاویر مناظر طبیعی و شهری مقایسه می‌کند:
    - **CNN Baseline**: مدل سفارشی آموزش‌دیده از ابتدا (~1.17M پارامتر)
    - **ResNet18 Transfer Learning**: مدل پیش‌آموزش‌دیده ImageNet با Fine-tuning (~11.18M پارامتر)
    
    **کلاس‌ها:** buildings, forest, glacier, mountain, sea, street
    """)
    
    with gr.Row():
        input_image = gr.Image(label="تصویر ورودی", type="numpy")
    
    predict_btn = gr.Button("🔍 مقایسه مدل‌ها", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            cnn_output = gr.Markdown(label="نتیجه CNN")
        with gr.Column():
            resnet_output = gr.Markdown(label="نتیجه ResNet18")
    
    with gr.Row():
        comparison_output = gr.Markdown(label="مقایسه")
    
    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[cnn_output, resnet_output, comparison_output]
    )
    
    gr.Markdown("""
    ---
    ### 📈 نتایج آموزش:
    - **CNN Baseline**: 90.20% دقت تست، ~31 دقیقه زمان آموزش
    - **ResNet18**: 93.80% دقت تست، ~7.5 دقیقه زمان آموزش
    
    ResNet18 نرخ خطا را 36.7% کاهش داده است (9.80% → 6.20%)
    """)

if __name__ == "__main__":
    demo.launch(share=False)