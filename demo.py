import torch
import os
import numpy as np

from models.multimodal_net import VaultPredictor

def simulate_clinical_features():
    # 医学参数及其范围/单位
    params = [
        ('ACD', '前房深度', 2.5, 3.5, 'mm'),
        ('WTW', '角膜白到白', 10.5, 12.5, 'mm'),
        ('Pupil Size', '瞳孔直径', 2.0, 6.0, 'mm'),
        ('AL', '眼轴长度', 22.0, 27.0, 'mm'),
        ('K1', '角膜屈光力1', 40.0, 47.0, 'D'),
        ('K2', '角膜屈光力2', 40.0, 47.0, 'D'),
        ('ATA', '房角到房角', 11.0, 13.0, 'mm'),
        ('LT', '晶状体厚度', 3.0, 5.0, 'mm'),
        ('ICL Size', 'ICL 选用片径', 12.0, 13.7, 'mm'),
        ('Age', '年龄', 18, 65, '岁')
    ]
    feats = []
    display_values = []
    np.random.seed()  # 确保每次都不同
    for (short, name, low, high, unit) in params:
        val = np.round(np.random.uniform(low, high), 2)
        feats.append(val)
        display_values.append(f"{short}={val}{unit}")
    return np.array(feats, dtype=np.float32), display_values

def main():
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 随机生成影像与参数
    oct_img = torch.randn(1, 3, 224, 224, device=device)
    ubm_img = torch.randn(1, 3, 224, 224, device=device)
    clinical_feats, display = simulate_clinical_features()
    clinical_feats = torch.tensor(clinical_feats, device=device).unsqueeze(0)  # (1, 10)
    
    print("模拟患者参数：")
    print(' | '.join(display))

    # 模型加载
    model = VaultPredictor().to(device)
    ckpt = "checkpoints/multimodal_simulated.pth"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("已加载训练权重。")
    else:
        print("[警告] 未找到训练权重，将使用随机初始化模型进行演示。")
    model.eval()

    # 推理过程提示
    print('\n正在分析 OCT 影像...')
    print('正在分析 UBM 影像...')
    print('正在融合多模态特征...')

    with torch.no_grad():
        pred = model(oct_img, ubm_img, clinical_feats)
        predicted_vault = float(pred.squeeze().cpu().numpy())  # 取浮点数

    print("\n" + "="*40)
    print(f"预测 ICL 术后拱高：\033[1;36m{predicted_vault:.2f} μm\033[0m")
    print("="*40)

if __name__ == '__main__':
    main()