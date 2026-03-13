import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# 模型路径
model_path = "D:/GithubProject/ViDiC-1K/models/Qwen3-VL-8B-Instruct"

print("加载模型...")
model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path)

# 视频路径
video_path = "D:/GithubProject/ViDiC-1K/videos/new.mp4"

# 检查文件
if not os.path.exists(video_path):
    print(f"❌ 视频文件不存在: {video_path}")
    exit()

print(f"✅ 视频文件: {video_path}")

# === 直接使用视频路径 ===
print("\n=== 处理视频 ===")

# 构建消息 - 直接传入视频路径
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,  # 直接传入文件路径
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 1280 * 28 * 28,  # 降低分辨率
                "total_pixels": 2048 * 28 * 28,  # 控制总token数
            },
            {"type": "text", "text": "描述这个视频的内容。"},
        ],
    }
]

# 应用聊天模板
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"提示词: {text[:100]}...")

# 处理输入 - 直接传入视频路径
inputs = processor(
    text=[text],
    image = None,
    videos=[video_path],  # 直接传入路径
    padding=True,
    return_tensors="pt"
).to(model.device)

# 打印看看是否有 grid_thw 信息
print("Inputs keys:", inputs.keys())
# 应该包含 'grid_thw' 或类似字段

print("生成描述中...")

# 添加明确的模态标识
if "video_grid_thw" in inputs:
    # 从 [1, 1, 3] 重塑为 [1, 3]
    inputs["video_grid_thw"] = inputs["video_grid_thw"].squeeze(1)
    print(f"Fixed video_grid_thw shape: {inputs['video_grid_thw'].shape}")
# 生成
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        # 尝试添加这些参数
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

# 解码
generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
response = processor.decode(generated_ids[0], skip_special_tokens=True)

print("\n📹 视频描述:")
print(response)