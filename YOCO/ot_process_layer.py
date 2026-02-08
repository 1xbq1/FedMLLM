import shutil, os, json, re, torch
import ot
from safetensors.torch import load_file, save_file
import numpy as np

def extract_layer_index(key):
    """提取层索引（假设层编号是数字）"""
    match = re.search(r"\d+", key)  # 找到数字
    return int(match.group()) if match else float('inf')  # 如果没有数字，放最后

def align_weights(state_dict1, state_dict2):
    aligned_state_dict2 = {}
    sorted_keys = sorted(state_dict1.keys(), key=lambda k: (extract_layer_index(k), "lora_A" in k))

    T_prev = None  # 记录上一层的传输矩阵

    for idx, key in enumerate(sorted_keys):
        if key in state_dict2 and 'lora' in key:
            print(f"Processing {key}")

            W1 = state_dict1[key].detach().cpu().to(torch.float32).numpy()
            W2 = state_dict2[key].detach().cpu().to(torch.float32).numpy()

            if W1.shape == W2.shape:
                W_shape = W2.shape  # 记录原始形状

                # **先右乘上一层的传输矩阵**
                if idx > 0 and T_prev is not None:
                    W2 = W2 @ T_prev  # 确保形状匹配

                # **计算当前层的传输矩阵 T_curr**
                n = W1.shape[0]
                a = np.ones(n) / n  # 均匀分布
                b = np.ones(n) / n  # 均匀分布

                M = ot.dist(W1, W2)  # 计算距离矩阵
                T_curr = ot.emd(a, b, M)  # 计算最优传输矩阵

                if T_curr.shape != (n, n):
                    raise ValueError(f"Transport matrix T_curr has incorrect shape {T_curr.shape}, expected ({n}, {n})")

                # **左乘当前层的传输矩阵**
                W2_aligned = T_curr @ W2

                print("update")

                # 更新 T_prev
                T_prev = T_curr

                original_dtype = state_dict2[key].dtype
                aligned_state_dict2[key] = torch.tensor(W2_aligned.reshape(W_shape), dtype=original_dtype, device=state_dict2[key].device)
            else:
                aligned_state_dict2[key] = state_dict2[key]
        else:
            aligned_state_dict2[key] = state_dict2[key]

    return aligned_state_dict2


def get_sorted_checkpoints(folder_path):
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    checkpoints = []
    for subdir in subdirs:
        match = re.match(r'checkpoint-(\d+)', subdir)
        if match:
            checkpoints.append((subdir, int(match.group(1))))

    checkpoints.sort(key=lambda x: x[1])

    return [checkpoint[0] for checkpoint in checkpoints]

def copy_files_exclude(src_dir, dst_dir, exclude_files=[]):
    """
    复制原文件夹中的所有文件到新文件夹，但排除指定文件
    :param src_dir: 源文件夹路径
    :param dst_dir: 目标文件夹路径
    :param exclude_files: 要排除的文件列表
    """
    # 确保目标文件夹存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 遍历源文件夹中的文件
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # 如果文件在排除列表中，则跳过复制
        if filename in exclude_files:
            continue

        # 如果是文件，则复制
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)
        # 如果是文件夹，则递归复制
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

epoch = 1
local_dict_list = []
for i in range(10):
    folder_path = f"./output/output__lora/client-{i}"
    sorted_checkpoints = get_sorted_checkpoints(folder_path)
    adapter_model_path=f"./output/output__lora/client-{i}/{sorted_checkpoints[epoch]}/adapter_model.safetensors"
    adapter_weights = load_file(adapter_model_path)
    local_dict_list.append(adapter_weights)

for i in range(len(local_dict_list)):
    if i > 0:
        local_dict_list[i] = align_weights(local_dict_list[0], local_dict_list[i])
        print(f"client_{i} aligned with client_0")

#     src_folder = f"./output/output__lora/client-{i}"
#     dst_folder = f"./output/output__lora/client-{i}-ot"

    folder_path = f"./output/output__lora/client-{i}"
    sorted_checkpoints = get_sorted_checkpoints(folder_path)
    src_folder=f"./output/output__lora/client-{i}/{sorted_checkpoints[epoch]}"
    dst_folder=f"./output/output__lora/client-{i}-otl/{sorted_checkpoints[epoch]}"

    exclude_files = ['adapter_model.safetensors']
    copy_files_exclude(src_folder, dst_folder, exclude_files)
    save_file(local_dict_list[i], os.path.join(dst_folder, 'adapter_model.safetensors'))
