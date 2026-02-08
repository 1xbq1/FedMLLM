import shutil, os, json, re, torch
import ot
from safetensors.torch import load_file, save_file

def align_weights(state_dict1, state_dict2):
    aligned_state_dict2 = {}

    for key in state_dict1.keys():
        if (key in state_dict2) and ('lora' in key):
#             W1 = state_dict1[key].detach().cpu().numpy()  # 原始权重
#             W2 = state_dict2[key].detach().cpu().numpy()  # 目标对齐权重
            W1 = state_dict1[key].detach().cpu().to(torch.float32).numpy()
            W2 = state_dict2[key].detach().cpu().to(torch.float32).numpy()

            print(f"{key}, W1 shape-{W1.shape}")
            print(f"{key}, W2 shape-{W2.shape}")

            if W1.shape == W2.shape:  # 只对形状匹配的权重进行对齐
                # 计算 OT 传输矩阵
                M = ot.dist(W1.reshape(-1, 1), W2.reshape(-1, 1))  # 计算欧式距离
                transport_plan = ot.emd([], [], M)  # 计算最优传输
                W2_aligned = transport_plan @ W2.reshape(-1, 1)  # 执行传输

                original_dtype = state_dict2[key].dtype
                aligned_state_dict2[key] = torch.tensor(W2_aligned.reshape(W2.shape), dtype=original_dtype, device=state_dict2[key].device)
            else:
                aligned_state_dict2[key] = state_dict2[key]  # 跳过形状不匹配的参数
        else:
            aligned_state_dict2[key] = state_dict2[key]  # 跳过缺失的参数

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
    dst_folder=f"./output/output__lora/client-{i}-ot/{sorted_checkpoints[epoch]}"

    exclude_files = ['adapter_model.safetensors']
    copy_files_exclude(src_folder, dst_folder, exclude_files)
    save_file(local_dict_list[i], os.path.join(dst_folder, 'adapter_model.safetensors'))
