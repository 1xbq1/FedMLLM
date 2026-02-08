import random
import torch
import numpy as np
import ot

def get_clients_this_round(fed_args, round, fixed=None):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    elif fixed is not None:
        return fixed
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round

def global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round])
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients

    elif fed_args.fed_alg == 'fedavgm' or 'fedavgm' in fed_args.fed_alg:
        # Momentum-based FedAvg
        for key in global_dict.keys():
#           print("global", key)
#           if 'v_proj' not in key:
#           print("local", key)
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'fedadagrad' or 'fedadagrad' in fed_args.fed_alg:
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedyogi' or 'fedyogi' in fed_args.fed_alg:
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedadam' or 'fedadam' in fed_args.fed_alg:
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'mean' or 'mean' in fed_args.fed_alg:
        for key in global_dict.keys():
            global_dict[key] = sum(
                local_dict_list[client][key] for client in clients_this_round
            ) / len(clients_this_round)

    elif fed_args.fed_alg == 'conflict' or 'conflict' in fed_args.fed_alg:
        global_dict = aggregate_lora_weights(global_dict, local_dict_list, clients_this_round, sample_num_list, sample_this_round, method='avgm')
#         global_dict = reverse_conflict(global_dict, local_dict_list, clients_this_round, sample_num_list, 'avgm')
#         for key in global_dict.keys():
#             if 'lora' in key:
#                 global_dict[key] = average_majority_direction(local_dict_list, clients_this_round, key)
#             else:
#                 global_dict[key] = sum(
#                     local_dict_list[client][key] for client in clients_this_round
#                 ) / len(clients_this_round)
#         for client in clients_this_round:
#             local_dict_list[client] = personalized_conflict(local_dict_list, client, sample_num_list, agg_type='mean')
#         global_dict = global_conflict(global_dict, clients_this_round, local_dict_list)

#         for key in global_dict.keys():
#             global_dict[key] = sum(
#                 local_dict_list[client][key] for client in clients_this_round
#             ) / len(clients_this_round)

    else:   # Normal dataset-size-based aggregation
        flag = True
        for key in global_dict.keys():
            for client in clients_this_round:
                weight = sample_num_list[client] / sample_this_round
                if flag:
                    global_dict[key] = local_dict_list[client][key] * weight
                    flag = False
                else:
                    global_dict[key] += local_dict_list[client][key] * weight
            # global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])

    return global_dict, global_auxiliary

def reverse_conflict(global_dict, local_dict_list, clients_this_round, sample_num_list, agg_type):
    key_conflict_counts = {key: [] for key in global_dict.keys()}
    all_conflict_values = []
    current_dict = {}
    for key in global_dict.keys():
        if 'lora' in key:
            current_i = get_base_client(clients_this_round, key, local_dict_list)
            current_dict[key] = current_i
            conflict_matrix = torch.zeros(len(local_dict_list))
            for i in range(len(local_dict_list)):
                if i != current_i:
                    product_signs = torch.sign(local_dict_list[current_i][key] * local_dict_list[i][key])
                    total_elements = product_signs.numel()
                    conflict_ratio = (product_signs == -1).float().sum() / total_elements
                    conflict_matrix[i] = conflict_ratio

            key_conflict_counts[key] = conflict_matrix
            all_conflict_values.extend(conflict_matrix)

    mean_conflict = np.mean(all_conflict_values)
    std_conflict = np.std(all_conflict_values)

    for key in global_dict.keys():
        if 'lora' in key:
            current_i = current_dict[key]
            valid_clients = []
            for i in range(len(local_dict_list)):
                max_val = mean_conflict + std_conflict
#                 min_val = mean_conflict - std_conflict
                if key_conflict_counts[key][i] <= mean_conflict:
                    valid_clients.append(i)
                elif key_conflict_counts[key][i] > max_val:
                    if cosine_similarity(local_dict_list[current_i][key], local_dict_list[i][key]) < -0.3:
                        valid_clients.append(i)
        else:
            valid_clients = [i for i in range(len(local_dict_list))]

        print(key)
        print(valid_clients)
        if valid_clients:
            if agg_type == 'avgm':
                sample_this_round_valid = sum([sample_num_list[client] for client in valid_clients])
                delta_w = sum([(local_dict_list[client][key] - local_dict_list[current_i][key]) * sample_num_list[client] / sample_this_round_valid for client in valid_clients])
                local_dict_list[current_i][key] = local_dict_list[current_i][key] + delta_w
            elif agg_type == 'mean':
                local_dict_list[current_i][key] = sum(
                    local_dict_list[client][key] for client in valid_clients
                ) / len(valid_clients)

            global_dict[key] = local_dict_list[current_i][key]

    return global_dict

def personalized_conflict(local_dict_list, current_i, sample_num_list, agg_type):
    key_conflict_counts = {key: [] for key in local_dict_list[current_i].keys()}
    all_conflict_values = []
    for key in local_dict_list[current_i].keys():
        if 'lora' in key:
            conflict_matrix = torch.zeros(len(local_dict_list))
            for i in range(len(local_dict_list)):
                if i != current_i:
                    product_signs = torch.sign(local_dict_list[current_i][key] * local_dict_list[i][key])
                    total_elements = product_signs.numel()
                    conflict_ratio = (product_signs == -1).float().sum() / total_elements
                    conflict_matrix[i] = conflict_ratio

            key_conflict_counts[key] = conflict_matrix
            all_conflict_values.extend(conflict_matrix)

    mean_conflict = np.mean(all_conflict_values)
    std_conflict = np.std(all_conflict_values)

    for key in local_dict_list[current_i].keys():
        if 'lora' in key:
            valid_clients = []
            for i in range(len(local_dict_list)):
                max_val = mean_conflict + std_conflict
                min_val = mean_conflict - std_conflict
                if (key_conflict_counts[key][i] <= max_val) and (key_conflict_counts[key][i] >= min_val):
                    valid_clients.append(i)
        else:
            valid_clients = [i for i in range(len(local_dict_list))]

        print(key)
        print(f"current_i: {current_i}", valid_clients)
        if valid_clients:
            if agg_type == 'avgm':
                sample_this_round_valid = sum([sample_num_list[client] for client in valid_clients])
                delta_w = sum([(local_dict_list[client][key] - local_dict_list[current_i][key]) * sample_num_list[client] / sample_this_round_valid for client in valid_clients])
                local_dict_list[current_i][key] = local_dict_list[current_i][key] + delta_w
            elif agg_type == 'mean':
                local_dict_list[current_i][key] = sum(
                    local_dict_list[client][key] for client in valid_clients
                ) / len(valid_clients)

    return local_dict_list[current_i]


#             if 'lora' in key:
#                 global_dict[key] = average_majority_direction(local_dict_list, clients_this_round, key)
#             else:

def global_conflict(global_dict, clients_this_round, local_dict_list):
    key_conflict_counts = {key: [] for key in global_dict.keys()}
    all_conflict_values = []
    for key in global_dict.keys():
        if 'lora' in key:
            num_clients = len(clients_this_round)
            conflict_matrix = torch.zeros(len(clients_this_round), len(clients_this_round))

            for i, client_i in enumerate(clients_this_round):
                for j, client_j in enumerate(clients_this_round):
                    if i < j:
                        product_signs = torch.sign(local_dict_list[client_i][key] * local_dict_list[client_j][key])
                        total_elements = product_signs.numel()
                        conflict_ratio = (product_signs == -1).float().sum() / total_elements

                        conflict_matrix[i, j] = conflict_ratio
                        conflict_matrix[j, i] = conflict_ratio

            conflict_counts = conflict_matrix.sum(axis=1)
            key_conflict_counts[key] = conflict_counts
            all_conflict_values.extend(conflict_counts)

    mean_conflict = np.mean(all_conflict_values)
    std_conflict = np.std(all_conflict_values)

    for key in global_dict.keys():
        if 'lora' in key:
            valid_clients = []
            for i, client in enumerate(clients_this_round):
#                 if key_conflict_counts[key][i] <= mean_conflict + std_conflict:
                max_val = mean_conflict + std_conflict
                min_val = mean_conflict - std_conflict
                if (key_conflict_counts[key][i] <= max_val) and (key_conflict_counts[key][i] >= min_val):
                    valid_clients.append(client)
        else:
            valid_clients = clients_this_round

        print("-----", key)
        print(valid_clients)
        if valid_clients:
            global_dict[key] = sum(
                local_dict_list[client][key] for client in valid_clients
            ) / len(valid_clients)
    return global_dict

def average_majority_direction(local_dict_list, clients_this_round, key):
    num_clients = len(clients_this_round)
    device = local_dict_list[clients_this_round[0]][key].device  # 获取设备信息

    # 初始化方向统计矩阵 (num_clients, tensor_shape)
    directions = torch.zeros((num_clients, *local_dict_list[clients_this_round[0]][key].shape), device=device)

    # 计算相对于 client_0 的方向 (基准)
    base_client = get_base_client(clients_this_round, key, local_dict_list)
    base_tensor = local_dict_list[base_client][key]

    for i, client in enumerate(clients_this_round):
        directions[i] = torch.sign(local_dict_list[client][key] * base_tensor)

    # 计算每个位置上大多数客户端的方向
    majority_direction = torch.sign(directions.sum(dim=0))

    # 按照多数方向对所有客户端的权重进行加权平均
    aggregated_tensor = torch.zeros_like(base_tensor)
    count = torch.zeros_like(base_tensor)

    for i, client in enumerate(clients_this_round):
        aligned_tensor = local_dict_list[client][key] * (directions[i] == majority_direction)
        aggregated_tensor += aligned_tensor
        count += (directions[i] == majority_direction).float()

    # 避免除以 0
    count[count == 0] = 1

    return aggregated_tensor / count

# 计算余弦相似度
def cosine_similarity(tensor1, tensor2):
    return torch.nn.functional.cosine_similarity(tensor1.flatten().unsqueeze(0), tensor2.flatten().unsqueeze(0))

def get_base_client(clients_this_round, key, local_dict_list):
    client_scores = {}
    for client in clients_this_round:
        similarities = []
        for other_client in clients_this_round:
            if client != other_client:
                sim = cosine_similarity(local_dict_list[client][key], local_dict_list[other_client][key])
                similarities.append(sim.item())

        if similarities:
            mean_sim = np.mean(similarities)
            var_sim = np.var(similarities)
            client_scores[client] = (mean_sim, var_sim)

    base_client = max(client_scores, key=lambda x: (client_scores[x][0], -client_scores[x][1]))

    return base_client

import torch
import torch.nn.functional as F

def cosine_similarity(tensor1, tensor2):
    return F.cosine_similarity(tensor1.view(1, -1), tensor2.view(1, -1), dim=1).item()

def normalize_weights(weights):
    weights = torch.tensor(weights)
    return weights / weights.sum()

def aggregate_lora_weights(global_dict, local_dict_list, clients_this_round, sample_num_list, sample_this_round, method='mean'):

    for key in global_dict.keys():
        if 'lora_A' in key:
            lora_A_list = [client[key] for client in local_dict_list]
            key_B = key.replace('lora_A', 'lora_B')
            lora_B_list = [client[key_B] for client in local_dict_list]

            similarities = []
            for i, B_i in enumerate(lora_B_list):
                sim_sum = sum(cosine_similarity(B_i, B_j) for j, B_j in enumerate(lora_B_list) if i != j)
                similarities.append(sim_sum)
            norm_weights = normalize_weights(similarities)

            if method == 'mean':
                global_dict[key] = sum(w * A for w, A in zip(norm_weights, lora_A_list))
                global_dict[key_B] = sum(w * B for w, B in zip(norm_weights, lora_B_list))
            elif method == 'avgm':
                delta_w = sum([(A - global_dict[key]) * w for w, A in zip(norm_weights, lora_A_list)])
                global_dict[key] = global_dict[key] + delta_w

                delta_w_B = sum([(B - global_dict[key_B]) * w for w, B in zip(norm_weights, lora_B_list)])
                global_dict[key_B] = global_dict[key_B] + delta_w_B

        elif 'lora_B' in key:
            pass

        else:
            if method == 'mean':
                global_dict[key] = sum(
                    local_dict_list[client][key] for client in clients_this_round
                ) / len(clients_this_round)
            elif method == 'avgm':
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
                global_dict[key] = global_dict[key] + delta_w

    return global_dict


