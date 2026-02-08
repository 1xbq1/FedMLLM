import random
import torch
import numpy as np
import ot
import torch.nn.functional as F

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
        with torch.no_grad():  # 避免计算图存储
            for key in global_dict.keys():
                # 预分配 delta_w，避免 sum([...]) 生成新张量
                delta_w = torch.zeros_like(global_dict[key])
                for client in clients_this_round:
                    weight = sample_num_list[client] / sample_this_round
                    delta_w.add_(local_dict_list[client][key] - global_dict[key], alpha=weight)

                # 计算 proxy_dict，使用 in-place 操作
                if round_idx > 0:
                    proxy_dict[key].mul_(fed_args.fedopt_beta1).add_(delta_w, alpha=(1 - fed_args.fedopt_beta1))
                else:
                    proxy_dict[key].copy_(delta_w)

                # 计算 global_dict，使用 in-place 操作
                global_dict[key].add_(proxy_dict[key])

        # Momentum-based FedAvg
        # for key in global_dict.keys():
        #     delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
        #     proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
        #     global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'fedadagrad' or 'fedadagrad' in fed_args.fed_alg:
        with torch.no_grad():  # 避免计算图存储
            for key, param in opt_proxy_dict.items():
                # 预分配 delta_w，避免 sum(...) 生成新张量
                delta_w = torch.zeros_like(global_dict[key])
                for client in clients_this_round:
                    delta_w.add_(local_dict_list[client][key] - global_dict[key], alpha=1 / len(clients_this_round))

                # 直接赋值 proxy_dict
                proxy_dict[key].copy_(delta_w)

                # 计算 opt_proxy_dict
                opt_proxy_dict[key].add_(torch.square(proxy_dict[key]))  # 直接 in-place 更新

                # 计算 global_dict，使用 addcdiv_ 进行 in-place 除法
                global_dict[key].addcdiv_(proxy_dict[key], torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau, value=fed_args.fedopt_eta)

        # for key, param in opt_proxy_dict.items():
        #     delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
        #     # In paper 'adaptive federated optimization', momentum is not used
        #     proxy_dict[key] = delta_w
        #     opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
        #     global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedyogi' or 'fedyogi' in fed_args.fed_alg:
        with torch.no_grad():  # 避免计算图存储
            for key, param in opt_proxy_dict.items():
                # 预分配 delta_w，避免 sum(...) 生成新张量
                delta_w = torch.zeros_like(global_dict[key])
                for client in clients_this_round:
                    delta_w.add_(local_dict_list[client][key] - global_dict[key], alpha=1 / len(clients_this_round))

                # 计算 proxy_dict
                if round_idx > 0:
                    proxy_dict[key].mul_(fed_args.fedopt_beta1).add_(delta_w, alpha=(1 - fed_args.fedopt_beta1))
                else:
                    proxy_dict[key].copy_(delta_w)

                # 计算 delta_square
                delta_square = torch.square(proxy_dict[key])

                # 计算 opt_proxy_dict
                update_mask = torch.sign(param - delta_square)  # 计算符号，避免显存重复计算
                opt_proxy_dict[key].sub_((1 - fed_args.fedopt_beta2) * delta_square * update_mask)

                # 计算 global_dict，使用 addcdiv_ 进行 in-place 更新
                global_dict[key].addcdiv_(proxy_dict[key], torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau, value=fed_args.fedopt_eta)

        # for key, param in opt_proxy_dict.items():
        #     delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
        #     proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
        #     delta_square = torch.square(proxy_dict[key])
        #     opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
        #     global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedadam' or 'fedadam' in fed_args.fed_alg:
        with torch.no_grad():  # 避免计算图存储
            for key, param in opt_proxy_dict.items():
                # 计算 delta_w，避免 sum 产生额外的 tensor
                delta_w = torch.zeros_like(global_dict[key])  # 预分配 delta_w，避免显存重复分配
                for client in clients_this_round:
                    delta_w.add_(local_dict_list[client][key] - global_dict[key], alpha=1 / len(clients_this_round))

                # 计算 proxy_dict
                if round_idx > 0:
                    proxy_dict[key].mul_(fed_args.fedopt_beta1).add_(delta_w, alpha=(1 - fed_args.fedopt_beta1))
                else:
                    proxy_dict[key].copy_(delta_w)

                # 计算 opt_proxy_dict，避免创建新 tensor
                opt_proxy_dict[key].mul_(fed_args.fedopt_beta2).addcmul_(1 - fed_args.fedopt_beta2, proxy_dict[key], proxy_dict[key])

                # 计算 global_dict，使用 add_ 进行 in-place 更新
                global_dict[key].addcdiv_(proxy_dict[key], (torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau), value=fed_args.fedopt_eta)

        # for key, param in opt_proxy_dict.items():
        #     delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
        #     proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
        #     opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
        #     global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'mean' or 'mean' in fed_args.fed_alg:
        with torch.no_grad():  # 避免计算图存储
            for key in global_dict.keys():
                global_dict[key].zero_()  # 先将 global_dict[key] 置零
                for client in clients_this_round:
                    global_dict[key].add_(local_dict_list[client][key], alpha=1 / len(clients_this_round))  # 逐步加权求均值

        # for key in global_dict.keys():
        #     global_dict[key] = sum(
        #         local_dict_list[client][key] for client in clients_this_round
        #     ) / len(clients_this_round)

    elif fed_args.fed_alg == 'combineam' or 'combineam' in fed_args.fed_alg:
        with torch.no_grad():  # 避免计算图占用显存
            for key in global_dict.keys():
                if 'lora_B' in key:
                    # 直接进行累加计算，避免创建额外列表
                    global_dict[key].zero_().add_(
                        sum(local_dict_list[client][key] for client in clients_this_round) / len(clients_this_round)
                    )
                else:
                    # 直接计算 delta_w，避免创建额外的 tensor 列表
                    delta_w = sum(
                        (local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round
                        for client in clients_this_round
                    )

                    if round_idx > 0:
                        proxy_dict[key].mul_(fed_args.fedopt_beta1).add_((1 - fed_args.fedopt_beta1) * delta_w)
                    else:
                        proxy_dict[key].copy_(delta_w)

                    global_dict[key].add_(proxy_dict[key])  # 直接 in-place 更新

        # for key in global_dict.keys():
        #     if 'lora_B' in key:
        #         global_dict[key] = sum(
        #             local_dict_list[client][key] for client in clients_this_round
        #         ) / len(clients_this_round)
        #     else:
        #         delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
        #         proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
        #         global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'combinema' or 'combinema' in fed_args.fed_alg:
        for key in global_dict.keys():
            if 'lora_B' in key:
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                global_dict[key] = global_dict[key] + proxy_dict[key]
            else:
                global_dict[key] = sum(
                    local_dict_list[client][key] for client in clients_this_round
                ) / len(clients_this_round)

    elif fed_args.fed_alg == 'conflict' or 'conflict' in fed_args.fed_alg:
        global_dict = aggregate_lora_weights(global_dict, local_dict_list, clients_this_round, sample_num_list, sample_this_round, method='avgm')

    else:   # Normal dataset-size-based aggregation
        with torch.no_grad():  # 避免计算图存储
            for key in global_dict.keys():
                global_dict[key].zero_()  # 直接就地清零
                for client in clients_this_round:
                    weight = sample_num_list[client] / sample_this_round
                    global_dict[key].add_(local_dict_list[client][key], alpha=weight)  # 直接加权更新

        # flag = True
        # for key in global_dict.keys():
        #     for client in clients_this_round:
        #         weight = sample_num_list[client] / sample_this_round
        #         if flag:
        #             global_dict[key] = local_dict_list[client][key] * weight
        #             flag = False
        #         else:
        #             global_dict[key] += local_dict_list[client][key] * weight
            # global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])

    return global_dict, global_auxiliary


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
            for i, A_i in enumerate(lora_A_list):
                sim_sum = sum(cosine_similarity(A_i, A_j) for j, A_j in enumerate(lora_A_list) if i != j)
                similarities.append(sim_sum)
            norm_weights_A = normalize_weights(similarities)

            similarities = []
            for i, B_i in enumerate(lora_B_list):
                sim_sum = sum(cosine_similarity(B_i, B_j) for j, B_j in enumerate(lora_B_list) if i != j)
                similarities.append(sim_sum)
            norm_weights_B = normalize_weights(similarities)

            print("norm_weights_A", norm_weights_A)
            print("norm_weights_B", norm_weights_B)

            if method == 'mean':
                global_dict[key] = sum(w * A for w, A in zip(norm_weights_A, lora_A_list))
                global_dict[key_B] = sum(w * B for w, B in zip(norm_weights_A, lora_B_list))
            elif method == 'avgm':
                delta_w = sum([(A - global_dict[key]) * w for w, A in zip(norm_weights_A, lora_A_list)])
                global_dict[key] = global_dict[key] + delta_w

                delta_w_B = sum([(B - global_dict[key_B]) * w for w, B in zip(norm_weights_A, lora_B_list)])
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


