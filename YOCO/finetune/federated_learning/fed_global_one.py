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
                    if key_conflict_counts[key][i] <= mean_conflict + std_conflict:
                        valid_clients.append(client)
            else:
                valid_clients = clients_this_round

            print("-----", key)
            print(valid_clients)
            if valid_clients:
                if ('-aftot' in fed_args.fed_alg) and ('lora' in key):
                    print("aftot", key)
#                     W1 = local_dict_list[valid_clients[0]][key].detach().cpu().to(torch.float32).numpy()
                    W1 = local_dict_list[valid_clients[0]][key].detach().to(torch.float16)
                    for client in valid_clients:
                        if client != valid_clients[0]:
                            W2 = local_dict_list[client][key].detach().to(torch.float16)

                            M = ot.dist(W1.reshape(-1, 1), W2.reshape(-1, 1))  # 仍然在 GPU 上
                            transport_plan = ot.sinkhorn([], [], M, reg=1e-3)  # 使用 Sinkhorn 近似
#                             transport_plan = ot.emd([], [], M)
                            W2_aligned = transport_plan @ W2.reshape(-1, 1)

                            original_dtype = local_dict_list[client][key].dtype
                            local_dict_list[client][key] = W2_aligned.reshape(W2.shape).to(original_dtype)

#                             W2 = local_dict_list[client][key].detach().cpu().to(torch.float32).numpy()
#                             M = ot.dist(W1.reshape(-1, 1), W2.reshape(-1, 1))
#                             transport_plan = ot.sinkhorn([], [], M, reg=1e-2)
#                             W2_aligned = transport_plan @ W2.reshape(-1, 1)
#
#                             original_dtype = local_dict_list[client][key].dtype
#                             local_dict_list[client][key] = torch.tensor(W2_aligned.reshape(W2.shape), dtype=original_dtype).cuda()

                if 'avgm' in fed_args.fed_alg:
                    sample_this_round_valid = sum([sample_num_list[client] for client in valid_clients])
                    delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round_valid for client in valid_clients])
                    proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                    global_dict[key] = global_dict[key] + proxy_dict[key]
                elif 'mean' in fed_args.fed_alg:
                    global_dict[key] = sum(
                        local_dict_list[client][key] for client in valid_clients
                    ) / len(valid_clients)

#                 global_dict[key] = sum(
#                     local_dict_list[client][key] * (sample_num_list[client] / sample_this_round)
#                     for client in valid_clients
#                 )

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
