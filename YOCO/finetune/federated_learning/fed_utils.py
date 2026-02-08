import torch
import copy
from sklearn.cluster import KMeans 

def get_proxy_dict(fed_args, global_dict):
    opt_proxy_dict = None
    proxy_dict = None
    if fed_args.fed_alg in ['fedadagrad', 'fedyogi', 'fedadam', 'combineam', 'combinema'] or 'fedadagrad' in fed_args.fed_alg or 'fedyogi' in fed_args.fed_alg or 'fedadam' in fed_args.fed_alg or 'combineam' in fed_args.fed_alg or 'combinema' in fed_args.fed_alg:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.ones_like(global_dict[key]) * fed_args.fedopt_tau**2
    elif fed_args.fed_alg in ['fedavgm', 'conflict'] or 'fedavgm' in fed_args.fed_alg or 'conflict' in fed_args.fed_alg:
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
    return proxy_dict, opt_proxy_dict

def get_modality_list(fed_args, local_dict_list):
    local_dict_stack = torch.stack(local_dict_list, dim=0)
    local_dict_np = local_dict_stack.cpu().numpy()
    kmeans = KMeans(n_clusters=fed_args.modality_num, random_state=0)
    kmeans.fit(local_dict_np)
    centers = torch.tensor(kmeans.cluster_centers_).cuda()
    modality_list = torch.unbind(centers, dim=0)
    return modality_list

def get_auxiliary_dict(fed_args, global_dict):

    if fed_args.fed_alg in ['scaffold']:
        global_auxiliary = {}               # c in SCAFFOLD
        for key in global_dict.keys():
            global_auxiliary[key] = torch.zeros_like(global_dict[key])
        auxiliary_model_list = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]    # c_i in SCAFFOLD
        auxiliary_delta_dict = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]    # delta c_i in SCAFFOLD

    else:
        global_auxiliary = None
        auxiliary_model_list = [None]*fed_args.num_clients
        auxiliary_delta_dict = [None]*fed_args.num_clients

    return global_auxiliary, auxiliary_model_list, auxiliary_delta_dict
