import numpy as np

from maxcut import QAOA_MaxCut,FQAOA

def bilinear_initialization(graph,q,p1params,p2params,gamma_bounds=(0, np.pi),beta_bounds=(0, np.pi/2)):
    param_dict = {1:p1params,2:p2params}
    exp_dict = {1:FQAOA(p1params,graph),2:FQAOA(p2params,graph)}
    for p in range(3,q+1):
        gammas = []
        betas = []
        for j in range(1,p+1):
            if j <= p-2:
                gamma = 2 * param_dict[p-1][:p-1][j-1] - param_dict[p-2][:p-2][j-1]
                beta = 2 * param_dict[p-1][p-1:][j-1] - param_dict[p-2][p-2:][j-1]
            elif j == p-1:
                gamma = param_dict[p-1][:p-1][j-1] + param_dict[p-1][:p-1][j-2] - param_dict[p-2][:p-2][j-2]
                beta = param_dict[p-1][p-1:][j-1] + param_dict[p-1][p-1:][j-2] - param_dict[p-2][p-2:][j-2]
            elif j == p:
                gamma = 2 * gammas[j-2] - gammas[j-3]
                beta = 2 * betas[j-2] - betas[j-3]
            gammas.append(gamma)
            betas.append(beta)
        init_param = gammas+betas
        # print(init_param)
        q = QAOA_MaxCut(graph,gamma_bounds=gamma_bounds,beta_bounds=beta_bounds)
        q.run(p,init_param)
        param_dict[p] = q.optimized_params
        exp_dict[p] = q.optimized_expected_value
    return param_dict,exp_dict