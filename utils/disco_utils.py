from BDD.bdd_solver_py import bdd_solver as bdd_solver
from BDD.bdd_solver_py import bdd_solver_options as bdd_solver_options


import copy
import os
from sys import platform
import numpy
import scipy.io
from scipy import sparse
import numpy as np
import pymeshfix
from shape_match_model_pb import ShapeMatchModel
from BDD.bdd_solver_py import bdd_solver as bdd_solver
from BDD.bdd_solver_py import bdd_solver_options as bdd_solver_options
from pathlib import Path
import igl
from sklearn.neighbors import NearestNeighbors
import time
import os
import scipy.io as sio
import torch
import torch.nn.functional as F
from utils.shape_util import compute_geodesic_distmat
import torch.optim as optim
from tqdm.auto import tqdm

from networks.permutation_network import Similarity

from utils.fmap_util import nn_query, fmap2pointmap
from utils.tensor_util import to_numpy


def get_genus(v, f):
    # from https://math.stackexchange.com/questions/85163/finding-the-topological-genus-of-a-triangulated-surface
    # 2 - 2g = #V - #E + #F
    # => g = - 0.5 * (#V - #E + #F - 2)
    e = igl.edges(f)

    return int(round(-0.5 * (v.shape[0] - e.shape[0] + f.shape[0] -2)))

def get_solver_opts():
    fastdogopts = {
        "omega": 0.5,
        "num_dual_iters": 2000,
        "rel_improvement_slope": 1e-6,
        "cuda_split_long_bdds": False,
        "non_learned_lbfgs_step_size": 1e-5,
        "init_delta": 1.1,
        "delta_growth_rate": 1.2,
        "num_dual_iters_primal_rounding": 100,
        "num_primal_rounds": 200,
    }
    return fastdogopts

def compute_mm_type(var_names, mm_diff, threshold):
    mm_type = {}
    for i in range(len(var_names)):
        var_name = var_names[i]
        cur_mm_diff = mm_diff[i]
        if cur_mm_diff >= threshold:
            direction = 'zero'
        elif cur_mm_diff <= -threshold:
            direction = 'one'
        else:
            direction = 'undecided'
        existing_direction = mm_type.get(var_name, None)
        if existing_direction is None:
            mm_type[var_name] = direction
        elif existing_direction != direction:
            mm_type[var_name] = 'undecided'
    return mm_type

def fix_consistent_variables(sol_var_names, mm_type):
    number_fixations = 0
    G = np.zeros(len(sol_var_names)) - 1
    for (i, var_name) in enumerate(sol_var_names):
        var_mm_type = mm_type[var_name]
        if var_mm_type == 'undecided':
            continue
        if var_mm_type == 'zero':
            G[i] = 0
        elif var_mm_type == 'one':
            G[i] = 1
        else:
            assert False
        number_fixations += 1
    print(f'Fixed: {number_fixations} / {len(mm_type)} = {100.0 * number_fixations / len(mm_type)}% of variables.', flush = True)
    return G

def disco_solver(smm, fastdogopts):
    opts = bdd_solver_options(smm.getIlpObj())
    opts.dual_max_iter = 500
    if "dual_max_iter" in fastdogopts:
        opts.dual_max_iter = fastdogopts["dual_max_iter"]
    print("dual_max_iter: " + str(opts.dual_max_iter))
    #
    opts.dual_tolerance = 1e-8
    if "dual_tolerance" in fastdogopts:
        opts.dual_tolerance = fastdogopts["dual_tolerance"]
    print("dual_tolerance: " + str(opts.dual_tolerance))
    #
    opts.dual_improvement_slope = 1e-7
    if "dual_improvement_slope" in fastdogopts:
        opts.dual_improvement_slope = fastdogopts["dual_improvement_slope"]
    print("dual_improvement_slope: " + str(opts.dual_improvement_slope))
    #
    opts.dual_time_limit = 3600  # seconds
    if "dual_time_limit" in fastdogopts:
        opts.dual_time_limit = fastdogopts["dual_time_limit"]
    print("dual_time_limit: " + str(opts.dual_time_limit))
    #
    opts.incremental_primal_rounding = True
    if "incremental_primal_rounding" in fastdogopts:
        opts.incremental_primal_rounding = fastdogopts["incremental_primal_rounding"]
    print("incremental_primal_rounding: " + str(opts.incremental_primal_rounding))
    #
    opts.incremental_initial_perturbation =  1.1
    if "incremental_initial_perturbation" in fastdogopts:
        opts.incremental_initial_perturbation = fastdogopts["incremental_initial_perturbation"]
    print("incremental_initial_perturbation: " + str(opts.incremental_initial_perturbation))
    #
    opts.incremental_growth_rate = 1.1
    if "incremental_growth_rate" in fastdogopts:
        opts.incremental_growth_rate = fastdogopts["incremental_growth_rate"]
    print("incremental_growth_rate: " + str(opts.incremental_growth_rate))
    #
    opts.incremental_primal_num_itr_lb = 100
    if "incremental_primal_num_itr_lb" in fastdogopts:
        opts.incremental_primal_num_itr_lb = fastdogopts["incremental_primal_num_itr_lb"]
    print("incremental_primal_num_itr_lb: " + str(opts.incremental_primal_num_itr_lb))
    #
    opts.incremental_primal_num_rounds = 100
    if "incremental_primal_num_rounds" in fastdogopts:
        opts.incremental_primal_num_rounds = fastdogopts["incremental_primal_num_rounds"]
    print("incremental_primal_num_rounds: " + str(opts.incremental_primal_num_rounds))
    opts.bdd_solver_type = bdd_solver_options.bdd_solver_types.lbfgs_cuda_mma
    if platform == 'darwin':
        # no cuda support on mac
        print("Warning: cuda not available on mac, fallback to cpu algorithm")
        opts.bdd_solver_type = bdd_solver_options.bdd_solver_types.lbfgs_parallel_mma
    print("bdd_solver_type: " + str(opts.bdd_solver_type))
    print("-- Fastdog + LBFGS--")
    #
    opts.lbfgs_step_size = 1e-8
    if "lbfgs_step_size" in fastdogopts:
        opts.lbfgs_step_size = fastdogopts["lbfgs_step_size"]
    print("lbfgs_step_size: " + str(opts.lbfgs_step_size))
    #
    opts.lbfgs_history_size = 5
    if "lbfgs_history_size" in fastdogopts:
        opts.lbfgs_history_size = fastdogopts["lbfgs_history_size"]
    print("lbfgs_history_size: " + str(opts.lbfgs_history_size))
    #
    opts.lbfgs_required_relative_lb_increase = 1e-6
    if "lbfgs_required_relative_lb_increase" in fastdogopts:
        opts.lbfgs_required_relative_lb_increase = fastdogopts["lbfgs_required_relative_lb_increase"]
    print("lbfgs_required_relative_lb_increase: " + str(opts.lbfgs_required_relative_lb_increase))
    #
    opts.lbfgs_step_size_decrease_factor = 0.8
    if "lbfgs_step_size_decrease_factor" in fastdogopts:
        opts.lbfgs_step_size_decrease_factor = fastdogopts["lbfgs_step_size_decrease_factor"]
    print("lbfgs_step_size_decrease_factor: " + str(opts.lbfgs_step_size_decrease_factor))
    #
    opts.lbfgs_step_size_increase_factor = 1.1
    if "lbfgs_step_size_increase_factor" in fastdogopts:
        opts.lbfgs_step_size_increase_factor = fastdogopts["lbfgs_step_size_increase_factor"]
    print("lbfgs_step_size_increase_factor: " + str(opts.lbfgs_step_size_increase_factor))
    #

    opts.cuda_split_long_bdds = False
    if "cuda_split_long_bdds" in fastdogopts:
        opts.cuda_split_long_bdds = fastdogopts["cuda_split_long_bdds"]
    print("cuda_split_long_bdds: " + str(opts.cuda_split_long_bdds))
    if opts.cuda_split_long_bdds:
        print("BDD Splitting of long BDDs enabled!")

    # Initialize solver:
    solver = bdd_solver(opts)

    # Solve dual problem:
    solver.solve_dual()
    lower_bound = solver.lower_bound()

    # Run primal heuristic:
    obj, G = solver.round()
    G = np.array(G)
    actual_size_G = smm.getFXCombo().shape[0]
    G = G[:actual_size_G]
    print("Final Objective: " + str(obj))
    return lower_bound, G, True

def to_tensor(vert_np, face_np, device):
    vert = torch.from_numpy(vert_np).to(device=device, dtype=torch.float32)
    face = torch.from_numpy(face_np).to(device=device, dtype=torch.long)

    return vert, face

def knn_search(x, X, k=1):
    """
    find indices of k-nearest neighbors of x in X
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(x)
    if k == 1:
        return indices.flatten()
    else:
        return indices

