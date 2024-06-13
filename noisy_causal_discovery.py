import ges.utils as utils
import numpy as np
from our_utils import *
from scores import HuberScore

#------------------------------- General -----------------------------------#

noise_scale = lambda eps, tau: 2 * tau / eps # Get noise scale given sensitivity (tau) and privacy (eps) parameters.
max_info = lambda eps, gamma, n: n * eps**2/2 + eps * np.sqrt(n*np.log(2/gamma)/2) # compute maximum information.

def alpha_tilde(err_lvl, eps, n, start=1e-5, disc=1e4):
    """Computed the adjusted error level given a privacy level.

    Args:
        err_lvl (float): The original/desired error level.
        eps (float): The privacy parameter.
        n (integer): The number of datapoints.

    Returns:
        The adjusted error level for noisy-select.
    """
    gammas = np.linspace(start, err_lvl, int(disc))
    alpha_tilde = [(err_lvl-gamma)*np.exp(-max_info(eps, gamma, n)) for gamma in gammas]
    
    return np.max(alpha_tilde)

#-----------------------------Noisy-Select Sensitivity----------------------#

def get_sensitivity(data, G_options, lambd=0.5, delta=np.inf, iterations=10):
    """Estimates the score sensitivity for the given graph options and data.

    Args:
        data (ndarray): Dataset.
        G_options (list): A list of graph variants.
        lambd (float): The regularization strength.
        delta (float): The delta parameter in the Huber score. 
                       Defaults to infinity (reverting to BIC score).
        iterations (int): The number of iterations for which we generate
                          a copy of the original data with one random entry
                          replaced.

    Returns:
        A sensitivity estimate.
    """
    
    max_change = 0
    
    for _ in range(iterations):
        idx_1 = np.random.randint(data.shape[0])
        idx_2 = np.random.randint(data.shape[0])
        
        data_prime = data.copy()
        data_prime[idx_1] = data[idx_2]
        
        for G in G_options:
            score_og = score(data, G, lambd=lambd, delta=delta)
            score_prime = score(data_prime, G, lambd=lambd, delta=delta)
            
            max_change = max(max_change, np.abs(score_og - score_prime))
            
    return max_change

#------------------------ NOISY-GES --------------------------#

def get_huber_sensitivity(score_class, G, iter=10):
    max_change = 0

    for _ in range(iter):
        idx_1 = np.random.randint(score_class.n)
        idx_2 = np.random.randint(score_class.n)

        data_prime = score_class.data.copy()
        data_prime[idx_1] = score_class.data[idx_2]
        score_class_2 = HuberScore(data_prime, delta=score_class.delta)

        x, y = np.random.randint(score_class.p), np.random.randint(score_class.p)
        G_prime = G.copy()
        G_prime[x, y] = (1-G_prime[x, y]) # flip one random edge edge

        score_og = score_class._compute_full_score(G_prime)
        score_prime = score_class_2._compute_full_score(G_prime)

        max_change = max(max_change, np.abs(score_og - score_prime))

    return max_change

def noisy_fit(score_class, A0=None, phases=['forward', 'backward'], eps_max=None, \
                           eps_thrsh=None, max_iter=np.inf, return_cpdag=True, iterate=False, debug=0):
    # Select the desired phases
    if len(phases) == 0:
        raise ValueError("Must specify at least one phase")
    # Unless indicated otherwise, initialize to the empty graph
    A0 = np.zeros((score_class.p, score_class.p)) if A0 is None else A0

    if((eps_max is not None and eps_thrsh is None) or (eps_max is None and eps_thrsh is not None)):
        raise ValueError("Either both eps_max and eps_thrsh are None or neither are.")
    # GES procedure
    total_score = 0
    A, score_change = A0, np.Inf
    max_scale = 0
    thrsh_scale = 0
    # Run each phase
    while True:
        last_total_score = total_score
        for phase in phases:
            if eps_max is not None:
                sensitivity = get_huber_sensitivity(score_class, A)
                thrsh_scale = 4*sensitivity/(eps_thrsh * score_class.n)
            threshold = np.random.laplace(scale=thrsh_scale)
            if phase == 'forward':
                fun = forward_step
                # add sensitivity_fun
            elif phase == 'backward':
                fun = backward_step
            elif phase == 'turning':
                fun = turning_step
            else:
                raise ValueError('Invalid phase "%s" specified' % phase)
            print("\nGES %s phase start" % phase) if debug else None
            print("-------------------------") if debug else None
            added = 0
            while added <= max_iter:
                if eps_max is not None:
                    sensitivity = get_huber_sensitivity(score_class, A)
                    max_scale = 4*sensitivity/(eps_max * score_class.n)
                    thrsh_scale = 4*sensitivity/(eps_thrsh * score_class.n)

                score_change, new_A = fun(A, score_class, max_scale, max(0, debug - 1))
                if score_change + np.random.laplace(scale=2*thrsh_scale) > threshold:
                    if return_cpdag:
                        A = utils.pdag_to_cpdag(new_A)
                    else:
                        A = new_A
                    total_score += score_change
                    added += 1
                else:
                    break
            print("-----------------------") if debug else None
            print("GES %s phase end" % phase) if debug else None
            print("Total score: %0.4f" % total_score) if debug else None
            [print(row) for row in A] if debug else None
        if total_score <= last_total_score or not iterate:
            break
    return A, total_score


def forward_step(A, cache, max_scale=0, debug=0):
    # Construct edge candidates (i.e. edges between non-adjacent nodes)
    fro, to = np.where((A + A.T + np.eye(len(A))) == 0)
    edge_candidates = list(zip(fro, to))
    # For each edge, enumerate and score all valid operators
    valid_operators = []
    print("  %d candidate edges" % len(edge_candidates)) if debug > 1 else None
    for (x, y) in edge_candidates:
        valid_operators += score_valid_insert_operators(x, y, A, cache, debug=max(0, debug - 1))
    # Pick the edge/operator with the highest score
    if len(valid_operators) == 0:
        print("  No valid insert operators remain") if debug else None
        return 0, A
    else:
        scores = [op[0] + np.random.laplace(scale=max_scale) for op in valid_operators]
        score, new_A, x, y, T = valid_operators[np.argmax(scores)]
        print("  Best operator: insert(%d, %d, %s) -> (%0.4f)" %
              (x, y, T, score)) if debug else None
        return score, new_A


def backward_step(A, cache, max_scale=0, debug=0):
    # Construct edge candidates:
    #   - directed edges
    #   - undirected edges, counted only once
    fro, to = np.where(utils.only_directed(A))
    directed_edges = zip(fro, to)
    fro, to = np.where(utils.only_undirected(A))
    undirected_edges = filter(lambda e: e[0] > e[1], zip(fro, to))  # zip(fro,to)
    edge_candidates = list(directed_edges) + list(undirected_edges)
    assert len(edge_candidates) == utils.skeleton(A).sum() / 2
    # For each edge, enumerate and score all valid operators
    valid_operators = []
    print("  %d candidate edges" % len(edge_candidates)) if debug > 1 else None
    for (x, y) in edge_candidates:
        valid_operators += score_valid_delete_operators(x, y, A, cache, debug=max(0, debug - 1))
    # Pick the edge/operator with the highest score
    if len(valid_operators) == 0:
        print("  No valid delete operators remain") if debug else None
        return 0, A
    else:
        scores = [op[0] + np.random.laplace(scale=max_scale) for op in valid_operators]
        score, new_A, x, y, H = valid_operators[np.argmax(scores)]
        print("  Best operator: delete(%d, %d, %s) -> (%0.4f)" %
              (x, y, H, score)) if debug else None
        return score, new_A
    
def insert(x, y, T, A):
    """
    Applies the insert operator:
      1) adds the edge x -> y
      2) for all t in T, orients the previously undirected edge t -> y

    Parameters
    ----------
    x : int
        the origin node (i.e. x -> y)
    y : int
        the target node
    T : iterable of ints
        a subset of the neighbors of y which are not adjacent to x
    A : np.array
        the current adjacency matrix

    Returns
    -------
    new_A : np.array
        the adjacency matrix resulting from applying the operator

    """
    # Check inputs
    T = sorted(T)
    if A[x, y] != 0 or A[y, x] != 0:
        raise ValueError("x=%d and y=%d are already connected" % (x, y))
    if len(T) == 0:
        pass
    elif not (A[T, y].all() and A[y, T].all()):
        raise ValueError("Not all nodes in T=%s are neighbors of y=%d" % (T, y))
    elif A[T, x].any() or A[x, T].any():
        raise ValueError("Some nodes in T=%s are adjacent to x=%d" % (T, x))
    # Apply operator
    new_A = A.copy()
    # Add edge x -> y
    new_A[x, y] = 1
    # Orient edges t - y to t -> y, for t in T
    new_A[T, y] = 1
    new_A[y, T] = 0
    return new_A


def score_valid_insert_operators(x, y, A, cache, debug=0):
    p = len(A)
    if A[x, y] != 0 or A[y, x] != 0:
        raise ValueError("x=%d and y=%d are already connected" % (x, y))
    # One-hot encode all subsets of T0, plus one extra column to mark
    # if they pass validity condition 2 (see below)
    T0 = sorted(utils.neighbors(y, A) - utils.adj(x, A))
    if len(T0) == 0:
        subsets = np.zeros((1, p + 1), dtype=np.bool)
    else:
        subsets = np.zeros((2**len(T0), p + 1), dtype=np.bool)
        subsets[:, T0] = utils.cartesian([np.array([False, True])] * len(T0), dtype=np.bool)
    valid_operators = []
    print("    insert(%d,%d) T0=" % (x, y), set(T0)) if debug > 1 else None
    while len(subsets) > 0:
        print("      len(subsets)=%d, len(valid_operators)=%d" %
              (len(subsets), len(valid_operators))) if debug > 1 else None
        # Access the next subset
        T = np.where(subsets[0, :-1])[0]
        passed_cond_2 = subsets[0, -1]
        subsets = subsets[1:]
        # Check that the validity conditions hold for T
        na_yxT = utils.na(y, x, A) | set(T)
        # Condition 1: Test that NA_yx U T is a clique
        cond_1 = utils.is_clique(na_yxT, A)
        if not cond_1:
            # Remove from consideration all other sets T' which
            # contain T, as the clique condition will also not hold
            supersets = subsets[:, T].all(axis=1)
            subsets = utils.delete(subsets, supersets, axis=0)
        # Condition 2: Test that all semi-directed paths from y to x contain a
        # member from NA_yx U T
        if passed_cond_2:
            # If a subset of T satisfied condition 2, so does T
            cond_2 = True
        else:
            # Check condition 2
            cond_2 = True
            for path in utils.semi_directed_paths(y, x, A):
                if len(na_yxT & set(path)) == 0:
                    cond_2 = False
                    break
            if cond_2:
                # If condition 2 holds for NA_yx U T, then it holds for all supersets of T
                supersets = subsets[:, T].all(axis=1)
                subsets[supersets, -1] = True
        print("      insert(%d,%d,%s)" % (x, y, T), "na_yx U T = ",
              na_yxT, "validity:", cond_1, cond_2) if debug > 1 else None
        # If both conditions hold, apply operator and compute its score
        if cond_1 and cond_2:
            # Apply operator
            new_A = insert(x, y, T, A)
            # Compute the change in score
            aux = na_yxT | utils.pa(y, A)
            old_score = cache.local_score(y, aux)
            new_score = cache.local_score(y, aux | {x})
            print("        new: s(%d, %s) = %0.6f old: s(%d, %s) = %0.6f" %
                  (y, aux | {x}, new_score, y, aux, old_score)) if debug > 1 else None
            # Add to the list of valid operators
            valid_operators.append((new_score - old_score, new_A, x, y, T))
            print("    insert(%d,%d,%s) -> %0.16f" %
                  (x, y, T, new_score - old_score)) if debug else None
    # Return all the valid operators
    return valid_operators




# --------------------------------------------------------------------
# Delete operator
#    1. definition in function delete
#    2. enumeration logic (to enumerate and score only valid
#    operators) function in valid_delete_operators


def delete(x, y, H, A):
    H = set(H)
    # Check inputs
    if A[x, y] == 0:
        raise ValueError("There is no (un)directed edge from x=%d to y=%d" % (x, y))
    # neighbors of y which are adjacent to x
    na_yx = utils.na(y, x, A)
    if not H <= na_yx:
        raise ValueError(
            "The given set H is not valid, H=%s is not a subset of NA_yx=%s" % (H, na_yx))
    # Apply operator
    new_A = A.copy()
    # delete the edge between x and y
    new_A[x, y], new_A[y, x] = 0, 0
    # orient the undirected edges between y and H towards H
    new_A[list(H), y] = 0
    # orient any undirected edges between x and H towards H
    n_x = utils.neighbors(x, A)
    new_A[list(H & n_x), x] = 0
    return new_A


def score_valid_delete_operators(x, y, A, cache, debug=0):
    # Check inputs
    if A[x, y] == 0:
        raise ValueError("There is no (un)directed edge from x=%d to y=%d" % (x, y))
    # One-hot encode all subsets of H0, plus one column to mark if
    # they have already passed the validity condition
    na_yx = utils.na(y, x, A)
    H0 = sorted(na_yx)
    p = len(A)
    if len(H0) == 0:
        subsets = np.zeros((1, (p + 1)), dtype=np.bool)
    else:
        subsets = np.zeros((2**len(H0), (p + 1)), dtype=np.bool)
        subsets[:, H0] = utils.cartesian([np.array([False, True])] * len(H0), dtype=np.bool)
    valid_operators = []
    print("    delete(%d,%d) H0=" % (x, y), set(H0)) if debug > 1 else None
    while len(subsets) > 0:
        print("      len(subsets)=%d, len(valid_operators)=%d" %
              (len(subsets), len(valid_operators))) if debug > 1 else None
        # Access the next subset
        H = np.where(subsets[0, :-1])[0]
        cond_1 = subsets[0, -1]
        subsets = subsets[1:]
        # Check if the validity condition holds for H, i.e. that
        # NA_yx \ H is a clique.
        # If it has not been tested previously for a subset of H,
        # check it now
        if not cond_1 and utils.is_clique(na_yx - set(H), A):
            cond_1 = True
            # For all supersets H' of H, the validity condition will also hold
            supersets = subsets[:, H].all(axis=1)
            subsets[supersets, -1] = True
        # If the validity condition holds, apply operator and compute its score
        print("      delete(%d,%d,%s)" % (x, y, H), "na_yx - H = ",
              na_yx - set(H), "validity:", cond_1) if debug > 1 else None
        if cond_1:
            # Apply operator
            new_A = delete(x, y, H, A)
            # Compute the change in score
            aux = (na_yx - set(H)) | utils.pa(y, A) | {x}
            # print(x,y,H,"na_yx:",na_yx,"old:",aux,"new:", aux - {x})
            old_score = cache.local_score(y, aux)
            new_score = cache.local_score(y, aux - {x})
            print("        new: s(%d, %s) = %0.6f old: s(%d, %s) = %0.6f" %
                  (y, aux - {x}, new_score, y, aux, old_score)) if debug > 1 else None
            # Add to the list of valid operators
            valid_operators.append((new_score - old_score, new_A, x, y, H))
            print("    delete(%d,%d,%s) -> %0.16f" %
                  (x, y, H, new_score - old_score)) if debug else None
    # Return all the valid operators
    return valid_operators