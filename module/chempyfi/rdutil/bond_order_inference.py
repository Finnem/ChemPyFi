
from ..mathutils.geometry import calculate_angle
from itertools import combinations, count
from fractions import Fraction
from .geometry import position
from .util import possible_geometries, ideal_bond_lengths
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components
from rdkit import Chem
from rdkit.Chem import rdmolops
import os
import multiprocessing as _mp
import logging
from logging.handlers import QueueHandler, QueueListener
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
_MP_MOL = None
_MP_ALL_PAIRWISE_BOND_ANGLES = None
_MP_ALL_BOND_INDICES = None
_MP_ALL_BOND_LENGTHS = None
_MP_CSR_ADJ_MATRIX = None
_MP_LOG_SUBGRAPHS = False
_MIN_SUBGRAPH_SIZE = 40
_SMT_PENALTY_SCALE = int(os.environ.get("XB_Z3_PENALTY_SCALE", "10000"))


def _scaled_penalty(value, scale=_SMT_PENALTY_SCALE):
    if isinstance(value, (int, np.integer)):
        return int(value) * scale
    return int(Fraction(str(value)).limit_denominator() * scale)


def _progress_add_total(progress, delta=1):
    if progress is None:
        return
    if getattr(progress, "_xb_total_mode", None) == "estimated":
        return
    progress.total = (progress.total or 0) + delta
    progress.refresh()


def _progress_update(progress, delta=1):
    if progress is None:
        return
    progress.update(delta)


def _progress_adjust_total(progress, delta=0):
    if progress is None or not delta:
        return
    progress.total = (progress.total or 0) + delta
    progress.refresh()


def _estimate_subgraph_leaf_count(subgraph, mol, csr_adj_matrix):
    if len(subgraph) <= _MIN_SUBGRAPH_SIZE:
        return 1
    candidates = _iter_cc_bridge_splits(subgraph, mol, csr_adj_matrix)
    if not candidates:
        return 1
    _, comp_a, comp_b = candidates[0]
    return _estimate_subgraph_leaf_count(comp_a, mol, csr_adj_matrix) + _estimate_subgraph_leaf_count(
        comp_b, mol, csr_adj_matrix
    )


def _get_parallel_processes():
    env = os.environ.get('XB_Z3_PROCESSES')
    if env:
        try:
            procs = int(env)
        except ValueError:
            procs = 1
        return max(1, procs)
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 2)


def _configure_worker_logging(log_queue):
    if log_queue is None:
        return
    root_logger = logging.getLogger()
    root_logger.handlers = [QueueHandler(log_queue)]


def _start_log_listener():
    log_queue = _mp.Queue()
    root_logger = logging.getLogger()
    handlers = root_logger.handlers
    if not handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        handlers = [handler]
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()
    return log_queue, listener


def _init_mp_context(
    mol,
    all_bond_indices,
    all_bond_lengths,
    all_pairwise_bond_angles,
    csr_adj_matrix,
    log_queue,
    log_subgraphs,
):
    global _MP_MOL, _MP_ALL_PAIRWISE_BOND_ANGLES, _MP_ALL_BOND_INDICES, _MP_ALL_BOND_LENGTHS, _MP_CSR_ADJ_MATRIX, _MP_LOG_SUBGRAPHS
    _MP_MOL = mol
    _MP_ALL_PAIRWISE_BOND_ANGLES = all_pairwise_bond_angles
    _MP_ALL_BOND_INDICES = all_bond_indices
    _MP_ALL_BOND_LENGTHS = all_bond_lengths
    _MP_CSR_ADJ_MATRIX = csr_adj_matrix
    _MP_LOG_SUBGRAPHS = log_subgraphs
    _configure_worker_logging(log_queue)


def _solve_subgraph_worker(subgraph):
    subgraph, free_bond_pairs, subgraph_id, parent_id = subgraph
    return _solve_subgraph_leaf(
        _MP_MOL,
        subgraph,
        _MP_ALL_PAIRWISE_BOND_ANGLES,
        _MP_ALL_BOND_INDICES,
        _MP_ALL_BOND_LENGTHS,
        _MP_CSR_ADJ_MATRIX,
        log_subgraph=_MP_LOG_SUBGRAPHS,
        free_bond_pairs=free_bond_pairs,
        subgraph_id=subgraph_id,
        parent_id=parent_id,
    )


def _solve_subgraph_any_worker(subgraph):
    subgraph, free_bond_pairs, subgraph_id, parent_id = subgraph
    return _solve_subgraph(
        _MP_MOL,
        subgraph,
        _MP_ALL_PAIRWISE_BOND_ANGLES,
        _MP_ALL_BOND_INDICES,
        _MP_ALL_BOND_LENGTHS,
        _MP_CSR_ADJ_MATRIX,
        log_subgraph=_MP_LOG_SUBGRAPHS,
        free_bond_pairs=free_bond_pairs,
        pool=None,
        subgraph_id=subgraph_id,
        parent_id=parent_id,
        id_counter=None,
        progress=None,
    )


def _solve_subgraph(
    mol,
    subgraph,
    all_pairwise_bond_angles,
    all_bond_indices,
    all_bond_lengths,
    csr_adj_matrix,
    log_subgraph=False,
    free_bond_pairs=None,
    pool=None,
    subgraph_id=None,
    parent_id=None,
    id_counter=None,
    progress=None,
):
    if os.environ.get('XB_DEBUG_SAT') == '1':
        print("Substructure:", [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in sorted(subgraph)])

    if free_bond_pairs is None:
        free_bond_pairs = set()

    estimated_leaf_count = None
    if progress is not None and getattr(progress, "_xb_total_mode", None) == "estimated":
        estimated_leaf_count = _estimate_subgraph_leaf_count(subgraph, mol, csr_adj_matrix)

    if pool is not None and len(subgraph) <= _MIN_SUBGRAPH_SIZE:
        _progress_add_total(progress, 1)
        args = (subgraph, frozenset(free_bond_pairs), subgraph_id, parent_id)
        async_result = pool.apply_async(_solve_subgraph_worker, (args,))
        result = async_result.get()
        _progress_update(progress, 1)
        return result

    if len(subgraph) > _MIN_SUBGRAPH_SIZE:
        parallel_splits = os.environ.get("XB_Z3_PARALLEL_SPLITS", "1") == "1"
        for bond_key, comp_a, comp_b in _iter_cc_bridge_splits(subgraph, mol, csr_adj_matrix):
            if log_subgraph:
                logger.debug("Solving bridge split with bond key %s", bond_key)
            try:
                child_counter = id_counter or count()
                child_a_id = next(child_counter)
                child_b_id = next(child_counter)
                if pool is not None and parallel_splits:
                    child_est_a = 1
                    child_est_b = 1
                    if progress is not None and getattr(progress, "_xb_total_mode", None) == "estimated":
                        child_est_a = _estimate_subgraph_leaf_count(comp_a, mol, csr_adj_matrix)
                        child_est_b = _estimate_subgraph_leaf_count(comp_b, mol, csr_adj_matrix)
                    _progress_add_total(progress, child_est_a + child_est_b)
                    args_a = (comp_a, frozenset(free_bond_pairs | {bond_key}), child_a_id, subgraph_id)
                    args_b = (comp_b, frozenset(free_bond_pairs | {bond_key}), child_b_id, subgraph_id)
                    async_a = pool.apply_async(_solve_subgraph_any_worker, (args_a,))
                    async_b = pool.apply_async(_solve_subgraph_any_worker, (args_b,))
                    result_a = async_a.get()
                    _progress_update(progress, child_est_a)
                    result_b = async_b.get()
                    _progress_update(progress, child_est_b)
                else:
                    result_a = _solve_subgraph(
                        mol,
                        comp_a,
                        all_pairwise_bond_angles,
                        all_bond_indices,
                        all_bond_lengths,
                        csr_adj_matrix,
                        log_subgraph,
                        free_bond_pairs=free_bond_pairs | {bond_key},
                        pool=pool,
                        subgraph_id=child_a_id,
                        parent_id=subgraph_id,
                        id_counter=child_counter,
                        progress=progress,
                    )
                    result_b = _solve_subgraph(
                        mol,
                        comp_b,
                        all_pairwise_bond_angles,
                        all_bond_indices,
                        all_bond_lengths,
                        csr_adj_matrix,
                        log_subgraph,
                        free_bond_pairs=free_bond_pairs | {bond_key},
                        pool=pool,
                        subgraph_id=child_b_id,
                        parent_id=subgraph_id,
                        id_counter=child_counter,
                        progress=progress,
                    )
            except ValueError:
                continue

            bond_order_a = _get_bond_order_from_result(result_a, bond_key)
            bond_order_b = _get_bond_order_from_result(result_b, bond_key)
            if bond_order_a is not None and bond_order_a == bond_order_b:
                if estimated_leaf_count is not None:
                    actual_estimate = (
                        _estimate_subgraph_leaf_count(comp_a, mol, csr_adj_matrix)
                        + _estimate_subgraph_leaf_count(comp_b, mol, csr_adj_matrix)
                    )
                    _progress_adjust_total(progress, actual_estimate - estimated_leaf_count)
                return _merge_subgraph_results(result_a, result_b, bond_key)

    if estimated_leaf_count is not None and estimated_leaf_count > 1:
        _progress_adjust_total(progress, 1 - estimated_leaf_count)
    _progress_add_total(progress, 1)
    result = _solve_subgraph_leaf(
        mol,
        subgraph,
        all_pairwise_bond_angles,
        all_bond_indices,
        all_bond_lengths,
        csr_adj_matrix,
        log_subgraph=log_subgraph,
        free_bond_pairs=free_bond_pairs,
        subgraph_id=subgraph_id,
        parent_id=parent_id,
    )
    _progress_update(progress, 1)
    return result


def _solve_subgraph_leaf(
    mol,
    subgraph,
    all_pairwise_bond_angles,
    all_bond_indices,
    all_bond_lengths,
    csr_adj_matrix,
    log_subgraph=False,
    free_bond_pairs=None,
    subgraph_id=None,
    parent_id=None,
):
    try:
        from z3 import Optimize, sat, Solver
    except ImportError:
        raise ImportError(
            "The z3 package is required for this function. Try to install via 'sudo apt install z3' and 'pip install z3-solver'."
        )

    if free_bond_pairs is None:
        free_bond_pairs = set()

    if log_subgraph:
        logger.debug(
            "Solving subgraph %s (parent=%s) with %s atoms: %s",
            subgraph_id,
            parent_id,
            len(subgraph),
            sorted(subgraph),
        )

    softened_atoms, strict_atoms = _resolve_unsat_cores_iteratively(
        mol,
        subgraph,
        all_pairwise_bond_angles,
        all_bond_indices,
        all_bond_lengths,
        csr_adj_matrix,
        free_bond_pairs=free_bond_pairs,
    )

    formula, objective, variable_dict, named_constraints = _construct_SMT_formulation(
        mol,
        subgraph,
        all_pairwise_bond_angles,
        all_bond_indices,
        all_bond_lengths,
        csr_adj_matrix,
        excluded_atoms=softened_atoms,
        free_bond_pairs=free_bond_pairs,
    )

    s = Solver()
    timeout_ms = os.environ.get('XB_Z3_TIMEOUT_MS')
    if timeout_ms and timeout_ms.isdigit():
        s.set("timeout", int(timeout_ms))
    if os.environ.get('XB_DEBUG_SAT') == '1':
        for cname, cexpr in named_constraints.items():
            s.assert_and_track(cexpr, cname)
    sat_result = s.check()
    if sat_result != sat:
        raise ValueError(f"System is not satisfiable even after constraint softening. SAT result: {sat_result}")

    opt = Optimize()
    timeout_ms = os.environ.get('XB_Z3_TIMEOUT_MS')
    if timeout_ms and timeout_ms.isdigit():
        opt.set("timeout", int(timeout_ms))
    opt.add(formula)
    opt.minimize(objective)
    opt_result = opt.check()
    if opt_result != sat:
        raise ValueError(
            f"Optimization failed for atoms {[mol.GetAtomWithIdx(idx).GetSymbol() + ':' + str(idx) for idx in subgraph]}. Result: {opt_result}"
        )

    model = opt.model()
    new_bond_orders, assigned_charges = _decode_solution(model)

    if log_subgraph:
        logger.debug(
            "Solved subgraph %s (parent=%s) with %s atoms",
            subgraph_id,
            parent_id,
            len(subgraph),
        )
    return subgraph, softened_atoms, new_bond_orders, assigned_charges

def _get_atom_statistics(atoms, mol):
    symbol_count = {}
    for atom in atoms:
        symbol = mol.GetAtomWithIdx(atom).GetSymbol()
        if symbol not in symbol_count:
            symbol_count[symbol] = 0
        symbol_count[symbol] += 1
    return symbol_count

def adj_matrix_from_bond_indices(bond_indices, num_atoms):
    rows = bond_indices[:, 0]
    cols = bond_indices[:, 1]
    data = np.ones(len(rows) * 2, dtype=np.int8)
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    return coo_matrix((data, (all_rows, all_cols)), shape=(num_atoms, num_atoms)).tocsr()

def correct_bond_orders(mol, add_hydrogens = False, try_full = True, verbose = "auto", bond_indices = None):
    try:
        from z3 import Optimize, sat, Solver, IntVal
    except ImportError:
        raise ImportError("The z3 package is required for this function. Try to install via 'sudo apt install z3' and 'pip install z3-solver'.")

    num_bonds = mol.GetNumBonds()
    verbose_mode = False if verbose == "auto" else bool(verbose)

    if verbose_mode: logger.info(f"Fetching approximate bond lengths for {mol.GetNumBonds()} bonds")
    if bond_indices is None:
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        bond_indices = np.argwhere(adj_matrix == 1)
    elif len(bond_indices) != num_bonds:
        logger.warning(f"Bond indices length {len(bond_indices)} does not match number of bonds {num_bonds}. Using all bonds.")
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        bond_indices = np.argwhere(adj_matrix == 1)
    all_bond_indices = bond_indices
    csr_adj_matrix = adj_matrix_from_bond_indices(all_bond_indices, mol.GetNumAtoms())
    positions = position(mol)
    bond_positions = positions[all_bond_indices]
    bond_lengths = np.linalg.norm(bond_positions[:, 0] - bond_positions[:, 1], axis=1)
    if verbose_mode: logger.info(f"Fetching approximate bond angles for {mol.GetNumBonds()} bonds")
    all_pairwise_bond_angles = _get_all_pairwise_bond_angles(mol)

    if verbose_mode:
        logger.info("Fetching violating atoms")
    violating_atom_indices = _get_violating_atom_clusters(mol, all_pairwise_bond_angles, add_hydrogens = add_hydrogens)
    if verbose == "auto":
        verbose_mode = (
            mol.GetNumAtoms() > 300 or len(violating_atom_indices) > 300
        )
        if verbose_mode:
            logger.info(
                "Auto-verbose enabled for bond order inference (atoms=%s, violating=%s)",
                mol.GetNumAtoms(),
                len(violating_atom_indices),
            )
    if verbose_mode: logger.info(f"Found {len(violating_atom_indices)} violating atoms: {_get_atom_statistics(violating_atom_indices, mol)}")
    if len(violating_atom_indices) == 0:
        return mol
    # Expand the set by one-hop neighbors so substructures include bonded context (incl. hydrogens)
    expanded_atom_indices = set(violating_atom_indices)
    for idx in list(violating_atom_indices):
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            expanded_atom_indices.add(nbr.GetIdx())
    if verbose_mode: logger.info(f"Determining connected subgraphs for {len(expanded_atom_indices)} atoms")
    connected_subgraphs = _get_connected_subgraphs(csr_adj_matrix, expanded_atom_indices)
    if verbose_mode: logger.info(f"Found {len(connected_subgraphs)} connected subgraphs")
    subgraph_id_counter = count()
    subgraph_entries = [
        (subgraph, next(subgraph_id_counter), None) for subgraph in connected_subgraphs
    ]
    new_mol = Chem.RWMol(mol)
    try:
        if verbose_mode: logger.info("Solving subgraphs...")
        if verbose_mode:
            try:
                from tqdm import tqdm as _tqdm
            except ImportError:
                _tqdm = None
        else:
            _tqdm = None
        use_parallel = len(connected_subgraphs) > 1
        processes = _get_parallel_processes()
        if use_parallel and processes > 1:
            if verbose_mode: logger.info("Using parallel processing with %s processes for SAT solving", processes)
            try:
                log_queue = None
                log_listener = None
                if verbose_mode:
                    log_queue, log_listener = _start_log_listener()
                ctx = _mp.get_context("fork")
                with ctx.Pool(
                    processes=processes,
                    initializer=_init_mp_context,
                    initargs=(
                        mol,
                        all_bond_indices,
                        bond_lengths,
                        all_pairwise_bond_angles,
                        csr_adj_matrix,
                        log_queue,
                        verbose,
                    ),
                ) as pool:
                    results = []
                    async_results = []
                    progress = None
                    if _tqdm:
                        estimated_total = sum(
                            _estimate_subgraph_leaf_count(subgraph, mol, csr_adj_matrix)
                            for subgraph, _, _ in subgraph_entries
                        )
                        progress = _tqdm(
                            total=estimated_total,
                            desc="Solving subgraphs",
                            unit="subgraph",
                        )
                        progress._xb_total_mode = "estimated"
                    for subgraph, subgraph_id, parent_id in subgraph_entries:
                        if len(subgraph) > _MIN_SUBGRAPH_SIZE:
                            results.append(
                                _solve_subgraph(
                                    mol,
                                    subgraph,
                                    all_pairwise_bond_angles,
                                    all_bond_indices,
                                    bond_lengths,
                                    csr_adj_matrix,
                                    verbose_mode,
                                    free_bond_pairs=set(),
                                    pool=pool,
                                    subgraph_id=subgraph_id,
                                    parent_id=parent_id,
                                    id_counter=subgraph_id_counter,
                                    progress=progress,
                                )
                            )
                        else:
                            _progress_add_total(progress, 1)
                            async_results.append(
                                pool.apply_async(
                                    _solve_subgraph_worker,
                                    ((subgraph, frozenset(), subgraph_id, parent_id),),
                                )
                            )
                    for async_result in async_results:
                        results.append(async_result.get())
                        _progress_update(progress, 1)
                    if progress:
                        progress.close()
            except Exception:
                progress = None
                if _tqdm:
                    estimated_total = sum(
                        _estimate_subgraph_leaf_count(subgraph, mol, csr_adj_matrix)
                        for subgraph, _, _ in subgraph_entries
                    )
                    progress = _tqdm(
                        total=estimated_total,
                        desc="Solving subgraphs",
                        unit="subgraph",
                    )
                    progress._xb_total_mode = "estimated"
                results = [
                                _solve_subgraph(
                        mol,
                        subgraph,
                        all_pairwise_bond_angles,
                        all_bond_indices,
                        bond_lengths,
                        csr_adj_matrix,
                                    verbose_mode,
                        subgraph_id=subgraph_id,
                        parent_id=parent_id,
                        id_counter=subgraph_id_counter,
                        progress=progress,
                    )
                    for subgraph, subgraph_id, parent_id in subgraph_entries
                ]
                if progress:
                    progress.close()
            finally:
                if log_listener:
                    log_listener.stop()
                if log_queue:
                    log_queue.close()
        else:
            progress = None
            if _tqdm:
                estimated_total = sum(
                    _estimate_subgraph_leaf_count(subgraph, mol, csr_adj_matrix)
                    for subgraph, _, _ in subgraph_entries
                )
                progress = _tqdm(
                    total=estimated_total,
                    desc="Solving subgraphs",
                    unit="subgraph",
                )
                progress._xb_total_mode = "estimated"
            results = [
                _solve_subgraph(
                    mol,
                    subgraph,
                    all_pairwise_bond_angles,
                    all_bond_indices,
                    bond_lengths,
                    csr_adj_matrix,
                    verbose_mode,
                    subgraph_id=subgraph_id,
                    parent_id=parent_id,
                    id_counter=subgraph_id_counter,
                    progress=progress,
                )
                for subgraph, subgraph_id, parent_id in subgraph_entries
            ]
            if progress:
                progress.close()
            if verbose_mode:
                logging.info(f"Solved {len(results)} subgraphs")
                logging.info(f"Assigning bond orders and charges to the molecule:")
            
        for subgraph, softened_atoms, new_bond_orders, assigned_charges in results:
            # Print only assignments relevant to softened atoms (only if any softened)
            if softened_atoms:
                print(f"\n=== FINAL ASSIGNMENTS (softened atoms only) ===")
                # Bonds per softened atom
                any_bond_printed = False
                for a in sorted(softened_atoms):
                    atom_symbol = mol.GetAtomWithIdx(a).GetSymbol()
                    printed_for_atom = False
                    for (a1, a2), bo in new_bond_orders.items():
                        if a1 == a or a2 == a:
                            if not any_bond_printed:
                                print("Bond orders:")
                                any_bond_printed = True
                            nbr = a2 if a1 == a else a1
                            nbr_symbol = mol.GetAtomWithIdx(nbr).GetSymbol()
                            print(f"  {atom_symbol}{a}-{nbr_symbol}{nbr}: {bo}")
                            printed_for_atom = True
                    if not printed_for_atom:
                        print(f"  {atom_symbol}{a}: no bond order selection (all 0)")
                # Charges for softened atoms
                any_charge_printed = False
                for a in sorted(softened_atoms):
                    atom_symbol = mol.GetAtomWithIdx(a).GetSymbol()
                    if a in assigned_charges:
                        if not any_charge_printed:
                            print("Formal charges:")
                            any_charge_printed = True
                        print(f"  {atom_symbol}{a}: {assigned_charges[a]} (softened)")
                if not any_charge_printed:
                    print("Formal charges: none selected for softened atoms")

            # Apply the assignments to the molecule
            for (atom1, atom2), bond_order in new_bond_orders.items():
                if bond_order == 0.0:
                    new_mol.RemoveBond(int(atom1), int(atom2))
                else:
                    new_mol.GetBondBetweenAtoms(int(atom1), int(atom2)).SetBondType(Chem.BondType(bond_order))
            for atom, charge in assigned_charges.items():
                new_mol.GetAtomWithIdx(atom).SetFormalCharge(charge)
        return new_mol
    except ValueError as e:
        if try_full:
            
            # Try to soften constraints iteratively for the full molecule
            softened_atoms, strict_atoms = _resolve_unsat_cores_iteratively(
                mol, range(mol.GetNumAtoms()), all_pairwise_bond_angles, all_bond_indices, bond_lengths, csr_adj_matrix
            )
            
            if not strict_atoms:
                pass
            
            # Now solve with all atoms, but with softened constraints for problematic ones
            
            formula, objective, variable_dict, named_constraints = _construct_SMT_formulation(
                mol, range(mol.GetNumAtoms()), all_pairwise_bond_angles, all_bond_indices, bond_lengths, csr_adj_matrix,
                excluded_atoms=softened_atoms
            )
            
            # First check if the system is SAT at all
            s = Solver()
            s.add(formula)
            sat_result = s.check()
            
            if sat_result != sat:
                raise ValueError(f"System is not satisfiable even after constraint softening. SAT result: {sat_result}")
            
            # Debug: dump SAT model (pre-optimization) gated by env var XB_DEBUG_SAT=1
            if os.environ.get('XB_DEBUG_SAT') == '1':
                try:
                    sat_model = s.model()
                    print("\n--- DEBUG: Pre-optimization SAT model (non-zero selections) ---")
                    bond_selected = []
                    charge_selected = []
                    geom_selected = []
                    for key, var in variable_dict.items():
                        try:
                            val = sat_model.evaluate(var, model_completion=True)
                            if hasattr(val, 'as_long'):
                                ival = val.as_long()
                            else:
                                ival = int(str(val))
                        except Exception:
                            continue
                        if ival != 1:
                            continue
                        if isinstance(key, tuple) and len(key) == 3:
                            a1, a2, bo = key
                            bond_selected.append((a1, a2, bo))
                        elif isinstance(key, tuple) and len(key) == 2:
                            a, ch = key
                            charge_selected.append((a, ch))
                        elif isinstance(key, tuple) and len(key) == 4:
                            a, nc, ch, ang = key
                            geom_selected.append((a, nc, ch, ang))
                    if bond_selected:
                        print("Bond orders (==1):")
                        for a1, a2, bo in bond_selected:
                            print(f"  {mol.GetAtomWithIdx(a1).GetSymbol()}{a1}-{mol.GetAtomWithIdx(a2).GetSymbol()}{a2}: {bo}")
                    if charge_selected:
                        print("Formal charges (==1):")
                        for a, ch in charge_selected:
                            ctype = "softened" if a in softened_atoms else "strict"
                            print(f"  {mol.GetAtomWithIdx(a).GetSymbol()}{a}: {ch} ({ctype})")
                    if geom_selected:
                        print("Geometries (==1):")
                        for a, nc, ch, ang in geom_selected:
                            print(f"  {mol.GetAtomWithIdx(a).GetSymbol()}{a}: neighbors={nc}, charge={ch}, angles={ang}")
                except Exception as e:
                    print(f"(debug model dump failed: {e})")

            # If SAT, use Optimize to find the best solution
            
            opt = Optimize()
            opt.add(formula)
            opt.minimize(objective)
            opt_result = opt.check()
            
            if opt_result == sat:
                model = opt.model()
                new_bond_orders, assigned_charges = _decode_solution(model)
                
                # Print only assignments relevant to softened atoms (only if any softened)
                if softened_atoms:
                    print(f"\n=== FINAL ASSIGNMENTS (softened atoms only) ===")
                    soft_bonds = [
                        (a1, a2, bo) for (a1, a2), bo in new_bond_orders.items()
                        if a1 in softened_atoms or a2 in softened_atoms
                    ]
                    if soft_bonds:
                        print(f"Bond orders:")
                        for a1, a2, bo in soft_bonds:
                            atom1_symbol = mol.GetAtomWithIdx(a1).GetSymbol()
                            atom2_symbol = mol.GetAtomWithIdx(a2).GetSymbol()
                            print(f"  {atom1_symbol}{a1}-{atom2_symbol}{a2}: {bo}")
                    soft_charges = [(a, ch) for a, ch in assigned_charges.items() if a in softened_atoms]
                    if soft_charges:
                        print(f"Formal charges:")
                        for atom, charge in soft_charges:
                            atom_symbol = mol.GetAtomWithIdx(atom).GetSymbol()
                            print(f"  {atom_symbol}{atom}: {charge} (softened)")
                
                # Apply the assignments to the molecule
                for (atom1, atom2), bond_order in new_bond_orders.items():
                    if bond_order == 0.0:
                        new_mol.RemoveBond(int(atom1), int(atom2))
                    else:
                        new_mol.GetBondBetweenAtoms(int(atom1), int(atom2)).SetBondType(Chem.BondType(bond_order))
                for atom, charge in assigned_charges.items():
                    new_mol.GetAtomWithIdx(atom).SetFormalCharge(charge)
                return new_mol
            else:
                raise ValueError(f"Optimization failed for the full molecule even after removing problematic atoms. Result: {opt_result}")


def _decode_solution(model):
    new_bond_orders = {}
    assigned_charges = {}
    for decl in model.decls():
        # Get the variable name
        key = decl.name()
        # Retrieve the value assigned to the variable
        value = model[decl]
        if value.as_long() == 1:
            if key.startswith("bond_order"):
                _, _, atom1, atom2, bond_order = key.split("_")
                new_bond_orders[(int(atom1), int(atom2))] = float(bond_order)
            elif key.startswith("charge"):
                _, atom, charge = key.split("_")
                assigned_charges[int(atom)] = int(charge)
    return new_bond_orders, assigned_charges

# .1 A deviation is as bad as 10 degrees
angle_deviation_weighting = 1
# might want to expand considered atoms by one layer of neighbors

ideal_bond_length_order_indices = {1.0: 0, 2.0: 2, 3.0: 3}

def _construct_SMT_formulation(
    mol,
    indices,
    all_pairwise_bond_angles,
    bond_indices,
    bond_lengths,
    csr_adj_matrix,
    angle_tolerance=20,
    enable_atom_activation=False,
    excluded_atoms=None,
    free_bond_pairs=None,
):
    try:
        from z3 import Or, And, Int, Sum, Solver, Implies, IntVal, Bool, BoolVal
    except ImportError:
        raise ImportError("The z3 package is required for this function. Try to install via 'sudo apt install z3' and 'pip install z3-solver'.")

    considered_atoms = [mol.GetAtomWithIdx(idx) for idx in indices]
    penalty_expressions = [IntVal(0)]

    bond_length_lookup = {}
    for (idx1, idx2), length in zip(bond_indices, bond_lengths):
        key = (int(idx1), int(idx2))
        bond_length_lookup[key] = float(length)
        bond_length_lookup[(key[1], key[0])] = float(length)

    if free_bond_pairs is None:
        free_bond_pairs = set()

    def _activation_condition(atom1_idx, atom2_idx):
        if not enable_atom_activation:
            return None
        conditions = []
        if atom1_idx in atom_activation_vars:
            conditions.append(atom_activation_vars[atom1_idx])
        if atom2_idx in atom_activation_vars:
            conditions.append(atom_activation_vars[atom2_idx])
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return And(*conditions)
    
    # Atom activation variables for unsat core analysis
    atom_activation_vars = {}
    if enable_atom_activation:
        for atom_idx in indices:
            atom_activation_vars[atom_idx] = Bool(f"atom_active_{atom_idx}")
    
    # Set of excluded atoms (will have relaxed constraints)
    excluded_atoms = excluded_atoms or set()
    
    # Track constraints with meaningful names based on atoms
    named_constraints = {}
    
    # variables for each bond order
    bond_order_variables = {}
    bond_order_exclusivity = []
    bond_order_constraints = []
    bond_order_is_bool = []

    indices_set = set(indices)
    seen_bonds = set()
    for atom in considered_atoms:
        atom1_idx = atom.GetIdx()
        for atom2_idx in csr_adj_matrix[atom1_idx].indices:
            neighbor = mol.GetAtomWithIdx(int(atom2_idx))
            bond_key = (min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx))
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)
            bond = mol.GetBondBetweenAtoms(int(atom1_idx), int(atom2_idx))
            if bond is None:
                continue
            bond_is_free = bond_key in free_bond_pairs
            if atom1_idx in indices_set and atom2_idx in indices_set:
                # bond order = 0 => no bond between the atoms
                for bond_order in range(4):
                    bond_order = float(bond_order)
                    symbol = Int(f"bond_order_{atom1_idx}_{atom2_idx}_{bond_order}")
                    bond_order_variables[(atom1_idx, atom2_idx, bond_order)] = symbol
                    bond_order_variables[(atom2_idx, atom1_idx, bond_order)] = symbol
                    # Booleanity guarded/relaxed
                    if atom1_idx in excluded_atoms or atom2_idx in excluded_atoms:
                        bool_constraint = BoolVal(True)
                        bool_name = f"bool_bond_relaxed_{atom1_idx}_{atom2_idx}_{int(bond_order)}"
                    else:
                        base_bool = Or(symbol == 0, symbol == 1)
                        if enable_atom_activation:
                            bool_constraint = Implies(And(atom_activation_vars[atom1_idx], atom_activation_vars[atom2_idx]), base_bool)
                            bool_name = f"bool_bond_{atom1_idx}_{atom2_idx}_{int(bond_order)}_conditional"
                        else:
                            bool_constraint = base_bool
                            bool_name = f"bool_bond_{atom1_idx}_{atom2_idx}_{int(bond_order)}"
                    bond_order_is_bool.append(bool_constraint)
                    named_constraints[bool_name] = bool_constraint
                    # add bond length deviation penalty
                    if bond_order > 0:
                        bond_length = bond_length_lookup.get((atom1_idx, atom2_idx))
                        if bond_length is None:
                            bond_length = bond_length_lookup.get((atom2_idx, atom1_idx))
                        if bond_length is None:
                            continue
                        ideal_bond_length = ideal_bond_lengths[atom.GetSymbol()][neighbor.GetSymbol()][ideal_bond_length_order_indices[bond_order]]
                        penalty_expressions.append(_scaled_penalty(ideal_bond_length - bond_length) * symbol)
                    else:
                        penalty_expressions.append(_scaled_penalty(10) * symbol)

                # bond order exclusivity
                atom1_excluded = atom1_idx in excluded_atoms
                atom2_excluded = atom2_idx in excluded_atoms
                if atom1_excluded or atom2_excluded:
                    relaxed_constraint = And([bond_order_variables[(atom1_idx, atom2_idx, bond_order)] >= 0 for bond_order in range(4)])
                    constraint = relaxed_constraint
                    constraint_name = f"bond_relaxed_{atom1_idx}_{atom2_idx}"
                else:
                    base_constraint = Sum([bond_order_variables[(atom1_idx, atom2_idx, bond_order)] for bond_order in range(4)]) == 1
                    if enable_atom_activation:
                        constraint = Implies(And(atom_activation_vars[atom1_idx], atom_activation_vars[atom2_idx]), base_constraint)
                        constraint_name = f"bond_exclusivity_{atom1_idx}_{atom2_idx}_conditional"
                    else:
                        constraint = base_constraint
                        constraint_name = f"bond_exclusivity_{atom1_idx}_{atom2_idx}"
                bond_order_exclusivity.append(constraint)
                named_constraints[constraint_name] = constraint
            elif bond_is_free and (atom1_idx in indices_set or atom2_idx in indices_set):
                # Allow bond order to vary even if only one atom is in indices.
                for bond_order in range(4):
                    bond_order = float(bond_order)
                    symbol = Int(f"bond_order_{atom1_idx}_{atom2_idx}_{bond_order}")
                    bond_order_variables[(atom1_idx, atom2_idx, bond_order)] = symbol
                    bond_order_variables[(atom2_idx, atom1_idx, bond_order)] = symbol
                    base_bool = Or(symbol == 0, symbol == 1)
                    activation_cond = _activation_condition(atom1_idx, atom2_idx)
                    if activation_cond is not None:
                        bool_constraint = Implies(activation_cond, base_bool)
                        bool_name = f"bool_bond_{atom1_idx}_{atom2_idx}_{int(bond_order)}_conditional"
                    else:
                        bool_constraint = base_bool
                        bool_name = f"bool_bond_{atom1_idx}_{atom2_idx}_{int(bond_order)}"
                    bond_order_is_bool.append(bool_constraint)
                    named_constraints[bool_name] = bool_constraint
                    if bond_order > 0:
                        bond_length = bond_length_lookup.get((atom1_idx, atom2_idx))
                        if bond_length is None:
                            continue
                        ideal_bond_length = ideal_bond_lengths[atom.GetSymbol()][neighbor.GetSymbol()][ideal_bond_length_order_indices[bond_order]]
                        penalty_expressions.append(_scaled_penalty(ideal_bond_length - bond_length) * symbol)
                    else:
                        penalty_expressions.append(_scaled_penalty(10) * symbol)

                base_constraint = Sum([bond_order_variables[(atom1_idx, atom2_idx, bond_order)] for bond_order in range(4)]) == 1
                activation_cond = _activation_condition(atom1_idx, atom2_idx)
                if activation_cond is not None:
                    constraint = Implies(activation_cond, base_constraint)
                    constraint_name = f"bond_exclusivity_{atom1_idx}_{atom2_idx}_conditional"
                else:
                    constraint = base_constraint
                    constraint_name = f"bond_exclusivity_{atom1_idx}_{atom2_idx}"
                bond_order_exclusivity.append(constraint)
                named_constraints[constraint_name] = constraint
            elif atom1_idx in indices_set or atom2_idx in indices_set:
                # Only one atom in indices: fix bond order to the existing value
                bond_type = bond.GetBondTypeAsDouble()
                for bond_order in range(4):
                    bond_order = float(bond_order)
                    val = IntVal(int(bond_type == bond_order))
                    bond_order_variables[(atom1_idx, atom2_idx, bond_order)] = val
                    bond_order_variables[(atom2_idx, atom1_idx, bond_order)] = val

    # determine possible charges for each element
    possible_charges = {}
    for atom in considered_atoms:
        this_possible_charges = possible_charges.get(atom.GetSymbol(), set())
        for (neighbor_count, atom_charge) in possible_geometries[atom.GetSymbol()].keys():
            this_possible_charges.add(atom_charge)
        possible_charges[atom.GetSymbol()] = this_possible_charges

    # variable for each charge
    charge_variables = {}
    charge_exclusivity = []
    charge_is_bool = []
    for atom in considered_atoms:
        for possible_charge in possible_charges[atom.GetSymbol()]:
            symbol = Int(f"charge_{atom.GetIdx()}_{possible_charge}")
            charge_variables[(atom.GetIdx(), possible_charge)] = symbol
            # Booleanity guarded/relaxed
            aidx = atom.GetIdx()
            if aidx in excluded_atoms:
                c_bool = BoolVal(True)
                c_bool_name = f"bool_charge_relaxed_{aidx}_{possible_charge}"
            else:
                base_c_bool = Or(symbol == 0, symbol == 1)
                if enable_atom_activation:
                    c_bool = Implies(atom_activation_vars[aidx], base_c_bool)
                    c_bool_name = f"bool_charge_{aidx}_{possible_charge}_conditional"
                else:
                    c_bool = base_c_bool
                    c_bool_name = f"bool_charge_{aidx}_{possible_charge}"
            charge_is_bool.append(c_bool)
            named_constraints[c_bool_name] = c_bool
            penalty_expressions.append(_scaled_penalty(np.abs(possible_charge)) * symbol)
        # charge exclusivity
        atom_idx = atom.GetIdx()
        
        if atom_idx in excluded_atoms:
            # For excluded atoms, just require non-negative values (relaxed constraint)
            relaxed_constraint = And([charge_variables[(atom_idx, possible_charge)] >= 0 for possible_charge in possible_charges[atom.GetSymbol()]])
            constraint = relaxed_constraint
            constraint_name = f"charge_relaxed_{atom_idx}"
        else:
            # For non-excluded atoms, enforce strict exclusivity
            base_constraint = Sum([charge_variables[(atom_idx, possible_charge)] for possible_charge in possible_charges[atom.GetSymbol()]]) == 1
            
            if enable_atom_activation:
                # Only enforce this constraint if the atom is active
                constraint = Implies(atom_activation_vars[atom_idx], base_constraint)
                constraint_name = f"charge_exclusivity_{atom_idx}_conditional"
            else:
                constraint = base_constraint
                constraint_name = f"charge_exclusivity_{atom_idx}"
        
        charge_exclusivity.append(constraint)
        named_constraints[constraint_name] = constraint

    # variable for each possible geometry for each atom
    geometry_variables = {}
    geometry_implications = []
    geometry_exclusivity = []
    geometry_is_bool = []
    allowed_bond_constraints = []
    for atom in considered_atoms:
        for (neighbor_count, atom_charge), angle_to_info in possible_geometries[atom.GetSymbol()].items():
            for possible_angles, info in angle_to_info.items():
                geometry_variables[(atom.GetIdx(), neighbor_count, atom_charge, tuple(possible_angles))] = Int(f"geometry_{atom.GetIdx()}_{neighbor_count}_{atom_charge}_{tuple(possible_angles)}")
                # Geometry booleanity guarded/relaxed
                gvar = geometry_variables[(atom.GetIdx(), neighbor_count, atom_charge, tuple(possible_angles))]
                gaidx = atom.GetIdx()
                if gaidx in excluded_atoms:
                    g_bool = BoolVal(True)
                    g_bool_name = f"bool_geom_relaxed_{gaidx}_{neighbor_count}_{atom_charge}_{tuple(possible_angles)}"
                else:
                    base_g_bool = Or(gvar == 0, gvar == 1)
                    if enable_atom_activation:
                        g_bool = Implies(atom_activation_vars[gaidx], base_g_bool)
                        g_bool_name = f"bool_geom_{gaidx}_{neighbor_count}_{atom_charge}_{tuple(possible_angles)}_conditional"
                    else:
                        g_bool = base_g_bool
                        g_bool_name = f"bool_geom_{gaidx}_{neighbor_count}_{atom_charge}_{tuple(possible_angles)}"
                geometry_is_bool.append(g_bool)
                named_constraints[g_bool_name] = g_bool

                # determine forbidden angles
                forbidden_angle_constraints = []
                for neighbor1, neighbor2 in combinations([neighbor.GetIdx() for neighbor in atom.GetNeighbors()], 2):
                    if (neighbor1, neighbor2) in all_pairwise_bond_angles[atom.GetIdx()]:
                        angle_deviation = np.absolute(np.array(possible_angles) - all_pairwise_bond_angles[atom.GetIdx()][(neighbor1, neighbor2)])
                        if len(angle_deviation) == 0: angle_deviation = np.array([180])
                        used_deviation = np.min(angle_deviation)
                        if used_deviation > angle_tolerance:
                            forbidden_angle_constraints.append(Or(
                                bond_order_variables[(atom.GetIdx(), neighbor1, 0)] == 1,
                                bond_order_variables[(atom.GetIdx(), neighbor2, 0)] == 1
                            ))
                        else:
                            # we weight based on the angle deviation
                            weighted_deviation = used_deviation * angle_deviation_weighting
                            # Check if weighted_deviation is a valid finite number (not NaN or infinity)
                            if not np.isfinite(weighted_deviation):
                                # Skip penalty if invalid, treat as forbidden angle instead
                                forbidden_angle_constraints.append(Or(
                                    bond_order_variables[(atom.GetIdx(), neighbor1, 0)] == 1,
                                    bond_order_variables[(atom.GetIdx(), neighbor2, 0)] == 1
                                ))
                            else:
                                scaled_deviation = _scaled_penalty(weighted_deviation)
                                total_penalty_expression = Sum([(scaled_deviation * bond_order_variables[(atom.GetIdx(), neighbor1, i)]) for i in range(1, 4)])
                                penalty_expressions.append(Sum([(scaled_deviation * bond_order_variables[(atom.GetIdx(), neighbor1, i)]) for i in range(0, 4)]))

                # bond order/charge/angle implications, guarded/relaxed
                center_idx = atom.GetIdx()
                original_impl = Implies(
                    geometry_variables[(center_idx, neighbor_count, atom_charge, tuple(possible_angles))] == 1,
                    And([
                        Or([
                            And([
                                Sum([bond_order_variables[(center_idx, neighbor.GetIdx(), bond_order)] for neighbor in atom.GetNeighbors()]) == bond_count
                            for bond_order, bond_count in bond_order_info.items()])
                        for bond_order_info in info["bond_orders"]]),
                        charge_variables[(center_idx, atom_charge)] == 1,
                        *forbidden_angle_constraints
                    ])
                )
                if center_idx in excluded_atoms:
                    impl_constraint = BoolVal(True)
                    impl_name = f"geom_impl_relaxed_{center_idx}_{neighbor_count}_{atom_charge}_{tuple(possible_angles)}"
                else:
                    if enable_atom_activation:
                        impl_constraint = Implies(atom_activation_vars[center_idx], original_impl)
                        impl_name = f"geom_impl_{center_idx}_{neighbor_count}_{atom_charge}_{tuple(possible_angles)}_conditional"
                    else:
                        impl_constraint = original_impl
                        impl_name = f"geom_impl_{center_idx}_{neighbor_count}_{atom_charge}_{tuple(possible_angles)}"
                geometry_implications.append(impl_constraint)
                named_constraints[impl_name] = impl_constraint
        # geometry exclusivity
        atom_idx = atom.GetIdx()
        base_geometry_excl = Sum([geometry_variables[(atom_idx, neighbor_count, atom_charge, tuple(possible_angles))]
                                   for (neighbor_count, atom_charge), angle_to_info in possible_geometries[atom.GetSymbol()].items()
                                   for possible_angles, info in angle_to_info.items()]) == 1
        # Relax geometry exclusivity for excluded atoms; otherwise condition on activation if enabled
        if atom_idx in excluded_atoms:
            constraint = BoolVal(True)
            constraint_name = f"geometry_relaxed_{atom_idx}"
        else:
            if enable_atom_activation:
                constraint = Implies(atom_activation_vars[atom_idx], base_geometry_excl)
                constraint_name = f"geometry_exclusivity_{atom_idx}_conditional"
            else:
                constraint = base_geometry_excl
                constraint_name = f"geometry_exclusivity_{atom_idx}"
        geometry_exclusivity.append(constraint)
        named_constraints[constraint_name] = constraint

    # forbid strong angle violations

    formula = And(bond_order_exclusivity + charge_exclusivity + geometry_implications + geometry_exclusivity + bond_order_constraints + geometry_is_bool + charge_is_bool + bond_order_is_bool)
    objective = Sum(penalty_expressions)
    variable_dicts = {
        **bond_order_variables,
        **charge_variables,
        **geometry_variables
    }
    if enable_atom_activation:
        return formula, objective, variable_dicts, named_constraints, atom_activation_vars
    else:
        return formula, objective, variable_dicts, named_constraints

def _resolve_unsat_cores_iteratively(
    mol,
    indices,
    all_pairwise_bond_angles,
    bond_indices,
    bond_lengths,
    csr_adj_matrix,
    free_bond_pairs=None,
):
    """
    Iteratively resolve unsat cores by softening constraints for problematic atoms until SAT is achieved.
    Returns the set of atoms that need softened constraints to make the system SAT.
    """
    from z3 import Solver, Optimize, sat, unsat
    
    all_atoms = set(indices)
    softened_atoms = set()
    
    while True:
        if os.environ.get('XB_DEBUG_SAT') == '1':
            print(f"\n--- Iteration with {len(softened_atoms)} softened atoms ---")
        
        # Build formula with current softened atoms
        formula, objective, variable_dict, named_constraints, atom_activation_vars = _construct_SMT_formulation(
            mol,
            list(all_atoms),
            all_pairwise_bond_angles,
            bond_indices,
            bond_lengths,
            csr_adj_matrix,
            enable_atom_activation=True,
            excluded_atoms=softened_atoms,
            free_bond_pairs=free_bond_pairs,
        )
        
        # Assumption-based SAT check so unsat core contains only atom activations
        s = Solver()
        s.add(formula)
        assumptions = []
        for atom_idx in all_atoms:
            if atom_idx not in softened_atoms and atom_idx in atom_activation_vars:
                assumptions.append(atom_activation_vars[atom_idx])
        
        result = s.check(assumptions)
        if os.environ.get('XB_DEBUG_SAT') == '1':
            print(f"SAT check result: {result}")
        
        if result == sat:
            if os.environ.get('XB_DEBUG_SAT') == '1':
                print("✓ System is now SAT! All constraints are compatible.")
            break
        else:
            # Extract problematic atoms from unsat core (subset of assumptions)
            unsat_core = s.unsat_core()
            if os.environ.get('XB_DEBUG_SAT') == '1':
                print(f"Unsat core: {unsat_core}")
            
            problematic_atoms = set()
            for lit in unsat_core:
                lit_name = str(lit)
                if lit_name.startswith("atom_active_"):
                    atom_idx = int(lit_name.split("_")[2])
                    problematic_atoms.add(atom_idx)
            
            if problematic_atoms:
                atom_symbols = [mol.GetAtomWithIdx(idx).GetSymbol() + str(idx) for idx in sorted(problematic_atoms)]
                if os.environ.get('XB_DEBUG_SAT') == '1':
                    print(f"Softening constraints for problematic atoms: {atom_symbols}")
                
                # Add problematic atoms to softened set
                softened_atoms.update(problematic_atoms)
            else:
                if os.environ.get('XB_DEBUG_SAT') == '1':
                    print("No atom activation constraints in unsat core - this means the constraints themselves are contradictory")
                    print("This usually happens when the system is fundamentally unsolvable even with relaxed constraints")
                    print("Breaking out of constraint softening loop")
                break
    
    if os.environ.get('XB_DEBUG_SAT') == '1':
        print(f"\n=== CONSTRAINT SOFTENING COMPLETE ===")
        print(f"Softened atoms: {len(softened_atoms)}")
        if softened_atoms:
            softened_symbols = [mol.GetAtomWithIdx(idx).GetSymbol() + str(idx) for idx in sorted(softened_atoms)]
            print(f"Softened atom symbols: {softened_symbols}")
        print(f"Strict atoms: {len(all_atoms - softened_atoms)}")
    
    return softened_atoms, all_atoms - softened_atoms

def _get_connected_subgraphs(csr_adj_matrix, indices):
    """
    Given a set of atom indices, identifies the connected subgraphs induced by those indices,
    using a sparse adjacency matrix for the full molecule.

    Args:
        csr_adj_matrix (scipy.sparse.csr_matrix): Full-molecule adjacency matrix.
        indices (list or set of int): The atom indices to consider.

    Returns:
        list of sets of int: Each set is a connected component of atom indices.
    """
    indices_set = set(indices)
    if not indices_set:
        return []

    indices_list = sorted(indices_set)
    sub_adj = csr_adj_matrix[indices_list][:, indices_list]

    n_components, labels = connected_components(sub_adj, directed=False, return_labels=True)
    components = [set() for _ in range(n_components)]
    for pos, label in enumerate(labels):
        components[label].add(indices_list[pos])
    return components


def _iter_cc_bridge_splits(subgraph, mol, csr_adj_matrix):
    indices_set = set(subgraph)
    if len(indices_set) <= _MIN_SUBGRAPH_SIZE:
        return []

    adjacency = {idx: [] for idx in indices_set}
    for idx in indices_set:
        adjacency[idx] = [int(nbr) for nbr in csr_adj_matrix[idx].indices if nbr in indices_set]

    time = 0
    disc = {}
    low = {}
    parent = {}
    bridges = []

    def dfs(u):
        nonlocal time
        time += 1
        disc[u] = low[u] = time
        for v in adjacency[u]:
            if v not in disc:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((u, v))
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for node in indices_set:
        if node not in disc:
            dfs(node)

    candidates = []
    for u, v in bridges:
        if mol.GetAtomWithIdx(u).GetSymbol() != "C" or mol.GetAtomWithIdx(v).GetSymbol() != "C":
            continue
        comp_a = _component_without_edge(adjacency, u, v)
        if not comp_a:
            continue
        comp_b = indices_set - comp_a
        if not comp_b:
            continue
        diff = abs(len(comp_a) - len(comp_b))
        bond_key = (min(u, v), max(u, v))
        candidates.append((diff, bond_key, comp_a, comp_b))

    candidates.sort(key=lambda item: item[0])
    return [(bond_key, comp_a, comp_b) for _, bond_key, comp_a, comp_b in candidates]


def _component_without_edge(adjacency, start, blocked_neighbor):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nbr in adjacency[node]:
            if (node == start and nbr == blocked_neighbor) or (node == blocked_neighbor and nbr == start):
                continue
            if nbr not in visited:
                stack.append(nbr)
    return visited


def _get_bond_order_from_result(result, bond_key):
    _, _, new_bond_orders, _ = result
    atom1, atom2 = bond_key
    if (atom1, atom2) in new_bond_orders:
        return new_bond_orders[(atom1, atom2)]
    if (atom2, atom1) in new_bond_orders:
        return new_bond_orders[(atom2, atom1)]
    return None


def _merge_subgraph_results(result_a, result_b, bond_key):
    subgraph_a, softened_a, bond_orders_a, charges_a = result_a
    subgraph_b, softened_b, bond_orders_b, charges_b = result_b

    merged_bond_orders = dict(bond_orders_a)
    for key, value in bond_orders_b.items():
        if key in merged_bond_orders and merged_bond_orders[key] != value:
            raise ValueError(f"Conflicting bond order for {key}: {merged_bond_orders[key]} vs {value}")
        merged_bond_orders[key] = value

    merged_charges = dict(charges_a)
    for key, value in charges_b.items():
        if key in merged_charges and merged_charges[key] != value:
            raise ValueError(f"Conflicting charge for atom {key}: {merged_charges[key]} vs {value}")
        merged_charges[key] = value

    merged_subgraph = set(subgraph_a) | set(subgraph_b)
    merged_softened = set(softened_a) | set(softened_b)
    return merged_subgraph, merged_softened, merged_bond_orders, merged_charges

def _get_all_pairwise_bond_angles(mol):
    positions = position(mol)
    pairwise_angles = {}
    for atom in mol.GetAtoms():
        center_atom_angles = {}
        for neighbor1, neighbor2 in combinations([neighbor.GetIdx() for neighbor in atom.GetNeighbors()], 2):
            angle = calculate_angle(positions[neighbor1], positions[atom.GetIdx()], positions[neighbor2])
            center_atom_angles[(neighbor1, neighbor2)] = angle
            center_atom_angles[(neighbor2, neighbor1)] = angle
        pairwise_angles[atom.GetIdx()] = center_atom_angles
    return pairwise_angles

def _get_violating_atom_clusters(mol, all_pairwise_bond_angles, angle_tolerance = 10, add_hydrogens = False):
    positions = position(mol)

    violating_atoms = set()
    considerable_geometries = {}
    # id violating atoms
    for atom in mol.GetAtoms():
        pairwise_bond_angles = all_pairwise_bond_angles[atom.GetIdx()]
        found_angles = set(pairwise_bond_angles.values())

        # get geometry: consider any charge state for this neighbor count
        symbol = atom.GetSymbol()
        neighbor_count = len(atom.GetNeighbors())
        all_geometries = possible_geometries.get(symbol, {})
        this_possible_geometries = {}
        for (geom_neighbors, geom_charge), geom_info in all_geometries.items():
            if geom_neighbors == neighbor_count:
                this_possible_geometries.update(geom_info)
        if len(this_possible_geometries) == 0:
            violating_atoms.add(atom.GetIdx())
            continue

        geometry_violation = True
        #print(f"Checking atom {atom.GetSymbol()}{atom.GetIdx()} with {len(atom.GetNeighbors())} neighbors and charge {atom.GetFormalCharge()} and possible_geometries {this_possible_geometries}")
        for possible_angles, possible_geometry_info in this_possible_geometries.items():
            # check if all of the angles are within the allowed range
            if not ((len(found_angles) == 0) and (len(possible_angles) == 0)):
                try:
                    angle_deviations = np.max(np.absolute(np.fromiter(found_angles, float, len(found_angles))[None, :] - np.array(possible_angles)[:, None]))
                except ValueError:
                    # if the found angles are not in the possible angles, this geometry is not possible
                    #print(f"Failed for atom {atom.GetSymbol()}{atom.GetIdx()} and possible angles {possible_angles}: as not all angles {found_angles} are in {possible_angles}")
                    continue
                if angle_deviations > angle_tolerance:
                    # if not, this geometry is not possible
                    #print(f"Failed for atom {atom.GetSymbol()}{atom.GetIdx()} and possible angles {possible_angles}: as not all angles {found_angles} are within the allowed range")
                    continue

            # check bond orders
            bond_order_violation = True
            for bond_order in possible_geometry_info.get("bond_orders", []):
                bond_order = dict(bond_order)
                for bond in atom.GetBonds():
                    try:
                        #print(bond.GetBondTypeAsDouble())
                        bond_order[bond.GetBondTypeAsDouble()] -= 1
                    except KeyError:
                        #print(f"Failed for atom {atom.GetSymbol()}{atom.GetIdx()} with bond {bond.GetIdx()} as {bond.GetBondTypeAsDouble()} is not in {bond_order}")
                        continue
                if not any(bond_order.values()):
                    bond_order_violation = False
                    break
                #print(f"Failed for atom {atom.GetSymbol()}{atom.GetIdx()} with bond {bond.GetIdx()} as bond order violation is {bond_order}")
            if bond_order_violation:
                #print(f"Failed for atom {atom.GetSymbol()}{atom.GetIdx()} and possible angles {possible_angles}: as no valid bond order was found")
                continue
            # if we reach this point, the geometry is possible
            geometry_violation = False
            considerable_geometries[atom.GetIdx()] = possible_angles
        
        if geometry_violation:
            #print(f"Failed for atom {atom.GetSymbol()}{atom.GetIdx()}")
            violating_atoms.add(atom.GetIdx())
    return violating_atoms