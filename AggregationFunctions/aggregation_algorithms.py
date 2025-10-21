# federated_agg_experiment.py (simple version)

# ===== imports & fixed paths =====
import os, sys, json, time, random
from collections import defaultdict
from typing import Dict, List, Tuple, Any


import numpy as np
import torch
import torch.nn.functional as F

# ==================== Aggregations ====================
StateDict = Dict[str, torch.Tensor]

def _selective_fedavg(client_states: List[StateDict], client_weights: List[float]) -> StateDict:
    grouped = defaultdict(list)
    for st, w in zip(client_states, client_weights):
        for k, v in st.items(): grouped[k].append((v, float(w)))
    out: StateDict = {}
    for k, ups in grouped.items():
        tw = sum(w for _, w in ups); acc = None
        for v, w in ups:
            scaled = v * (w / max(1.0, tw))
            acc = scaled if acc is None else acc + scaled
        out[k] = acc
    return out




def aggregate_fedavg(global_model: torch.nn.Module, client_states: List[StateDict],
                     client_weights: List[float], server_state: Dict[str, Any] | None = None, **_) -> Tuple[StateDict, Dict[str, Any]]:
    try:
        total = float(sum(client_weights))
        keys = client_states[0].keys()
        agg = {k: torch.zeros_like(client_states[0][k]) for k in keys}
        for st, w in zip(client_states, client_weights):
            scale = float(w) / max(1.0, total)
            for k in keys: agg[k] += st[k] * scale
        return agg, (server_state or {})
    except Exception:
        return _selective_fedavg(client_states, client_weights), (server_state or {})





def aggregate_fedprox(global_model, client_states, client_weights, server_state=None, **_):
    return aggregate_fedavg(global_model, client_states, client_weights, server_state)






def aggregate_fedadam(global_model, client_states, client_weights, server_state=None,
                      eta=0.1, beta1=0.9, beta2=0.99, eps=1e-8, **_) -> Tuple[StateDict, Dict[str, Any]]:
    if server_state is None: server_state = {}
    target, _ = aggregate_fedavg(global_model, client_states, client_weights, server_state)
    if server_state.get("t") is None:
        server_state["m"] = {k: torch.zeros_like(v) for k, v in target.items()}
        server_state["v"] = {k: torch.zeros_like(v) for k, v in target.items()}
        server_state["t"] = 0
    cur = global_model.state_dict(); m, v = server_state["m"], server_state["v"]; t = server_state["t"] + 1
    new_state: StateDict = {}
    for k in target.keys():
        g = target[k] - cur[k]
        m[k] = beta1 * m[k] + (1 - beta1) * g
        v[k] = beta2 * v[k] + (1 - beta2) * (g * g)
        m_hat = m[k] / (1 - (beta1 ** t)); v_hat = v[k] / (1 - (beta2 ** t))
        new_state[k] = cur[k] + eta * m_hat / (torch.sqrt(v_hat) + eps)
    server_state["t"] = t
    return new_state, server_state



AGG_FNS = {"fedavg": aggregate_fedavg, "fedprox": aggregate_fedprox, "fedadam": aggregate_fedadam}