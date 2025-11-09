AggregationFunctions

purpose
Server-side weight aggregation for federated learning. This folder currently contains a single implementation file with several aggregation rules (FedAvg, FedProx-as-FedAvg, FedAdam) and a small registry.

file

* aggregation_algorithms.py
  Implements the aggregation functions that combine client encoder updates into a new global state.

apis
All public functions share the same signature and return type:

```
def aggregate_xxx(
    global_model: torch.nn.Module,
    client_states: list[dict[str, torch.Tensor]],
    client_weights: list[float],
    server_state: dict[str, any] | None = None,
    **kwargs
) -> tuple[dict[str, torch.Tensor], dict[str, any]]
```

inputs

* global_model: model whose state_dict defines the parameter keys and shapes.
* client_states: list of state_dict-like mappings (each from one site), typically encoder weights after local training.
* client_weights: list of non-negative weights (often number of training samples per site).
* server_state: optional mutable state for optimisers that require momentum/variance tracking (used by FedAdam).
* kwargs: optional algorithm-specific hyperparameters.

outputs

* new_global_state: aggregated state_dict for the server to broadcast next round.
* new_server_state: updated optimiser state (empty for FedAvg/FedProx; contains m, v, t for FedAdam).

functions

* aggregate_fedavg(...)
  Weighted average of client states using client_weights / sum(weights). Falls back to a selective per-key averaging when tensors mismatch via `_selective_fedavg`.

* aggregate_fedprox(...)
  Alias of FedAvg in this simplified setting (same aggregation rule; Prox term affects local training, not server averaging).

* aggregate_fedadam(..., eta=0.1, beta1=0.9, beta2=0.99, eps=1e-8)
  Server-side Adam using the aggregated target as a pseudo-gradient step from the current global weights.
  Maintains in server_state:

  * m: first moment dict (per-parameter)
  * v: second moment dict (per-parameter)
  * t: step counter

internal helper

* _selective_fedavg(client_states, client_weights)
  Groups by parameter key and averages only among clients that provided that key. Useful when some clients miss certain params.

registry

```
AGG_FNS = {
  "fedavg":  aggregate_fedavg,
  "fedprox": aggregate_fedprox,
  "fedadam": aggregate_fedadam
}
```

minimal usage

```
from AggregationFunctions.aggregation_algorithms import AGG_FNS

agg_fn = AGG_FNS["fedavg"]  # or "fedadam"
new_state, server_state = agg_fn(
    global_model=server_model,
    client_states=[st1, st2, st3],
    client_weights=[n1, n2, n3],
    server_state=prev_server_state  # None for first round
)

server_model.load_state_dict(new_state, strict=True)
```

tips

* keep parameter names and shapes identical across all client states; if you change the model structure mid-run, expect `_selective_fedavg` to kick in.
* use sample counts as client_weights to approximate FedAvg.
* when switching between algorithms mid-experiment, reset server_state to avoid mixing optimiser moments.
