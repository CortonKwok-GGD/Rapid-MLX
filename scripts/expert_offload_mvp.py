#!/usr/bin/env python3
"""MoE Expert Offloading for Apple Silicon — model-agnostic engine.

Runs large MoE models on memory-constrained Macs by keeping shared layers
in RAM and streaming expert weights from SSD via an LRU cache.

Supports two expert weight formats:
  - Per-expert: .experts.{N}.w1.weight (Mixtral, DBRX, etc.)
  - Stacked: .switch_mlp.gate_proj.weight with shape (num_experts, ...) (Qwen3.6, etc.)

v3 optimizations:
  1. Single eval per MoE block (not 3 per SwitchLinear)
  2. Pre-cached remap shared across projections
  3. Model-agnostic: auto-detects SwitchGLU/SwitchLinear in any mlx-lm model
  4. Auto-adaptive cache sizing based on available RAM

Usage:
    python scripts/expert_offload_mvp.py --model mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit
    python scripts/expert_offload_mvp.py --model mlx-community/Qwen3.6-35B-A3B-8bit --serve
"""

import argparse
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Expert LRU Cache
# ---------------------------------------------------------------------------

class ExpertCache:
    """LRU cache keyed by (layer, expert_id). Each entry holds all projections.

    On first init, splits the model's safetensors into per-expert files
    for fast mx.load() access (no numpy intermediate).

    Supports two source formats:
      - per_expert: weights named .experts.{N}.w1.weight (one expert per key)
      - stacked: weights named .switch_mlp.gate_proj.weight with dim 0 = num_experts
    """

    def __init__(self, model_path: str, weight_map: dict,
                 expert_key_pattern: str, proj_map: dict,
                 num_layers: int, num_experts: int,
                 max_experts: int = 256, format: str = "per_expert",
                 stacked_key_template: str | None = None):
        self.model_path = Path(model_path)
        self.weight_map = weight_map
        self.expert_key_pattern = expert_key_pattern
        self.proj_map = proj_map
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts
        self.max_experts = max_experts
        self.format = format
        self.stacked_key_template = stacked_key_template
        self.cache: OrderedDict[tuple, dict] = OrderedDict()
        self.hits = 0
        self.misses = 0

        # Pre-split expert weights into per-expert files for fast loading
        self.expert_dir = self.model_path / ".expert_cache"
        self._ensure_expert_files()

    def _ensure_expert_files(self):
        """Split safetensors into per-expert files if not already done."""
        marker = self.expert_dir / ".done"
        if marker.exists():
            return

        self.expert_dir.mkdir(exist_ok=True)
        print(f"  Splitting expert weights (one-time)...", flush=True)
        t0 = time.perf_counter()

        if self.format == "stacked":
            count = self._split_stacked()
        else:
            count = self._split_per_expert()

        marker.touch()
        print(f"  Split complete: {count} expert files in {time.perf_counter() - t0:.1f}s")

    def _split_per_expert(self) -> int:
        """Split per-expert format (Mixtral-style .experts.N. keys)."""
        from safetensors import safe_open
        from safetensors.numpy import save_file

        groups: dict[tuple, dict] = {}
        for key in self.weight_map:
            if ".experts." not in key:
                continue
            m = re.search(r'\.(\d+)\..*\.experts\.(\d+)\.(.*)', key)
            if not m:
                continue
            layer, expert = int(m.group(1)), int(m.group(2))
            groups.setdefault((layer, expert), {})[key] = None

        file_handles = {}
        for (layer, expert), keys in sorted(groups.items()):
            out_path = self.expert_dir / f"L{layer}_E{expert}.safetensors"
            if out_path.exists():
                continue

            tensors = {}
            for key in keys:
                src_file = self.weight_map[key]
                src_path = str(self.model_path / src_file)
                if src_path not in file_handles:
                    file_handles[src_path] = safe_open(src_path, framework="numpy")
                tensors[key] = file_handles[src_path].get_tensor(key)

            save_file(tensors, str(out_path))

        for fh in file_handles.values():
            del fh

        return len(groups)

    def _split_stacked(self) -> int:
        """Split stacked format (Qwen3.6-style switch_mlp.proj.weight with dim0=num_experts)."""
        # Group stacked keys by (layer, proj, component)
        # Template: "prefix.{layer}.mlp.switch_mlp.{proj}.{component}"
        stacked_keys = {}
        for key in self.weight_map:
            if ".switch_mlp." not in key:
                continue
            m = re.search(r'\.(\d+)\.\w+\.switch_mlp\.(\w+)\.(\w+)$', key)
            if not m:
                continue
            layer = int(m.group(1))
            proj = m.group(2)  # gate_proj, up_proj, down_proj
            component = m.group(3)  # weight, scales, biases
            stacked_keys.setdefault(layer, {})[f"{proj}.{component}"] = key

        count = 0
        # Process one layer at a time to limit memory usage
        for layer in sorted(stacked_keys.keys()):
            print(f"    Layer {layer}/{max(stacked_keys.keys())}...", end="\r", flush=True)
            keys_in_layer = stacked_keys[layer]

            # Load all stacked tensors for this layer
            files_needed = {}
            for pc, key in keys_in_layer.items():
                fname = self.weight_map[key]
                files_needed.setdefault(fname, []).append((pc, key))

            stacked_tensors = {}
            for fname, items in files_needed.items():
                fpath = str(self.model_path / fname)
                loaded = mx.load(fpath)
                for pc, key in items:
                    stacked_tensors[pc] = loaded[key]
                del loaded

            # Materialize to avoid lazy-load memory issues
            mx.eval(*stacked_tensors.values())

            # Slice and save per-expert files
            for expert_id in range(self.num_experts_per_layer):
                out_path = self.expert_dir / f"L{layer}_E{expert_id}.safetensors"
                if out_path.exists():
                    count += 1
                    continue

                expert_tensors = {}
                for pc, stacked in stacked_tensors.items():
                    proj, component = pc.split(".")
                    # Use synthetic per-expert key naming
                    ekey = self.expert_key_pattern.format(
                        layer=layer, expert=expert_id, wname=proj
                    )
                    expert_tensors[f"{ekey}.{component}"] = stacked[expert_id]

                mx.save_safetensors(str(out_path), expert_tensors)
                count += 1

            # Free stacked tensors before next layer
            del stacked_tensors

        print()  # clear \r
        return count

    def ensure_experts(self, layer: int, expert_ids: list[int]):
        for eid in expert_ids:
            key = (layer, eid)
            if key in self.cache:
                self.hits += 1
                self.cache.move_to_end(key)
            else:
                self.misses += 1
                self._load_expert(layer, eid)

    def get_proj(self, layer: int, expert_id: int, proj_name: str):
        return self.cache[(layer, expert_id)][proj_name]

    def _load_expert(self, layer: int, expert_id: int):
        while len(self.cache) >= self.max_experts:
            self.cache.popitem(last=False)

        expert_file = self.expert_dir / f"L{layer}_E{expert_id}.safetensors"
        all_tensors = mx.load(str(expert_file))

        entry = {}
        for proj_name, w_name in self.proj_map.items():
            prefix = self.expert_key_pattern.format(
                layer=layer, expert=expert_id, wname=w_name
            )
            entry[proj_name] = (
                all_tensors[f"{prefix}.weight"],
                all_tensors[f"{prefix}.scales"],
                all_tensors[f"{prefix}.biases"],
            )

        self.cache[(layer, expert_id)] = entry

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self):
        return (
            f"cache={len(self.cache)}/{self.max_experts} "
            f"hits={self.hits} misses={self.misses} "
            f"rate={self.hit_rate:.1%}"
        )


# ---------------------------------------------------------------------------
# Offloaded SwitchGLU — replaces the entire SwitchGLU, evals indices ONCE
# ---------------------------------------------------------------------------

class OffloadedSwitchGLU(nn.Module):
    """Drop-in replacement for SwitchGLU that loads experts on demand.

    Key optimization: evaluates indices ONCE and reuses the remap for all
    3 projections (gate_proj, up_proj, down_proj), reducing mx.eval calls
    from 3 to 1 per layer.
    """

    def __init__(self, input_dims, hidden_dims, num_experts,
                 group_size, bits, cache: ExpertCache, layer_idx: int):
        super().__init__()
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._num_experts = num_experts
        self.group_size = group_size
        self.bits = bits
        self._cache = cache
        self._layer_idx = layer_idx
        from mlx_lm.models.switch_layers import SwiGLU
        self.activation = SwiGLU()

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # Check if ALL experts for THIS layer are already cached.
        layer_all_cached = all(
            (self._layer_idx, eid) in self._cache.cache
            for eid in range(self._num_experts)
        )

        if layer_all_cached:
            unique_experts = list(range(self._num_experts))
            new_indices = indices
        else:
            mx.eval(indices)
            idx_list = indices.reshape(-1).tolist()
            unique_experts = sorted(set(idx_list))
            self._cache.ensure_experts(self._layer_idx, unique_experts)

            remap = {old: new for new, old in enumerate(unique_experts)}
            new_idx_flat = [remap[i] for i in idx_list]
            new_indices = mx.array(new_idx_flat, dtype=mx.uint32).reshape(indices.shape)

        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
        do_sort = indices.size >= 64
        idx = new_indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, new_indices)

        x_up = self._proj_call("up_proj", x, idx, unique_experts, do_sort)
        x_gate = self._proj_call("gate_proj", x, idx, unique_experts, do_sort)
        x = self._proj_call("down_proj", self.activation(x_up, x_gate), idx, unique_experts, do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, new_indices.shape)

        return x.squeeze(-2)

    def _proj_call(self, proj_name, x, indices, unique_experts, sorted_indices):
        cache_key = (self._layer_idx, proj_name)
        if cache_key in _STACKED_CACHE:
            w, s, b = _STACKED_CACHE[cache_key]
        else:
            w_list, s_list, b_list = [], [], []
            for eid in unique_experts:
                w, s, b = self._cache.get_proj(self._layer_idx, eid, proj_name)
                w_list.append(w)
                s_list.append(s)
                b_list.append(b)
            w = mx.stack(w_list)
            s = mx.stack(s_list)
            b = mx.stack(b_list)
            if len(unique_experts) == self._num_experts:
                _STACKED_CACHE[cache_key] = (w, s, b)

        return mx.gather_qmm(
            x, w, s, b,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            sorted_indices=sorted_indices,
        )

# Pre-stacked tensor cache: {(layer, proj_name): (weight, scales, biases)}
_STACKED_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Model-agnostic detection and patching
# ---------------------------------------------------------------------------

def _detect_expert_pattern(weight_map: dict):
    """Auto-detect expert weight naming pattern from the safetensors index.

    Returns (pattern, proj_map, expert_layers, num_experts, format).
    format is "per_expert" or "stacked".
    """
    # Try per-expert format first (.experts.N.)
    for k in weight_map:
        if ".experts." in k and k.endswith(".weight"):
            result = _detect_per_expert_pattern(weight_map)
            return (*result, "per_expert")

    # Try stacked format (.switch_mlp.proj.weight with dim 0 = num_experts)
    for k in weight_map:
        if ".switch_mlp." in k and k.endswith(".weight"):
            result = _detect_stacked_pattern(weight_map)
            return (*result, "stacked")

    raise ValueError("No expert weights found in model — not a MoE model?")


def _detect_per_expert_pattern(weight_map: dict):
    """Detect per-expert naming pattern (Mixtral-style)."""
    sample = next(k for k in weight_map if ".experts." in k and k.endswith(".weight"))

    m = re.match(r"(.+?)\.(\d+)\.(.+?)\.experts\.(\d+)\.(\w+)\.weight", sample)
    if not m:
        raise ValueError(f"Cannot parse expert weight pattern from: {sample}")

    prefix = m.group(1)
    moe_block = m.group(3)

    pattern = f"{prefix}.{{layer}}.{moe_block}.experts.{{expert}}.{{wname}}"

    proj_names = set()
    for k in weight_map:
        m2 = re.match(rf"{re.escape(prefix)}\.(\d+)\.{re.escape(moe_block)}\.experts\.(\d+)\.(\w+)\.", k)
        if m2:
            proj_names.add(m2.group(3))

    proj_map = {}
    for pn in sorted(proj_names):
        if pn in ("w1", "gate_proj"):
            proj_map["gate_proj"] = pn
        elif pn in ("w2", "down_proj"):
            proj_map["down_proj"] = pn
        elif pn in ("w3", "up_proj"):
            proj_map["up_proj"] = pn
        else:
            proj_map[pn] = pn

    layers = set()
    experts = set()
    for k in weight_map:
        m3 = re.match(rf"{re.escape(prefix)}\.(\d+)\.{re.escape(moe_block)}\.experts\.(\d+)\.", k)
        if m3:
            layers.add(int(m3.group(1)))
            experts.add(int(m3.group(2)))

    print(f"  Detected per-expert MoE pattern: {pattern}")
    print(f"  Projections: {proj_map}")
    print(f"  Layers: {len(layers)}, Experts/layer: {len(experts)}")

    return pattern, proj_map, sorted(layers), len(experts)


def _detect_stacked_pattern(weight_map: dict):
    """Detect stacked expert format (Qwen3.6-style switch_mlp.proj.weight)."""
    # Find the prefix pattern: e.g. "language_model.model.layers.{N}.mlp.switch_mlp.{proj}.weight"
    sample = next(k for k in weight_map if ".switch_mlp." in k and k.endswith(".weight"))
    m = re.match(r"(.+?)\.(\d+)\.(\w+)\.switch_mlp\.(\w+)\.weight", sample)
    if not m:
        raise ValueError(f"Cannot parse stacked pattern from: {sample}")

    prefix = m.group(1)  # e.g. "language_model.model.layers"
    mlp_name = m.group(3)  # e.g. "mlp"

    # Synthetic per-expert key pattern for cache files
    pattern = f"{prefix}.{{layer}}.{mlp_name}.switch_mlp.e.{{expert}}.{{wname}}"

    # Detect projections
    proj_names = set()
    for k in weight_map:
        m2 = re.search(r'\.switch_mlp\.(\w+)\.weight$', k)
        if m2:
            proj_names.add(m2.group(1))

    proj_map = {pn: pn for pn in sorted(proj_names)}

    # Detect layers
    layers = set()
    for k in weight_map:
        if ".switch_mlp." in k:
            m3 = re.search(rf'{re.escape(prefix)}\.(\d+)\.', k)
            if m3:
                layers.add(int(m3.group(1)))

    # Get num_experts from config (can't determine from weight_map alone for stacked)
    # We'll detect it during loading from tensor shapes
    num_experts = None

    print(f"  Detected stacked MoE pattern: {prefix}.{{N}}.{mlp_name}.switch_mlp.{{proj}}")
    print(f"  Projections: {proj_map}")
    print(f"  Layers with experts: {len(layers)}")

    return pattern, proj_map, sorted(layers), num_experts


def _find_switch_glus(model):
    """Walk the model tree and find all SwitchGLU instances with their paths."""
    from mlx_lm.models.switch_layers import SwitchGLU
    results = []

    def _walk(module, path=""):
        if isinstance(module, SwitchGLU):
            results.append((path, module))
            return
        for name, child in module.children().items():
            if isinstance(child, nn.Module):
                _walk(child, f"{path}.{name}" if path else name)
            elif isinstance(child, list):
                for i, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        _walk(item, f"{path}.{name}.{i}" if path else f"{name}.{i}")

    _walk(model)
    return results


def _make_cache(model):
    """Create the right cache for the model (handles hybrid attention+SSM models)."""
    if hasattr(model, 'make_cache'):
        return model.make_cache()
    # Fallback: plain KVCache for each layer
    from mlx_lm.models.cache import KVCache
    return [KVCache() for _ in _get_layers(model)]


def _get_layers(model):
    """Find the layers list in the model, handling wrappers like VL models."""
    # Direct: model.layers
    if hasattr(model, 'layers'):
        return model.layers
    # VL wrapper: model.language_model.model.layers
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            return lm.model.layers
        if hasattr(lm, 'layers'):
            return lm.layers
    # model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    raise AttributeError("Cannot find layers in model")


# ---------------------------------------------------------------------------
# Auto-adaptive cache sizing
# ---------------------------------------------------------------------------

def _auto_cache_size(shared_bytes: int, num_experts_total: int, expert_size_estimate: int):
    """Determine cache size based on available RAM."""
    import subprocess
    result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    total_ram = int(result.stdout.strip())

    os_overhead = 4 * 1024**3  # 4 GB for OS
    kv_reserve = 2 * 1024**3   # 2 GB for KV cache
    available = total_ram - os_overhead - shared_bytes - kv_reserve

    # Use 60% of available for expert cache
    cache_budget = int(available * 0.6)
    max_experts = max(64, cache_budget // max(expert_size_estimate, 1))
    max_experts = min(max_experts, num_experts_total)

    print(f"  Total RAM: {total_ram / 1e9:.1f} GB")
    print(f"  Available for cache: {cache_budget / 1e9:.1f} GB")
    print(f"  Auto cache size: {max_experts} experts "
          f"(~{max_experts * expert_size_estimate / 1e9:.1f} GB)")

    return max_experts


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_offloaded(model_path: str, max_cached_experts: int | None = None):
    """Load any mlx-lm MoE model with expert weights offloaded to disk."""
    model_path = Path(model_path)

    with open(model_path / "config.json") as f:
        config = json.load(f)

    quant_config = config.get("quantization", {})
    group_size = quant_config.get("group_size", 64)
    bits = quant_config.get("bits", 4)

    # Load weight map
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = {k: "model.safetensors"
                      for k in mx.load(str(model_path / "model.safetensors")).keys()}

    # Auto-detect MoE pattern
    pattern, proj_map, expert_layers, num_experts, fmt = _detect_expert_pattern(weight_map)

    # For stacked format, get num_experts from config
    if num_experts is None:
        text_cfg = config.get("text_config", config)
        num_experts = text_cfg.get("num_experts", text_cfg.get("num_local_experts"))
        if num_experts is None:
            raise ValueError("Cannot determine num_experts from config")
        print(f"  Experts per layer: {num_experts} (from config)")

    # Separate expert vs shared weights
    expert_keys = set()
    shared_keys = set()
    for k in weight_map:
        if fmt == "stacked":
            is_expert = ".switch_mlp." in k
        else:
            is_expert = ".experts." in k
        if is_expert:
            expert_keys.add(k)
        else:
            shared_keys.add(k)

    print(f"  Weight split: {len(shared_keys)} shared, {len(expert_keys)} expert ({fmt} format)")

    # Load ONLY shared weights using selective tensor access
    shared_weights = {}
    from safetensors import safe_open
    files_needed = {}
    for k in shared_keys:
        fname = weight_map[k]
        files_needed.setdefault(fname, []).append(k)

    for fname, keys in sorted(files_needed.items()):
        fpath = str(model_path / fname)
        print(f"  Loading {len(keys)} shared tensors from {fname}...")
        # Use mx.load for files that may contain bf16 tensors
        loaded = mx.load(fpath)
        for k in keys:
            if k in loaded:
                shared_weights[k] = loaded[k]
        del loaded

    mx.eval(*shared_weights.values())

    shared_bytes = sum(v.nbytes for v in shared_weights.values())
    print(f"  Shared weights: {shared_bytes / 1e9:.2f} GB")

    # Estimate expert size for cache sizing
    # Per expert: ~3 projections * weight+scales+biases
    text_cfg = config.get("text_config", config)
    hidden_size = text_cfg.get("hidden_size", config.get("hidden_size", 4096))
    moe_hidden = text_cfg.get("moe_intermediate_size",
                              text_cfg.get("intermediate_size",
                                           config.get("intermediate_size", 14336)))
    # Rough estimate based on quantized sizes
    expert_size_est = (hidden_size * moe_hidden * 3 * bits) // (8 * 1024) * 1024  # bytes
    expert_size_est = max(expert_size_est, 1024 * 1024)  # at least 1MB
    total_expert_entries = len(expert_layers) * num_experts

    print(f"  Expert size estimate: {expert_size_est / 1e6:.1f} MB/expert, {total_expert_entries} total")

    # Build model
    model_type = config.get("model_type", "")
    model_module = __import__(f"mlx_lm.models.{model_type}", fromlist=["Model", "ModelArgs"])
    ModelClass = model_module.Model
    ModelArgsClass = model_module.ModelArgs

    model_args = ModelArgsClass.from_dict(config)
    model = ModelClass(model_args)

    # Quantize shared layers only (skip switch_mlp/expert layers)
    def class_predicate(path, module):
        if "switch_mlp" in path or "switch_glu" in path or "experts" in path:
            return False
        return f"{path}.scales" in shared_weights

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=class_predicate)

    # Auto-size cache
    if max_cached_experts is None:
        max_cached_experts = _auto_cache_size(shared_bytes, total_expert_entries, expert_size_est)
    else:
        print(f"  Manual cache size: {max_cached_experts} experts")

    # Create cache
    cache = ExpertCache(
        str(model_path), weight_map, pattern, proj_map,
        num_layers=len(expert_layers), num_experts=num_experts,
        max_experts=max_cached_experts, format=fmt,
    )

    # Find and replace all SwitchGLU instances
    switch_glus = _find_switch_glus(model)
    print(f"  Found {len(switch_glus)} SwitchGLU layers to offload")

    for path, original_glu in switch_glus:
        m = re.search(r'\.(\d+)\.', path)
        if not m:
            print(f"    WARNING: Cannot determine layer index from {path}, skipping")
            continue
        layer_idx = int(m.group(1))

        offloaded = OffloadedSwitchGLU(
            input_dims=hidden_size,
            hidden_dims=moe_hidden,
            num_experts=num_experts,
            group_size=group_size, bits=bits,
            cache=cache, layer_idx=layer_idx,
        )

        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        setattr(parent, parts[-1], offloaded)

    # Load shared weights
    model.load_weights(list(shared_weights.items()), strict=False)
    mx.eval(model.parameters())

    # Pre-warm: incrementally load all experts if cache can hold them all
    if max_cached_experts >= total_expert_entries:
        print(f"  Pre-warming {total_expert_entries} experts...", end="", flush=True)
        t_warm = time.perf_counter()
        for layer in expert_layers:
            for eid in range(num_experts):
                cache.ensure_experts(layer, [eid])
            entries = [cache.cache[(layer, e)] for e in range(num_experts)]
            tensors = [t for entry in entries for proj in entry.values() for t in proj]
            mx.eval(*tensors)
        elapsed = time.perf_counter() - t_warm
        print(f" {elapsed:.0f}s")

        # Pre-build stacked tensors, then clear individual cache
        print(f"  Pre-building stacked tensors...", end="", flush=True)
        t_stack = time.perf_counter()
        for layer in expert_layers:
            for proj_name in proj_map:
                all_w, all_s, all_b = [], [], []
                for eid in range(num_experts):
                    w, s, b = cache.get_proj(layer, eid, proj_name)
                    all_w.append(w)
                    all_s.append(s)
                    all_b.append(b)
                sw = mx.stack(all_w)
                ss = mx.stack(all_s)
                sb = mx.stack(all_b)
                _STACKED_CACHE[(layer, proj_name)] = (sw, ss, sb)
        mx.eval(*[t for v in _STACKED_CACHE.values() for t in v])
        cache.cache.clear()
        for layer in expert_layers:
            for eid in range(num_experts):
                cache.cache[(layer, eid)] = None  # sentinel
        print(f" {time.perf_counter() - t_stack:.0f}s")

    return model, cache


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str, max_tokens: int = 100, temp: float = 0.0):
    tokens = mx.array(tokenizer.encode(prompt))[None]
    cache = _make_cache(model)

    t0 = time.perf_counter()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0

    if temp > 0:
        token = mx.random.categorical(logits[:, -1] / temp)
    else:
        token = mx.argmax(logits[:, -1], axis=-1)

    generated = [token.item()]

    t_decode_start = time.perf_counter()
    for i in range(max_tokens - 1):
        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)

        if temp > 0:
            token = mx.random.categorical(logits[:, -1] / temp)
        else:
            token = mx.argmax(logits[:, -1], axis=-1)

        tok_id = token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated.append(tok_id)

    t_decode = time.perf_counter() - t_decode_start

    return tokenizer.decode(generated), {
        "prompt_tokens": tokens.shape[1],
        "generated_tokens": len(generated),
        "prefill_time": t_prefill,
        "prefill_tok_s": tokens.shape[1] / t_prefill,
        "decode_time": t_decode,
        "decode_tok_s": len(generated) / t_decode if t_decode > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_memory_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _extract_text(content):
    """Handle both string and list content formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    return str(content)


def _build_prompt(tokenizer, messages):
    """Build prompt using tokenizer's chat template, with fallbacks."""
    clean = []
    for m in messages:
        role = m.get("role", "user")
        content = _extract_text(m.get("content", ""))
        if role == "system":
            if clean and clean[-1]["role"] == "user":
                clean[-1]["content"] = content + "\n\n" + clean[-1]["content"]
            else:
                clean.append({"role": "user", "content": content})
            continue
        if clean and clean[-1]["role"] == role:
            clean[-1]["content"] += "\n" + content
        else:
            clean.append({"role": role, "content": content})
    if clean and clean[0]["role"] != "user":
        clean.insert(0, {"role": "user", "content": "Hi"})
    if not clean:
        clean = [{"role": "user", "content": "Hello"}]

    # Try apply_chat_template first
    try:
        return tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass

    # Fallback: Mixtral/Llama instruct format
    prompt = ""
    for m in clean:
        if m["role"] == "user":
            prompt += f"<s>[INST] {m['content']} [/INST]"
        elif m["role"] == "assistant":
            prompt += f" {m['content']}</s>"
    return prompt or "<s>[INST] Hello [/INST]"


def stream_generate(model, tokenizer, messages, max_tokens=512, temp=0.7):
    """Generate tokens as a stream, yielding incremental text chunks."""
    prompt = _build_prompt(tokenizer, messages)
    tokens = mx.array(tokenizer.encode(prompt))[None]
    cache = _make_cache(model)

    logits = model(tokens, cache=cache)
    mx.eval(logits)

    if temp > 0:
        token = mx.random.categorical(logits[:, -1] / temp)
    else:
        token = mx.argmax(logits[:, -1], axis=-1)

    generated_ids = []
    prev_text = ""

    for i in range(max_tokens):
        tok_id = token.item()
        if tok_id == tokenizer.eos_token_id:
            break

        generated_ids.append(tok_id)
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        delta = full_text[len(prev_text):]
        prev_text = full_text
        if delta:
            yield delta

        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)
        if temp > 0:
            token = mx.random.categorical(logits[:, -1] / temp)
        else:
            token = mx.argmax(logits[:, -1], axis=-1)


# ---------------------------------------------------------------------------
# Serve mode — minimal OpenAI-compatible API
# ---------------------------------------------------------------------------

def _make_chat_html(model_name: str, description: str):
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MoE Expert Offloading Chat</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}}
.header{{padding:12px 20px;background:#16213e;border-bottom:1px solid #333;font-size:14px;color:#8888aa}}
.header b{{color:#4fc3f7}}
#chat{{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}}
.msg{{max-width:80%;padding:12px 16px;border-radius:12px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}}
.user{{align-self:flex-end;background:#1e3a5f;border-bottom-right-radius:4px}}
.assistant{{align-self:flex-start;background:#2a2a3e;border-bottom-left-radius:4px}}
.input-area{{padding:12px 20px;background:#16213e;border-top:1px solid #333;display:flex;gap:8px}}
#input{{flex:1;padding:10px 14px;border:1px solid #444;border-radius:8px;background:#1a1a2e;color:#e0e0e0;font-size:15px;outline:none}}
#input:focus{{border-color:#4fc3f7}}
button{{padding:10px 20px;border:none;border-radius:8px;background:#4fc3f7;color:#000;font-weight:bold;cursor:pointer}}
button:disabled{{opacity:0.5}}
.typing{{color:#888;font-style:italic}}
</style></head><body>
<div class="header"><b>{model_name}</b> &mdash; {description}</div>
<div id="chat"></div>
<div class="input-area">
<input id="input" placeholder="Type a message..." autofocus>
<button id="send" onclick="send()">Send</button>
</div>
<script>
const chat=document.getElementById('chat'),input=document.getElementById('input'),btn=document.getElementById('send');
let messages=[];
input.addEventListener('keydown',e=>{{if(e.key==='Enter'&&!e.shiftKey){{e.preventDefault();send()}}}});
async function send(){{
  const text=input.value.trim();if(!text)return;
  input.value='';btn.disabled=true;
  messages.push({{role:'user',content:text}});
  addMsg('user',text);
  const el=addMsg('assistant','');
  el.classList.add('typing');el.textContent='thinking...';
  try{{
    const res=await fetch('/v1/chat/completions',{{method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{messages,stream:true,max_tokens:512,temperature:0.7}})}});
    const reader=res.body.getReader();const dec=new TextDecoder();
    let full='';el.classList.remove('typing');el.textContent='';
    while(true){{
      const{{done,value}}=await reader.read();if(done)break;
      const lines=dec.decode(value).split('\\n');
      for(const line of lines){{
        if(!line.startsWith('data: ')||line==='data: [DONE]')continue;
        try{{const j=JSON.parse(line.slice(6));const d=j.choices?.[0]?.delta?.content;
        if(d){{full+=d;el.textContent=full;chat.scrollTop=chat.scrollHeight;}}}}catch{{}}}}
    }}
    messages.push({{role:'assistant',content:full}});
  }}catch(e){{el.textContent='Error: '+e.message;el.classList.remove('typing');}}
  btn.disabled=false;input.focus();
}}
function addMsg(role,text){{
  const d=document.createElement('div');d.className='msg '+role;d.textContent=text;
  chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d;
}}
</script></body></html>"""


def serve_mode(model, tokenizer, expert_cache, port=8080, model_id="offloaded-moe"):
    """Start a minimal OpenAI-compatible server with web chat UI."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import uuid

    # Build chat HTML with model info
    total_experts = expert_cache.num_layers * expert_cache.num_experts_per_layer
    cached = len(expert_cache.cache)
    pct = cached * 100 // total_experts if total_experts else 0
    desc = f"expert-offloaded, {cached}/{total_experts} experts cached ({pct}%)"
    chat_html = _make_chat_html(model_id, desc)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            messages = body.get("messages", [])
            max_tokens = body.get("max_tokens", 512)
            temperature = body.get("temperature", 0.7)
            stream = body.get("stream", False)

            if stream:
                self._handle_stream(messages, max_tokens, temperature)
            else:
                self._handle_sync(messages, max_tokens, temperature)

        def _handle_stream(self, messages, max_tokens, temperature):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            try:
                for text in stream_generate(model, tokenizer, messages, max_tokens, temperature):
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
            except Exception as e:
                err_chunk = {"id": req_id, "object": "chat.completion.chunk",
                             "model": model_id,
                             "choices": [{"index": 0, "delta": {"content": f"\n\n[Error: {e}]"}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(err_chunk)}\n\n".encode())
                import traceback
                traceback.print_exc()

            done = {"id": req_id, "object": "chat.completion.chunk",
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            self.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()

        def _handle_sync(self, messages, max_tokens, temperature):
            try:
                tokens = list(stream_generate(model, tokenizer, messages, max_tokens, temperature))
            except Exception as e:
                import traceback
                traceback.print_exc()
                tokens = [f"[Error: {e}]"]
            text = "".join(tokens)

            resp = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": len(tokens), "total_tokens": len(tokens)},
            }
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                resp = {"data": [{"id": model_id, "object": "model"}]}
                body = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/healthz":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            elif self.path in ("/", "/chat"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(chat_html.encode())
            else:
                self.send_error(404)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()

        def log_message(self, format, *args):
            pass  # quiet

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\nServer ready at http://localhost:{port}/v1/chat/completions")
    print(f"   Model: {model_id}")
    print(f"   Expert cache: {expert_cache.stats()}")
    print(f"   Try: curl -X POST http://localhost:{port}/v1/chat/completions \\")
    print(f'     -H "Content-Type: application/json" \\')
    print(f'     -d \'{{"messages":[{{"role":"user","content":"Hi"}}],"stream":true}}\'')
    print(f"\n   Or connect any OpenAI-compatible client to http://localhost:{port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MoE Expert Offloading — run large MoE models on small Macs")
    parser.add_argument("--model", default="mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit")
    parser.add_argument("--max-cached-experts", type=int, default=None,
                        help="Expert cache size (default: auto based on RAM)")
    parser.add_argument("--serve", action="store_true", help="Start OpenAI-compatible server")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()

    model_path = args.model
    if not os.path.isdir(model_path):
        from huggingface_hub import snapshot_download
        print(f"Downloading {model_path}...")
        model_path = snapshot_download(model_path)
    print(f"Model: {model_path}")

    # Derive a short model ID for API responses
    mp = Path(model_path)
    # Handle HuggingFace cache paths: .../models--org--name/snapshots/hash/
    if "snapshots" in mp.parts:
        # Go up to find models--org--name
        for parent in mp.parents:
            if parent.name.startswith("models--"):
                model_id = parent.name.replace("models--", "").replace("--", "/")
                break
        else:
            model_id = mp.name
    else:
        model_id = mp.name
    model_id = model_id.split("/")[-1].lower().replace(" ", "-") + "-offloaded"

    print(f"\n=== Loading with expert offloading ===")
    model, expert_cache = load_model_offloaded(model_path, args.max_cached_experts)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    mem = get_memory_mb()
    print(f"\n  RAM: {mem:.0f} MB ({32768 - mem:.0f} MB free)")

    if args.serve:
        serve_mode(model, tokenizer, expert_cache, port=args.port, model_id=model_id)
    else:
        print(f"\n=== Generating ({args.max_tokens} tokens max) ===")
        text, stats = generate(model, tokenizer, args.prompt, args.max_tokens, args.temp)
        print(f"\nOutput: {text}")
        print(f"\n=== Performance ===")
        print(f"  Prefill: {stats['prompt_tokens']} tok / {stats['prefill_time']:.1f}s")
        print(f"  Decode:  {stats['generated_tokens']} tok / {stats['decode_time']:.1f}s ({stats['decode_tok_s']:.2f} tok/s)")
        print(f"  Cache:   {expert_cache.stats()}")


if __name__ == "__main__":
    main()
