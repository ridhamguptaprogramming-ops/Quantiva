# Quantiva — Distributed Module Enhancement TODO

## Plan Steps

- [ ] 1. Improve `init_process_group` (backend auto-detect, timeout, explicit rank/world_size, gloo fallback)
- [ ] 2. Add environment helpers: `is_distributed`, `get_local_rank`, `get_local_world_size`
- [ ] 3. Add device helpers: `get_device`, `get_rank_device`, `_get_backend_device`
- [ ] 4. Add rank-0 helpers: `master_print`, `barrier`
- [ ] 5. Add collective ops: `all_reduce_tensor`, `all_reduce_mean`, `all_reduce_sum`, `all_gather`
- [ ] 6. Add object collectives: `gather_object`, `broadcast_object`, `reduce_dict`
- [ ] 7. Add model/checkpoint helpers: `sync_params`, `save_checkpoint_distributed`, `load_checkpoint_distributed`
- [ ] 8. Add utility helpers: `set_seed`, `get_world_info`, `enable_tf32`
- [ ] 9. Improve `wrap_ddp` (find_unused_parameters, gradient_as_bucket_view, static_graph)
- [ ] 10. Improve `wrap_fsdp` (device_id, cpu_offload, sync_module_states, use_orig_params)
- [ ] 11. Improve `wrap_deepspeed` (micro-batch config, docstrings)
- [ ] 12. Harden `broadcast_state_dict` (lists/tuples, non-tensor values)
- [ ] 13. Export distributed helpers from `quantiva/training/__init__.py`
- [ ] 14. Verify: import + single-process smoke test

