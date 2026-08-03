# Quantiva — Distributed Module Enhancement TODO

## Plan Steps

- [x] 1. Improve `init_process_group` (backend auto-detect, timeout, explicit rank/world_size, gloo fallback)
- [x] 2. Add environment helpers: `is_distributed`, `get_local_rank`, `get_local_world_size`
- [x] 3. Add device helpers: `get_device`, `get_rank_device`, `_get_backend_device`
- [x] 4. Add rank-0 helpers: `master_print`, `barrier`
- [x] 5. Add collective ops: `all_reduce_tensor`, `all_reduce_mean`, `all_reduce_sum`, `all_gather`
- [x] 6. Add object collectives: `gather_object`, `broadcast_object`, `reduce_dict`
- [x] 7. Add model/checkpoint helpers: `sync_params`, `save_checkpoint_distributed`, `load_checkpoint_distributed`
- [x] 8. Add utility helpers: `set_seed`, `get_world_info`, `enable_tf32`
- [x] 9. Improve `wrap_ddp` (find_unused_parameters, gradient_as_bucket_view, static_graph)
- [x] 10. Improve `wrap_fsdp` (device_id, cpu_offload, sync_module_states, use_orig_params)
- [x] 11. Improve `wrap_deepspeed` (micro-batch config, docstrings)
- [x] 12. Harden `broadcast_state_dict` (lists/tuples, non-tensor values)
- [x] 13. Export distributed helpers from `quantiva/training/__init__.py`
- [x] 14. Verify: import + single-process smoke test

## New Enhancements (Round 2)

- [ ] 15. Add tensor scalar collectives: `all_reduce_mean_tensor`, `all_reduce_sum_tensor`
- [ ] 16. Add scalar min/max reductions: `all_reduce_max`, `all_reduce_min`
- [ ] 17. Add `all_gather_into_tensor` (efficient gather into a single tensor)
- [ ] 18. Add tensor-parallel sharding helpers: `split_tensor`, `gather_tensor`
- [ ] 19. Add rank-0 logging helper: `master_log`
- [ ] 20. Add formatted world summary: `log_world_info`
- [ ] 21. Add `get_device_count` helper
- [ ] 22. Add shared-filesystem checkpoint helpers: `save_shared_checkpoint`, `load_shared_checkpoint`
- [ ] 23. Add buffer sync helper: `broadcast_buffers`
- [ ] 24. Add `replace_parameter` helper
- [ ] 25. Add convenience wrapper: `wrap_ddp_auto`
- [ ] 26. Add parameter-count helpers: `get_parameter_total`, `get_parameter_breakdown`
- [ ] 27. Export all new helpers from `quantiva/training/__init__.py`
- [ ] 28. Verify: import + single-process smoke test for new helpers
