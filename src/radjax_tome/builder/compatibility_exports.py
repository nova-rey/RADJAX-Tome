"""Compatibility-only builder exports kept outside the production façade."""

from radjax_tome._lazy_exports import LazyExportMap

COMPATIBILITY_EXPORTS: LazyExportMap = {
    name: ("radjax_tome.builder.multi_gpu_path_b", name)
    for name in (
        "MULTI_GPU_PATH_B_REPORT_FILENAME",
        "MULTI_GPU_PATH_B_REPORT_SCHEMA",
        "MULTI_GPU_WORKER_MANIFEST_FILENAME",
        "MULTI_GPU_WORKER_MANIFEST_SCHEMA",
        "MultiGPUPathBConfig",
        "build_path_b_assignments",
        "merge_path_b_candidate_records",
        "normalize_multi_gpu_devices",
        "render_multi_gpu_path_b_summary",
        "run_multi_gpu_path_b_candidate_harness",
    )
}
