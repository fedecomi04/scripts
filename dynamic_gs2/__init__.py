"""dynamic_gs2 package marker.

Side effect at import time: ``_ensure_ninja_on_path`` (below) guarantees the
``ninja`` build tool is resolvable on PATH so gsplat's JIT CUDA extension can
compile on first use. This was previously inherited as an import side effect of
the old ``dynamic_gs`` package; it is ported here so dynamic_gs2 owns it directly
(idempotent + best-effort — a no-op when ninja is already on PATH). The old
package's nerfstudio outputs/-suppression monkeypatches are deliberately NOT
ported: dynamic_gs2 never constructs an ExperimentConfig/Trainer, so those code
paths are unreachable from its entry points.
"""


def _ensure_ninja_on_path() -> None:
    """Guarantee the ``ninja`` build tool is on PATH for gsplat's JIT.

    gsplat JIT-compiles its ``csrc`` CUDA extension on first use via
    torch.utils.cpp_extension, which calls ``verify_ninja_availability()``
    and hard-fails with ``RuntimeError: Ninja is required to load C++
    extensions`` if the ``ninja`` *executable* isn't on PATH. The ``ninja``
    pip package (a gsplat/torch dependency) ships the binary under
    ``ninja.BIN_DIR``, but that dir is only on PATH inside an *activated*
    conda env. A pipeline launched any other way (bare env-python invocation,
    cron, a non-login shell) inherits a PATH without it and crashes deep in
    the first rasterization. Self-heal: if the executable isn't resolvable,
    prepend the pip package's BIN_DIR. Idempotent + best-effort.
    """
    import os
    import shutil

    if shutil.which("ninja") is not None:
        return
    try:
        import ninja  # the pip package
        bin_dir = getattr(ninja, "BIN_DIR", None)
        if bin_dir and os.path.isfile(os.path.join(bin_dir, "ninja")):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        # No ninja package either — nothing we can do here; the gsplat JIT
        # will raise its own clear error. Don't mask it.
        pass


_ensure_ninja_on_path()
