"""Tool implementations registered against the shared ``ToolRegistry``.

Each module here exposes a plain function plus a ``build_spec()`` helper
that wraps it into a ``ToolSpec`` ready to hand to
``ToolRegistry.register()``. Keeping the function and its schema wrapper
separate makes the underlying logic trivially unit-testable without going
through the registry at all.
"""
