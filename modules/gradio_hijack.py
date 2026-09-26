"""gr.Image() component hijack and modern Gradio compatibility bridge."""

from __future__ import annotations

try:
    import gradio as gr
except (ImportError, ModuleNotFoundError):
    gr = None

# Detect whether environment has legacy Gradio 3 vs modern Gradio 4/5/6
IS_LEGACY_GRADIO_3 = False
if gr is not None:
    try:
        from gradio.components.base import IOComponent
        IS_LEGACY_GRADIO_3 = True
    except (ImportError, ModuleNotFoundError):
        IS_LEGACY_GRADIO_3 = False


if IS_LEGACY_GRADIO_3:
    from modules.gradio_hijack_legacy import Image
else:
    def Image(*args, **kwargs):
        tool = kwargs.pop('tool', None)
        brush_color = kwargs.pop('brush_color', None)
        brush_radius = kwargs.pop('brush_radius', None)
        mask_opacity = kwargs.pop('mask_opacity', None)
        source = kwargs.pop('source', None)
        if source is not None and 'sources' not in kwargs:
            kwargs['sources'] = [source] if isinstance(source, str) else list(source)

        if tool in ('sketch', 'color-sketch', 'editor') and hasattr(gr, 'ImageEditor'):
            editor_kwargs = dict(kwargs)
            if hasattr(gr, 'Brush'):
                colors = [brush_color] if brush_color else ['#FFFFFF']
                editor_kwargs['brush'] = gr.Brush(colors=colors, default_color=brush_color or '#FFFFFF')
            if hasattr(gr, 'Eraser'):
                editor_kwargs['eraser'] = gr.Eraser()
            return gr.ImageEditor(*args, **editor_kwargs)

        return gr.Image(*args, **kwargs)



all_components = []

target_block = None
try:
    from gradio.components.base import Block as target_block
except (ImportError, ModuleNotFoundError):
    pass

if target_block is None:
    try:
        from gradio.blocks import Block as target_block
    except (ImportError, ModuleNotFoundError):
        target_block = getattr(gr, 'Block', None)

if target_block is not None:
    if not hasattr(target_block, 'original_init'):
        target_block.original_init = target_block.__init__

    def blk_ini(self, *args, **kwargs):
        all_components.append(self)
        return target_block.original_init(self, *args, **kwargs)

    target_block.__init__ = blk_ini

try:
    import gradio.routes
    import importlib
    gradio.routes.asyncio = importlib.reload(gradio.routes.asyncio)
    if hasattr(gradio.routes.asyncio, 'wait_for'):
        if not hasattr(gradio.routes.asyncio, 'original_wait_for'):
            gradio.routes.asyncio.original_wait_for = gradio.routes.asyncio.wait_for

        def patched_wait_for(fut, timeout):
            del timeout
            return gradio.routes.asyncio.original_wait_for(fut, timeout=65535)

        gradio.routes.asyncio.wait_for = patched_wait_for
except Exception:
    pass
