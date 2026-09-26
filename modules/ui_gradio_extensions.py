# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py

import os
import gradio as gr
import args_manager

from modules.localization import localization_js


GradioTemplateResponseOriginal = getattr(getattr(getattr(gr, 'routes', None), 'templates', None), 'TemplateResponse', None)

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'


def read_asset_file(rel_path):
    path = os.path.join(script_path, rel_path) if not os.path.isabs(rel_path) else rel_path
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ''
    return ''


def get_css_content():
    return read_asset_file('css/style.css')


def javascript_html():
    js_files = [
        'javascript/script.js',
        'javascript/contextMenus.js',
        'javascript/localization.js',
        'javascript/zoom.js',
        'javascript/edit-attention.js',
        'javascript/viewer.js',
        'javascript/imageviewer.js',
    ]
    samples_path = webpath(os.path.abspath('./sdxl_styles/samples/fooocus_v2.jpg'))
    head = f'<script type="text/javascript">{localization_js(args_manager.args.language)}</script>\n'
    for js_file in js_files:
        content = read_asset_file(js_file)
        if content:
            head += f'<script type="text/javascript">\n{content}\n</script>\n'
    head += f'<meta name="samples-path" content="{samples_path}">\n'

    if args_manager.args.theme:
        head += f'<script type="text/javascript">if (typeof set_theme === "function") set_theme(\"{args_manager.args.theme}\");</script>\n'

    return head


def css_html():
    content = get_css_content()
    if content:
        return f'<style>\n{content}\n</style>'
    return ''


def reload_javascript():
    if GradioTemplateResponseOriginal is None:
        return

    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        if hasattr(res, 'body') and res.body:
            res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
            res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
            if hasattr(res, 'init_headers'):
                res.init_headers()
        return res

    try:
        gr.routes.templates.TemplateResponse = template_response
    except Exception:
        pass
