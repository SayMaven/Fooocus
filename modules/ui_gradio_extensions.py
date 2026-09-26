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


def setup_gradio_routes():
    routes_module = getattr(gr, 'routes', None)
    if routes_module is None:
        return

    app_cls = getattr(routes_module, 'App', None)
    if app_cls is None or not hasattr(app_cls, 'create_app'):
        return

    if getattr(app_cls, '_fooocus_patched', False):
        return

    orig_create_app = app_cls.create_app

    def patched_create_app(blocks, *args, **kwargs):
        app = orig_create_app(blocks, *args, **kwargs)

        try:
            from fastapi.responses import FileResponse, HTMLResponse
            from fastapi import HTTPException
            import urllib.parse
            import modules.config

            def get_allowed_roots():
                raw_dirs = [
                    getattr(modules.config, 'path_outputs', None),
                    getattr(modules.config, 'temp_path', None),
                    os.path.abspath('outputs'),
                    os.path.abspath('../outputs'),
                    script_path,
                ]
                roots = []
                for rd in raw_dirs:
                    if rd:
                        try:
                            roots.append(os.path.realpath(os.path.abspath(rd)))
                        except Exception:
                            pass
                return roots

            def is_allowed(target_abs: str) -> bool:
                try:
                    real_t = os.path.realpath(target_abs)
                    for r in get_allowed_roots():
                        if real_t == r or real_t.startswith(r + os.sep):
                            return True
                except Exception:
                    pass
                return False

            @app.get("/file={file_path:path}")
            @app.get("/file/{file_path:path}")
            async def serve_file(file_path: str):
                clean_path = urllib.parse.unquote(file_path).split('?')[0]
                if not os.path.isabs(clean_path):
                    clean_path = "/" + clean_path if not (len(clean_path) > 1 and clean_path[1] == ':') else clean_path

                abs_path = os.path.abspath(clean_path)

                if not is_allowed(abs_path):
                    raise HTTPException(status_code=403, detail="Access denied")

                if not os.path.exists(abs_path):
                    if abs_path.endswith('log.html'):
                        return HTMLResponse(
                            "<!DOCTYPE html><html><body style='background:#121212;color:#eee;font-family:sans-serif;padding:2rem;text-align:center;'>"
                            "<h2>Belum ada riwayat gambar untuk hari ini.</h2>"
                            "<p>Silakan buat gambar terlebih dahulu untuk melihat History Log.</p>"
                            "</body></html>"
                        )
                    raise HTTPException(status_code=404, detail="File not found")

                media_type = "text/html; charset=utf-8" if abs_path.endswith(".html") else None
                return FileResponse(abs_path, media_type=media_type)

            @app.get("/history_log")
            async def serve_history_log():
                from modules.private_logger import get_current_html_path
                log_path = os.path.abspath(get_current_html_path())
                return await serve_file(log_path)

        except Exception as e:
            print("Failed to register custom Fooocus file routes:", e)

        return app

    app_cls.create_app = patched_create_app
    app_cls._fooocus_patched = True


def reload_javascript():
    setup_gradio_routes()

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
