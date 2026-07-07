"""WSGI entry point.

Run locally:      python wsgi.py
Run with gunicorn (recommended in production, from the web/ directory):
                  gunicorn --workers 2 --timeout 120 wsgi:app
"""

from corpussearch import create_app

app = create_app()

if __name__ == "__main__":
    import os

    # Run from this file's directory so relative paths resolve regardless of the
    # caller's cwd (the auto-reloader re-execs and would otherwise fail to find
    # this file on some network filesystems).
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # The reloader watches every imported module (including the FindMyText core
    # modules the user actively edits) and re-execs on any change, which both
    # kills the server mid-session and misbehaves over CIFS. Disable it; keep
    # debug error pages. Set FINDMYTEXT_RELOAD=1 to opt back in.
    use_reloader = os.environ.get("FINDMYTEXT_RELOAD") == "1"

    # Threaded so the streaming progress endpoint doesn't block other requests.
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=use_reloader,
    )
