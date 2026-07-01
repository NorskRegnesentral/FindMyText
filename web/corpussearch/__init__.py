"""Flask application factory for the findmytext.nr.no demo site."""

from __future__ import annotations

import json

from flask import Flask, Response, render_template, request, jsonify

from .config import load_config, config_public_dict
from .detection import IndexManager, run_detection, ALGORITHMS
from .protection import guard_request, install_rate_limiter


def create_app() -> Flask:
    app = Flask(__name__)
    cfg = load_config()
    manager = IndexManager(max_loaded=cfg.max_loaded_indexes)

    app.config["APP_CONFIG"] = cfg
    app.config["INDEX_MANAGER"] = manager

    rate_limited = install_rate_limiter(app, cfg)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            public_config=json.dumps(config_public_dict(cfg)),
        )

    @app.get("/api/config")
    def api_config():
        return jsonify(config_public_dict(cfg))

    @app.post("/api/detect")
    @rate_limited
    def api_detect():
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        corpus_id = data.get("corpus")
        algorithms = data.get("algorithms") or []
        password = data.get("password")
        captcha_token = data.get("captcha_token")

        # --- Abuse protection (no-ops unless configured) -------------------
        ok, message = guard_request(cfg, password, captcha_token, request.remote_addr)
        if not ok:
            return jsonify({"error": message}), 403

        # --- Validation ----------------------------------------------------
        corpus = cfg.corpus(corpus_id)
        if corpus is None:
            return jsonify({"error": "Unknown corpus."}), 400
        if not text:
            return jsonify({"error": "Please enter some text to check."}), 400
        if len(text) > cfg.max_text_chars:
            return jsonify({
                "error": f"Text too long (max {cfg.max_text_chars:,} characters)."
            }), 400
        algorithms = [a for a in algorithms if a in ALGORITHMS]
        if not algorithms:
            return jsonify({"error": "Select at least one detection algorithm."}), 400

        # --- Stream progress + result as newline-delimited JSON ------------
        def generate():
            try:
                for event in run_detection(manager, cfg, corpus, text, algorithms):
                    yield json.dumps(event) + "\n"
            except Exception as exc:  # noqa: BLE001
                app.logger.exception("detection failed")
                yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

        return Response(generate(), mimetype="application/x-ndjson")

    return app
