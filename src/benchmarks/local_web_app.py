"""
Deterministic local web app for the Local Web Navigation benchmark.

Routes are intentionally simple (no client-side JS required) to allow
programmatic navigation via HTTP requests without a full browser.

Scenarios covered:
- Simple navigation: home -> section -> details -> success
- Auth gate: login required to reach dashboard -> success
- Form constraint: fill two numbers that must sum to 10 -> success
- Token handoff: get token, submit it on next page -> success

This server is started once in a background thread by the benchmark.
"""

from __future__ import annotations

import threading
import time
from typing import Optional
import requests as _requests

from flask import Flask, request, redirect, url_for, render_template_string, session
import socket


_app: Optional[Flask] = None
_server_thread: Optional[threading.Thread] = None
_server_started: bool = False


def _get_app() -> Flask:
    global _app
    if _app is not None:
        return _app

    app = Flask(__name__)
    @app.route("/favicon.ico")
    def favicon():
        return ("", 204)

    # Deterministic secret key for local sessions only; no external exposure.
    app.secret_key = "agentic-evals-local-secret"

    base_template = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>{{ title }}</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
        #spacer { height: 1400px; }
        #footer { margin-top: 24px; padding-top: 8px; border-top: 1px solid #ddd; }
      </style>
    </head>
    <body>
      <h1 id="page-title">{{ title }}</h1>
      <div id="content">{{ content|safe }}</div>
      <div id="nav">
        <a id="home-link" href="{{ url_for('home') }}">/</a>
      </div>
      <div id="spacer"></div>
      <div id="footer">
        <a id="site-guide-link" href="{{ url_for('site_guide') }}">/site-guide</a>
      </div>
    </body>
    </html>
    """

    @app.route("/")
    def home():
        content_html = (
            f"""
              <p>Welcome to the Local Navigation Testbed.</p>
              <a id=\"products-link\" href=\"{url_for('section_a')}\">/section-a</a>
              <a id=\"offers-link\" href=\"{url_for('offers')}\">/offers</a>
              <a id=\"login-link\" href=\"{url_for('login')}\">/login</a>
              <a id=\"form-sum-link\" href=\"{url_for('form_sum')}\">/form-sum</a>
              <a id=\"token-link\" href=\"{url_for('token_page')}\">/token</a>
              <a id=\"catalog-link\" href=\"{url_for('catalog')}\">/catalog</a>
              <a id=\"otp-link\" href=\"{url_for('otp_request')}\">/otp-request</a>
              <a id=\"prefs-link\" href=\"{url_for('preferences')}\">/preferences</a>
            """
        )
        return render_template_string(base_template, title="Home", content=content_html)

    @app.route("/offers")
    def offers():
        content_html = (
            f"""
              <p>Decoy page. Nothing to do here.</p>
              <a id=\"back-home\" href=\"{url_for('home')}\">/</a>
            """
        )
        return render_template_string(base_template, title="Offers", content=content_html)

    @app.route("/section-a")
    def section_a():
        content_html = (
            f"""
              <p>Section A</p>
              <a id=\"item-13\" href=\"{url_for('details', item_id=13)}\">/details/13</a>
              <a id=\"item-37\" href=\"{url_for('details', item_id=37)}\">/details/37</a>
            """
        )
        return render_template_string(base_template, title="Section A", content=content_html)

    @app.route("/details/<int:item_id>")
    def details(item_id: int):
        # Only item 37 leads to confirmation path
        proceed_href = url_for("confirm", item_id=item_id) if item_id == 37 else url_for("offers")
        content_html = (
            f"""
              <p>Details for Item {item_id}</p>
              <a id=\"confirm\" href=\"{proceed_href}\">{proceed_href}</a>
            """
        )
        return render_template_string(base_template, title=f"Details {item_id}", content=content_html)

    @app.route("/confirm/<int:item_id>")
    def confirm(item_id: int):
        # Reaching this means correct item chosen; next leads to success.
        content_html = (
            f"""
              <p>Confirm Item {item_id}</p>
              <a id=\"to-success\" href=\"{url_for('success_section')}\">/success/section</a>
            """
        )
        return render_template_string(base_template, title="Confirm", content=content_html)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            if username == "alice" and password == "wonderland":
                session["logged_in"] = True
                session["login_method"] = "login"
                return redirect(url_for("dashboard"))
            else:
                error = "Invalid credentials"
        err_html = f"<p id=\"error\">{error}</p>" if error else ""
        content_html = (
            err_html
            + """
              <form id="login-form" method="post">
                <p>form#login-form</p>
                <label>#username <input id="username" name="username" placeholder="username" /></label>
                <label>#password <input id="password" name="password" type="password" placeholder="password" /></label>
                <button id="login-submit" type="submit">Login</button>
              </form>
            """
        )
        return render_template_string(base_template, title="Login", content=content_html)

    @app.route("/dashboard")
    def dashboard():
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        content_html = (
            f"""
              <p>Welcome, Alice</p>
              <a id=\"goto-success\" href=\"{url_for('success_login') if session.get('login_method') in ('login', None) else url_for('success_otp')}\">{ '/success/login' if session.get('login_method') in ('login', None) else '/success/otp' }</a>
            """
        )
        return render_template_string(base_template, title="Dashboard", content=content_html)

    # Catalog and Checkout (multi-step)
    @app.route("/catalog")
    def catalog():
        content_html = (
            f"""
              <p>Catalog</p>
              <a id=\"widgets-link\" href=\"{url_for('catalog_widgets')}\">/catalog/widgets</a>
              <a id=\"gadgets-link\" href=\"{url_for('catalog_gadgets')}\">/catalog/gadgets</a>
            """
        )
        return render_template_string(base_template, title="Catalog", content=content_html)

    @app.route("/catalog/widgets")
    def catalog_widgets():
        content_html = (
            f"""
              <p>Widgets</p>
              <a id=\"widget-13\" href=\"{url_for('catalog_item', item_id=13)}\">/catalog/widgets/13</a>
              <a id=\"widget-37\" href=\"{url_for('catalog_item', item_id=37)}\">/catalog/widgets/37</a>
            """
        )
        return render_template_string(base_template, title="Widgets", content=content_html)

    @app.route("/catalog/gadgets")
    def catalog_gadgets():
        content_html = (
            f"""
              <p>Gadgets (decoy)</p>
              <a id=\"gadgets-back\" href=\"{url_for('catalog')}\">/catalog</a>
            """
        )
        return render_template_string(base_template, title="Gadgets", content=content_html)

    @app.route("/catalog/widgets/<int:item_id>")
    def catalog_item(item_id: int):
        add_href = url_for("cart_add", item_id=item_id)
        content_html = (
            f"""
              <p>Widget {item_id}</p>
              <a id=\"add-to-cart\" href=\"{add_href}\">{add_href}</a>
            """
        )
        return render_template_string(base_template, title=f"Widget {item_id}", content=content_html)

    @app.route("/cart-add/<int:item_id>")
    def cart_add(item_id: int):
        session.setdefault("cart", [])
        cart = session["cart"]
        if item_id not in cart:
            cart.append(item_id)
            session["cart"] = cart
        return redirect(url_for("cart"))

    @app.route("/cart")
    def cart():
        cart = session.get("cart", [])
        next_href = url_for("checkout") if 37 in cart else url_for("offers")
        content_html = (
            f"""
              <p>Cart: {cart}</p>
              <a id=\"to-checkout\" href=\"{next_href}\">{next_href}</a>
            """
        )
        return render_template_string(base_template, title="Cart", content=content_html)

    @app.route("/checkout")
    def checkout():
        cart = session.get("cart", [])
        if 37 not in cart:
            return redirect(url_for("cart"))
        content_html = (
            f"""
              <p>Checkout</p>
              <a id=\"to-address\" href=\"{url_for('address')}\">/address</a>
            """
        )
        return render_template_string(base_template, title="Checkout", content=content_html)

    @app.route("/address", methods=["GET", "POST"])
    def address():
        if request.method == "POST":
            street = request.form.get("street", "").strip()
            if street:
                session["address_ok"] = True
                return redirect(url_for("review"))
        content_html = (
            """
              <form id="address-form" method="post">
                <p>form#address-form</p>
                <label>#street <input id="street" name="street" placeholder="street" /></label>
                <button id="address-submit" type="submit">Save</button>
              </form>
            """
        )
        return render_template_string(base_template, title="Address", content=content_html)

    @app.route("/review")
    def review():
        if not session.get("address_ok"):
            return redirect(url_for("address"))
        content_html = (
            f"""
              <p>Review</p>
              <a id=\"review-to-success\" href=\"{url_for('success_catalog')}\">/success/catalog</a>
            """
        )
        return render_template_string(base_template, title="Review", content=content_html)

    # OTP login flow
    @app.route("/otp-request")
    def otp_request():
        session["otp"] = "9137"
        content_html = (
            f"""
              <p id=\"otp-code\">{session['otp']}</p>
              <a id=\"to-otp-verify\" href=\"{url_for('otp_verify')}\">/otp-verify</a>
            """
        )
        return render_template_string(base_template, title="OTP Request", content=content_html)

    @app.route("/otp-verify", methods=["GET", "POST"])
    def otp_verify():
        if request.method == "POST":
            code = request.form.get("otp", "")
            if code == session.get("otp"):
                session["logged_in"] = True
                session["login_method"] = "otp"
                return redirect(url_for("dashboard"))
        content_html = (
            """
              <form id="otp-form" method="post">
                <p>form#otp-form</p>
                <label>#otp-input <input id="otp-input" name="otp" placeholder="code" /></label>
                <button id="otp-submit" type="submit">Verify</button>
              </form>
            """
        )
        return render_template_string(base_template, title="OTP Verify", content=content_html)

    # Preferences and Admin key flow
    @app.route("/preferences", methods=["GET", "POST"])
    def preferences():
        if request.method == "POST":
            session["consent"] = True
            return redirect(url_for("admin_key"))
        content_html = (
            f"""
              <p>Preferences</p>
              <form id=\"consent-form\" method=\"post\">
                <button id=\"consent-submit\" type=\"submit\">Enable Consent</button>
              </form>
              <a id=\"to-admin-key\" href=\"{url_for('admin_key')}\">/admin-key</a>
            """
        )
        return render_template_string(base_template, title="Preferences", content=content_html)

    @app.route("/admin-key")
    def admin_key():
        if not session.get("consent"):
            return redirect(url_for("preferences"))
        key = "KEY-ALPHA"
        content_html = (
            f"""
              <p id=\"admin-key\">{key}</p>
              <a id=\"to-admin-form\" href=\"{url_for('admin_form')}\">/admin-form</a>
            """
        )
        return render_template_string(base_template, title="Admin Key", content=content_html)

    @app.route("/admin-form", methods=["GET", "POST"])
    def admin_form():
        if request.method == "POST":
            key = request.form.get("admin_key", "")
            if key == "KEY-ALPHA":
                return redirect(url_for("success_admin"))
        content_html = (
            """
              <form id="admin-form" method="post">
                <p>form#admin-form</p>
                <label>#admin-key-input <input id="admin-key-input" name="admin_key" placeholder="key" /></label>
                <button id="admin-submit" type="submit">Submit</button>
              </form>
            """
        )
        return render_template_string(base_template, title="Admin Form", content=content_html)

    @app.route("/form-sum", methods=["GET", "POST"])
    def form_sum():
        message = None
        if request.method == "POST":
            try:
                a = int(request.form.get("a", "0"))
                b = int(request.form.get("b", "0"))
                if a + b == 10:
                    return redirect(url_for("success_sum"))
                else:
                    message = "Numbers must sum to 10"
            except ValueError:
                message = "Enter valid integers"
        msg_html = f"<p id=\"message\">{message}</p>" if message else ""
        content_html = (
            msg_html
            + """
              <form id="sum-form" method="post">
                <p>form#sum-form</p>
                <label>#a <input id="a" name="a" placeholder="integer" /></label>
                <label>#b <input id="b" name="b" placeholder="integer" /></label>
                <button id="sum-submit" type="submit">Submit</button>
              </form>
              <p id="hint">The sum of A and B must equal 10.</p>
            """
        )
        return render_template_string(base_template, title="Sum Form", content=content_html)

    @app.route("/token")
    def token_page():
        token_value = "TOKEN-XYZ"
        content_html = (
            f"""
              <p id=\"token\">{token_value}</p>
              <a id=\"to-token-form\" href=\"{url_for('form_token')}\">/form-token</a>
            """
        )
        return render_template_string(base_template, title="Token", content=content_html)

    @app.route("/form-token", methods=["GET", "POST"])
    def form_token():
        message = None
        if request.method == "POST":
            token = request.form.get("token", "")
            if token == "TOKEN-XYZ":
                return redirect(url_for("success_token"))
            else:
                message = "Invalid token"
        msg_html = f"<p id=\"message\">{message}</p>" if message else ""
        content_html = (
            msg_html
            + """
              <form id="token-form" method="post">
                <p>form#token-form</p>
                <label>#token-input <input id="token-input" name="token" placeholder="enter token" /></label>
                <button id="token-submit" type="submit">Submit</button>
              </form>
            """
        )
        return render_template_string(base_template, title="Token Form", content=content_html)

    # Unique success pages per flow
    @app.route("/success/section")
    def success_section():
        content_html = (
            """
              <h2 id=\"success\">Success!</h2>
              <p>Benchmark goal reached: section flow.</p>
            """
        )
        return render_template_string(base_template, title="Success (Section)", content=content_html)

    @app.route("/success/login")
    def success_login():
        content_html = (
            """
              <h2 id=\"success\">Success!</h2>
              <p>Benchmark goal reached: login flow.</p>
            """
        )
        return render_template_string(base_template, title="Success (Login)", content=content_html)

    @app.route("/success/otp")
    def success_otp():
        content_html = (
            """
              <h2 id=\"success\">Success!</h2>
              <p>Benchmark goal reached: otp flow.</p>
            """
        )
        return render_template_string(base_template, title="Success (OTP)", content=content_html)

    @app.route("/success/sum")
    def success_sum():
        content_html = (
            """
              <h2 id=\"success\">Success!</h2>
              <p>Benchmark goal reached: sum flow.</p>
            """
        )
        return render_template_string(base_template, title="Success (Sum)", content=content_html)

    @app.route("/success/token")
    def success_token():
        content_html = (
            """
              <h2 id=\"success\">Success!</h2>
              <p>Benchmark goal reached: token flow.</p>
            """
        )
        return render_template_string(base_template, title="Success (Token)", content=content_html)

    @app.route("/success/catalog")
    def success_catalog():
        content_html = (
            """
              <h2 id=\"success\">Success!</h2>
              <p>Benchmark goal reached: catalog flow.</p>
            """
        )
        return render_template_string(base_template, title="Success (Catalog)", content=content_html)

    @app.route("/success/admin")
    def success_admin():
        content_html = (
            """
              <h2 id=\"success\">Success!</h2>
              <p>Benchmark goal reached: admin flow.</p>
            """
        )
        return render_template_string(base_template, title="Success (Admin)", content=content_html)

    @app.route("/site-guide")
    def site_guide():
        # Deterministic site guide to encourage meta-planning
        site_map_html = (
            f"""
              <h2>Site Guide</h2>
              <p>This page lists the main routes, key selectors, and success pages.</p>
              <ul>
                <li>Global Navigation:
                  <ul>
                    <li>Home: <code>/</code> (available from every page)</li>
                    <li>Site Guide: <code>/site-guide</code> (footer link on every page)</li>
                  </ul>
                </li>
                <li>Section Flow:
                  <ul>
                    <li><code>/</code> → <code>/section-a</code> → <code>/details/37</code> → <code>/confirm/37</code> → <code>/success/section</code></li>
                  </ul>
                </li>
                <li>Login Flow:
                  <ul>
                    <li>Password: <code>/</code> → <code>/login</code> → submit (alice/wonderland) → <code>/dashboard</code> → <code>/success/login</code></li>
                    <li>OTP: <code>/</code> → <code>/otp-request</code> → <code>/otp-verify</code> (enter shown OTP) → <code>/dashboard</code> → <code>/success/otp</code></li>
                  </ul>
                </li>
                <li>Sum Flow:
                  <ul>
                    <li><code>/</code> → <code>/form-sum</code> (submit A+B=10) → <code>/success/sum</code></li>
                  </ul>
                </li>
                <li>Token Flow:
                  <ul>
                    <li><code>/</code> → <code>/token</code> → <code>/form-token</code> (submit TOKEN-XYZ) → <code>/success/token</code></li>
                  </ul>
                </li>
                <li>Catalog Flow:
                  <ul>
                    <li><code>/</code> → <code>/catalog</code> → <code>/catalog/widgets</code> → <code>/catalog/widgets/37</code> → <code>/cart-add/37</code> → <code>/cart</code> → <code>/checkout</code> → <code>/address</code> (submit) → <code>/review</code> → <code>/success/catalog</code></li>
                  </ul>
                </li>
                <li>Admin Flow:
                  <ul>
                    <li><code>/</code> → <code>/preferences</code> (submit consent) → <code>/admin-key</code> → <code>/admin-form</code> (submit KEY-ALPHA) → <code>/success/admin</code></li>
                  </ul>
                </li>
              </ul>
            """
        )
        return render_template_string(base_template, title="Site Guide", content=site_map_html)

    @app.route("/__shutdown__")
    def __shutdown__():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            return ("", 204)
        func()
        return ("", 200)

    @app.route("/__reset__")
    def __reset__():
        try:
            session.clear()
        except Exception:
            pass
        return ("", 200)

    @app.route("/__health__")
    def __health__():
        return ("OK", 200)

    _app = app
    return app


def ensure_server_running(host: str = "127.0.0.1", port: int = 5005):
    """Start the Flask app once in a background thread if not already running."""
    global _server_thread, _server_started
    if _server_started:
        return

    # If another process/thread already serves this port, just mark as started
    try:
        with socket.create_connection((host, port), timeout=0.25):
            _server_started = True
            return
    except OSError:
        pass

    app = _get_app()

    def run_app():
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    _server_thread = threading.Thread(target=run_app, name="local_web_app_server", daemon=True)
    _server_thread.start()

    # Wait until health endpoint responds OK to avoid races
    for _ in range(100):  # up to ~5s
        try:
            r = _requests.get(f"http://{host}:{port}/__health__", timeout=0.05)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.05)
    _server_started = True


def stop_server(host: str = "127.0.0.1", port: int = 5005):
    """Attempt to stop the background Flask dev server started by ensure_server_running."""
    global _server_started
    try:
        _requests.get(f"http://{host}:{port}/__shutdown__", timeout=0.25)
    except Exception:
        pass
    _server_started = False


def reset_server(host: str = "127.0.0.1", port: int = 5005):
    """Reset server-side session/state for determinism."""
    try:
        _requests.get(f"http://{host}:{port}/__reset__", timeout=0.25)
    except Exception:
        pass


__all__ = ["ensure_server_running", "stop_server", "reset_server"]



if __name__ == "__main__":
    # Simple CLI to run the local web app directly for manual inspection
    # Defaults to 127.0.0.1:5005 to match the benchmark
    app = _get_app()
    app.run(host="127.0.0.1", port=5005, debug=False, use_reloader=False, threaded=True)