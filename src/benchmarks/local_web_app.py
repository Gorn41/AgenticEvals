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
        <a id="home-link" href="{{ url_for('home') }}">Home</a>
      </div>
      <div id="spacer"></div>
      <div id="footer">
        <a id="site-guide-link" href="{{ url_for('site_guide') }}">Site Guide</a>
      </div>
    </body>
    </html>
    """

    @app.route("/")
    def home():
        content_html = (
            f"""
              <p>Welcome to the Local Navigation Testbed.</p>
              <a id=\"products-link\" href=\"{url_for('section_a')}\">Products</a>
              <a id=\"offers-link\" href=\"{url_for('offers')}\">Offers</a>
              <a id=\"login-link\" href=\"{url_for('login')}\">Login</a>
              <a id=\"form-sum-link\" href=\"{url_for('form_sum')}\">Sum Form</a>
              <a id=\"token-link\" href=\"{url_for('token_page')}\">Get Token</a>
              <a id=\"catalog-link\" href=\"{url_for('catalog')}\">Catalog</a>
              <a id=\"otp-link\" href=\"{url_for('otp_request')}\">OTP Login</a>
              <a id=\"prefs-link\" href=\"{url_for('preferences')}\">Preferences</a>
            """
        )
        return render_template_string(base_template, title="Home", content=content_html)

    @app.route("/offers")
    def offers():
        content_html = (
            f"""
              <p>Decoy page. Nothing to do here.</p>
              <a id=\"back-home\" href=\"{url_for('home')}\">Back</a>
            """
        )
        return render_template_string(base_template, title="Offers", content=content_html)

    @app.route("/section-a")
    def section_a():
        content_html = (
            f"""
              <p>Section A</p>
              <a id=\"item-13\" href=\"{url_for('details', item_id=13)}\">Item 13</a>
              <a id=\"item-37\" href=\"{url_for('details', item_id=37)}\">Item 37</a>
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
              <a id=\"confirm\" href=\"{proceed_href}\">Proceed</a>
            """
        )
        return render_template_string(base_template, title=f"Details {item_id}", content=content_html)

    @app.route("/confirm/<int:item_id>")
    def confirm(item_id: int):
        # Reaching this means correct item chosen; next leads to success.
        content_html = (
            f"""
              <p>Confirm Item {item_id}</p>
              <a id=\"to-success\" href=\"{url_for('success')}\">Confirm and Continue</a>
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
                return redirect(url_for("dashboard"))
            else:
                error = "Invalid credentials"
        err_html = f"<p id=\"error\">{error}</p>" if error else ""
        content_html = (
            err_html
            + """
              <form id="login-form" method="post">
                <label>Username <input id="username" name="username" placeholder="username" /></label>
                <label>Password <input id="password" name="password" type="password" placeholder="password" /></label>
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
              <a id=\"goto-success\" href=\"{url_for('success')}\">Proceed to Success</a>
            """
        )
        return render_template_string(base_template, title="Dashboard", content=content_html)

    # Catalog and Checkout (multi-step)
    @app.route("/catalog")
    def catalog():
        content_html = (
            f"""
              <p>Catalog</p>
              <a id=\"widgets-link\" href=\"{url_for('catalog_widgets')}\">Widgets</a>
              <a id=\"gadgets-link\" href=\"{url_for('catalog_gadgets')}\">Gadgets</a>
            """
        )
        return render_template_string(base_template, title="Catalog", content=content_html)

    @app.route("/catalog/widgets")
    def catalog_widgets():
        content_html = (
            f"""
              <p>Widgets</p>
              <a id=\"widget-13\" href=\"{url_for('catalog_item', item_id=13)}\">Widget 13</a>
              <a id=\"widget-37\" href=\"{url_for('catalog_item', item_id=37)}\">Widget 37</a>
            """
        )
        return render_template_string(base_template, title="Widgets", content=content_html)

    @app.route("/catalog/gadgets")
    def catalog_gadgets():
        content_html = (
            f"""
              <p>Gadgets (decoy)</p>
              <a id=\"gadgets-back\" href=\"{url_for('catalog')}\">Back to Catalog</a>
            """
        )
        return render_template_string(base_template, title="Gadgets", content=content_html)

    @app.route("/catalog/widgets/<int:item_id>")
    def catalog_item(item_id: int):
        add_href = url_for("cart_add", item_id=item_id)
        content_html = (
            f"""
              <p>Widget {item_id}</p>
              <a id=\"add-to-cart\" href=\"{add_href}\">Add to Cart</a>
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
              <a id=\"to-checkout\" href=\"{next_href}\">Proceed to Checkout</a>
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
              <a id=\"to-address\" href=\"{url_for('address')}\">Enter Address</a>
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
                <label>Street <input id="street" name="street" placeholder="street" /></label>
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
              <a id=\"review-to-success\" href=\"{url_for('success')}\">Place Order</a>
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
              <a id=\"to-otp-verify\" href=\"{url_for('otp_verify')}\">Go to OTP Verify</a>
            """
        )
        return render_template_string(base_template, title="OTP Request", content=content_html)

    @app.route("/otp-verify", methods=["GET", "POST"])
    def otp_verify():
        if request.method == "POST":
            code = request.form.get("otp", "")
            if code == session.get("otp"):
                session["logged_in"] = True
                return redirect(url_for("dashboard"))
        content_html = (
            """
              <form id="otp-form" method="post">
                <label>OTP <input id="otp-input" name="otp" placeholder="code" /></label>
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
              <a id=\"to-admin-key\" href=\"{url_for('admin_key')}\">Admin Key</a>
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
              <a id=\"to-admin-form\" href=\"{url_for('admin_form')}\">Enter Key</a>
            """
        )
        return render_template_string(base_template, title="Admin Key", content=content_html)

    @app.route("/admin-form", methods=["GET", "POST"])
    def admin_form():
        if request.method == "POST":
            key = request.form.get("admin_key", "")
            if key == "KEY-ALPHA":
                return redirect(url_for("success"))
        content_html = (
            """
              <form id="admin-form" method="post">
                <label>Key <input id="admin-key-input" name="admin_key" placeholder="key" /></label>
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
                    return redirect(url_for("success"))
                else:
                    message = "Numbers must sum to 10"
            except ValueError:
                message = "Enter valid integers"
        msg_html = f"<p id=\"message\">{message}</p>" if message else ""
        content_html = (
            msg_html
            + """
              <form id="sum-form" method="post">
                <label>A <input id="a" name="a" placeholder="integer" /></label>
                <label>B <input id="b" name="b" placeholder="integer" /></label>
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
              <a id=\"to-token-form\" href=\"{url_for('form_token')}\">Go to Token Form</a>
            """
        )
        return render_template_string(base_template, title="Token", content=content_html)

    @app.route("/form-token", methods=["GET", "POST"])
    def form_token():
        message = None
        if request.method == "POST":
            token = request.form.get("token", "")
            if token == "TOKEN-XYZ":
                return redirect(url_for("success"))
            else:
                message = "Invalid token"
        msg_html = f"<p id=\"message\">{message}</p>" if message else ""
        content_html = (
            msg_html
            + """
              <form id="token-form" method="post">
                <label>Token <input id="token-input" name="token" placeholder="enter token" /></label>
                <button id="token-submit" type="submit">Submit</button>
              </form>
            """
        )
        return render_template_string(base_template, title="Token Form", content=content_html)

    @app.route("/success")
    def success():
        content_html = (
            """
              <h2 id="success">Success!</h2>
              <p>Benchmark goal reached.</p>
            """
        )
        return render_template_string(base_template, title="Success", content=content_html)

    @app.route("/site-guide")
    def site_guide():
        # Deterministic site guide to encourage meta-planning
        site_map_html = (
            f"""
              <h2>Site Guide</h2>
              <p>This page lists the main routes and key selectors for navigation.</p>
              <ul>
                <li>Home: <code>/</code>
                  <ul>
                    <li>Products: <code>/section-a</code></li>
                    <li>Login: <code>/login</code> (username: <b>alice</b>, password: <b>wonderland</b>)</li>
                    <li>Sum Form: <code>/form-sum</code></li>
                    <li>Token: <code>/token</code></li>
                    <li>Catalog: <code>/catalog</code> → <code>/catalog/widgets</code> → <code>/catalog/widgets/37</code> → <code>/cart-add/37</code> → <code>/cart</code> → <code>/checkout</code> → <code>/address</code> → <code>/review</code> → <code>/success</code></li>
                    <li>OTP Login: <code>/otp-request</code> (OTP shown) → <code>/otp-verify</code> → <code>/dashboard</code> → <code>/success</code></li>
                    <li>Preferences/Admin: <code>/preferences</code> (enable consent) → <code>/admin-key</code> (KEY-ALPHA) → <code>/admin-form</code> → <code>/success</code></li>
                  </ul>
                </li>
                <li>Section A: <code>/section-a</code>
                  <ul>
                    <li>Correct item details: <code>/details/37</code></li>
                    <li>Confirm then success: <code>/confirm/37</code> → <code>/success</code></li>
                  </ul>
                </li>
                <li>Token Flow: <code>/token</code> → <code>/form-token</code> (token: <b>TOKEN-XYZ</b>) → <code>/success</code></li>
                <li>Login Flow: <code>/login</code> → <code>/dashboard</code> → <code>/success</code></li>
                <li>Sum Form: <code>/form-sum</code> (A+B must equal 10) → <code>/success</code></li>
              </ul>
              <p>Tip: The "Site Guide" link is at the bottom of each page (may require scrolling).</p>
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

    # Give the server a brief moment to start
    time.sleep(0.5)
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


