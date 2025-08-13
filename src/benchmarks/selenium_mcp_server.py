from __future__ import annotations

import asyncio
import time
import json
import threading
import socket
from typing import Any, Dict, Optional

from bs4 import BeautifulSoup
from websockets.server import serve
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By


_server_thread: Optional[threading.Thread] = None
_server_started: bool = False
_server_stop: Optional[asyncio.Future] = None


def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class _Driver:
    def __init__(self) -> None:
        options = ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=options)
        try:
            self.driver.delete_all_cookies()
        except Exception:
            pass

    def quit(self) -> None:
        try:
            self.driver.quit()
        except Exception:
            pass


class SeleniumMCPServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._driver = _Driver()

    async def _handle(self, websocket) -> None:
        try:
            async for raw in websocket:
                try:
                    req = json.loads(raw)
                    tool = req.get("tool")
                    data = req.get("input", {}) or {}
                    result: Dict[str, Any] = {}
                    if tool == "browser.getLocation":
                        result = self._get_location()
                    elif tool == "browser.navigate":
                        url = data.get("url")
                        if not url:
                            raise ValueError("missing url")
                        self._nav(url)
                        result = self._get_location()
                    elif tool == "browser.click":
                        sel = data.get("selector")
                        if not sel:
                            raise ValueError("missing selector")
                        self._click(sel)
                        result = self._get_location()
                    elif tool == "browser.type":
                        sel = data.get("selector")
                        txt = data.get("text", "")
                        if not sel:
                            raise ValueError("missing selector")
                        self._type(sel, txt, clear=data.get("clear", True))
                        result = {"ok": True}
                    elif tool == "browser.submit":
                        sel = data.get("selector")
                        if not sel:
                            raise ValueError("missing selector")
                        self._submit(sel)
                        result = self._get_location()
                    elif tool == "browser.clearCookies":
                        try:
                            self._driver.driver.delete_all_cookies()
                        except Exception:
                            pass
                        result = {"ok": True}
                    elif tool == "browser.getDomSummary":
                        viewport_only = bool(data.get("viewportOnly", True))
                        max_links = int(data.get("maxLinks", 25))
                        max_inputs = int(data.get("maxInputs", 25))
                        max_forms = int(data.get("maxForms", 10))
                        result = self._get_dom_summary(viewport_only, max_links, max_inputs, max_forms)
                    elif tool == "browser.querySelectorExists":
                        sel = data.get("selector")
                        if not sel:
                            raise ValueError("missing selector")
                        result = {"exists": self._selector_exists(sel)}
                    else:
                        await websocket.send(json.dumps({"error": f"unknown tool: {tool}"}))
                        continue
                    await websocket.send(json.dumps(result))
                except Exception as e:
                    try:
                        await websocket.send(json.dumps({"error": str(e)}))
                    except Exception:
                        break
        except Exception:
            # connection closed or cancelled
            return

    def _get_location(self) -> Dict[str, Any]:
        d = self._driver.driver
        return {"url": d.current_url, "title": d.title}

    def _nav(self, url: str) -> None:
        self._driver.driver.get(url)

    def _click(self, selector: str) -> None:
        d = self._driver.driver
        elem = None
        if selector.startswith("/"):
            elem = d.find_element(By.CSS_SELECTOR, f'a[href="{selector}"]')
        else:
            elem = d.find_element(By.CSS_SELECTOR, selector)
        elem.click()

    def _type(self, selector: str, text: str, clear: bool = True) -> None:
        d = self._driver.driver
        el = d.find_element(By.CSS_SELECTOR, selector)
        if clear:
            try:
                el.clear()
            except Exception:
                pass
        el.send_keys(text)

    def _submit(self, selector: str) -> None:
        d = self._driver.driver
        try:
            form = d.find_element(By.CSS_SELECTOR, selector)
            form.submit()
        except Exception:
            # Fallback: click first submit button inside the form
            form = d.find_element(By.CSS_SELECTOR, selector)
            btn = form.find_element(By.CSS_SELECTOR, 'button[type="submit"], input[type="submit"]')
            btn.click()

    def _selector_exists(self, selector: str) -> bool:
        d = self._driver.driver
        try:
            d.find_element(By.CSS_SELECTOR, selector)
            return True
        except Exception:
            return False

    def _get_dom_summary(self, viewport_only: bool, max_links: int, max_inputs: int, max_forms: int) -> Dict[str, Any]:
        d = self._driver.driver
        html = d.page_source
        soup = BeautifulSoup(html, "html.parser")
        clickables = []
        for a in soup.find_all("a"):
            text = (a.get_text() or "").strip()
            href = a.get("href")
            if href:
                clickables.append({"selector": href if href.startswith("/") else href, "text": text, "role": "link", "href": href})
        inputs = []
        for inp in soup.find_all("input"):
            name = inp.get("name")
            placeholder = inp.get("placeholder")
            itype = inp.get("type", "text")
            sel = f"#{inp.get('id')}" if inp.get("id") else (name or "")
            inputs.append({"selector": sel, "name": name, "placeholder": placeholder, "type": itype})
        forms = []
        for form in soup.find_all("form"):
            selector = f"form#{form.get('id')}" if form.get("id") else "form"
            fields = [f.get("name") for f in form.find_all("input") if f.get("name")]
            forms.append({"selector": selector, "fields": fields})
        return {
            "url": d.current_url,
            "title": d.title,
            "elements": {
                "clickables": clickables[:max_links],
                "inputs": inputs[:max_inputs],
                "forms": forms[:max_forms],
            },
        }

    async def _serve(self) -> None:
        global _server_stop
        stop = asyncio.Future()
        _server_stop = stop
        async with serve(self._handle, self.host, self.port, ping_interval=None, ping_timeout=None, close_timeout=1):
            await stop

    def start_background(self) -> None:
        loop = asyncio.new_event_loop()

        def runner():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._serve())

        t = threading.Thread(target=runner, name="selenium_mcp_server", daemon=True)
        t.start()


def stop_mcp_server() -> None:
    global _server_stop
    try:
        if _server_stop is not None and not _server_stop.done():
            _server_stop.get_loop().call_soon_threadsafe(_server_stop.set_result, None)
    except Exception:
        pass


def ensure_mcp_server_running(host: str = "127.0.0.1", port: int = 7007) -> None:
    global _server_thread, _server_started
    if _server_started:
        return
    if _port_open(host, port):
        _server_started = True
        return
    # Start the server
    server = SeleniumMCPServer(host, port)

    def bootstrap():
        server.start_background()

    _server_thread = threading.Thread(target=bootstrap, name="selenium_mcp_bootstrap", daemon=True)
    _server_thread.start()
    _server_started = True
    # Wait briefly until port is listening to avoid race with clients
    for _ in range(100):  # up to ~5s
        if _port_open(host, port):
            break
        time.sleep(0.05)


