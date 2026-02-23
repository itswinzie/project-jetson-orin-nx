#!/usr/bin/env python3
"""
core/ollama_client.py — Optional Ollama sidecar for language summaries.

Consumes metadata events from EventBus asynchronously and generates:
  A) Periodic 1-2 sentence summaries of scene activity
  B) On-demand Q&A answers (called synchronously from web handler)

IMPORTANT: Never sends images to Ollama — only metadata text.
"""

import json
import logging
import threading
import time
from typing import List, Optional

import requests

log = logging.getLogger("ollama")


class OllamaSidecar:
    """
    Background worker that watches the EventBus and queries Ollama periodically.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        summary_interval: float,
        max_tokens: int,
        timeout: float,
        event_bus,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.summary_interval = summary_interval
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.event_bus = event_bus

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available: Optional[bool] = None  # None=unknown, True/False

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ollama", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _check_availability(self) -> bool:
        """Ping Ollama to see if it's running."""
        try:
            r = requests.get(f"{self.endpoint}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _run(self):
        """Background loop: generate summary every N seconds."""
        last_summary_time = 0.0

        log.info(f"Ollama sidecar started. endpoint={self.endpoint} model={self.model}")

        while not self._stop.is_set():
            now = time.time()
            if now - last_summary_time >= self.summary_interval:
                # Check availability (cache for 30s)
                if self._available is None or (now - last_summary_time) > 30:
                    self._available = self._check_availability()
                    if not self._available:
                        log.warning(f"Ollama not available at {self.endpoint}. Will retry.")
                        self.event_bus.set_summary("Ollama unavailable — running vision-only mode.")

                if self._available:
                    events = self.event_bus.get_recent_meta(n=20)
                    if events:
                        summary = self._generate_summary(events)
                        if summary:
                            self.event_bus.set_summary(summary)
                            log.debug(f"Ollama summary: {summary[:80]}...")

                last_summary_time = now

            time.sleep(1.0)

    def _generate_summary(self, recent_events: List[dict]) -> str:
        """Call Ollama to summarize recent events."""
        context = _format_events_for_prompt(recent_events)
        prompt = (
            "You are a security camera assistant. "
            "Based on the following sensor metadata, write a 1-2 sentence summary of what is happening. "
            "Be concise and factual. Do not describe what you cannot know.\n\n"
            f"Metadata:\n{context}\n\n"
            "Summary:"
        )
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        """Make a synchronous call to Ollama generate API."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": 0.3,
                },
            }
            r = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("response", "").strip()
            else:
                log.warning(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
                return ""
        except requests.Timeout:
            log.warning("Ollama request timed out.")
            self._available = False
            return ""
        except Exception as exc:
            log.warning(f"Ollama error: {exc}")
            self._available = False
            return ""

    @staticmethod
    def ask_sync(
        endpoint: str,
        model: str,
        question: str,
        context: List[dict],
        timeout: float = 8.0,
    ) -> str:
        """
        Synchronous Q&A call. Used by web handler.
        Returns answer string or error message.
        """
        ctx_text = _format_events_for_prompt(context[:30])
        prompt = (
            "You are a security camera assistant. "
            "You have access to recent detection/segmentation metadata from a camera. "
            "Answer the following question based only on the metadata. "
            "If you cannot answer from the metadata, say so briefly.\n\n"
            f"Recent metadata:\n{ctx_text}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 200, "temperature": 0.2},
            }
            r = requests.post(
                f"{endpoint.rstrip('/')}/api/generate",
                json=payload,
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip() or "No answer generated."
            return f"Ollama error: HTTP {r.status_code}"
        except requests.Timeout:
            return "Ollama timed out. Try a smaller model or increase timeout in web.yaml."
        except requests.ConnectionError:
            return "Cannot connect to Ollama. Make sure it's running: ollama serve"
        except Exception as exc:
            return f"Ollama error: {exc}"


def _format_events_for_prompt(events: List[dict]) -> str:
    """Format event dicts as compact text for LLM context."""
    lines = []
    for ev in events[-20:]:  # Keep last 20 events
        mode = ev.get("mode", "?")
        fps = ev.get("fps", 0)
        if mode == "detect_open_vocab":
            labels = ev.get("labels", [])
            n = ev.get("n_detections", 0)
            lines.append(f"[detection] {n} objects: {', '.join(labels)} @ {fps:.0f}fps")
        elif mode == "segment":
            classes = ev.get("classes", [])
            areas = ev.get("areas", {})
            area_str = ", ".join(f"{k}:{v:.0f}%" for k, v in list(areas.items())[:4])
            lines.append(f"[segment] classes: {', '.join(str(c) for c in classes)} ({area_str}) @ {fps:.0f}fps")
        else:
            lines.append(str(ev))
    return "\n".join(lines) if lines else "(no events)"
