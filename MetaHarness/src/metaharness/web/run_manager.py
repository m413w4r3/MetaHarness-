"""Background execution manager for runs submitted through the local UI."""

from __future__ import annotations

import threading
from typing import Callable

from ..config import HarnessConfig
from ..orchestrator import Orchestrator, OrchestrationError, _safe_run_id, generate_run_id


class RunManagerError(RuntimeError):
    pass


class RunCapacityError(RunManagerError):
    pass


class RunManager:
    def __init__(
        self,
        config: HarnessConfig,
        *,
        max_active_runs: int = 1,
        orchestrator_factory: Callable[[HarnessConfig], Orchestrator] | None = None,
    ) -> None:
        if max_active_runs < 1:
            raise ValueError("max_active_runs must be positive")
        self._config = config
        self._max_active_runs = max_active_runs
        self._orchestrator_factory = orchestrator_factory or Orchestrator
        self._lock = threading.Lock()
        self._active_run_ids: set[str] = set()

    def start_run(self, spec: str, *, run_id: str | None = None) -> str:
        try:
            selected_run_id = _safe_run_id(run_id) if run_id else generate_run_id()
        except (OrchestrationError, TypeError) as exc:
            raise RunManagerError(str(exc)) from exc

        with self._lock:
            if len(self._active_run_ids) >= self._max_active_runs:
                raise RunCapacityError("maximum active runs reached")
            self._active_run_ids.add(selected_run_id)

        created_event = threading.Event()

        def worker() -> None:
            try:
                orchestrator = self._orchestrator_factory(self._config)
                orchestrator.run_text(
                    spec,
                    run_id=selected_run_id,
                    on_created=lambda _run_dir: created_event.set(),
                )
            except BaseException:
                # The orchestrator records business failures after durable
                # initialization. Factory/thread failures must not escape the
                # daemon thread or strand capacity forever.
                pass
            finally:
                with self._lock:
                    self._active_run_ids.discard(selected_run_id)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        if not created_event.wait(timeout=5.0):
            if not thread.is_alive():
                raise RunManagerError("run worker failed before durable creation")
            raise RunManagerError("run creation timed out")
        return selected_run_id


__all__ = ["RunCapacityError", "RunManager", "RunManagerError"]
