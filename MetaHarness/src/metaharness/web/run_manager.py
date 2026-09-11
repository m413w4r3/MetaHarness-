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


class RunCollisionError(RunManagerError):
    pass


_CREATION_TIMEOUT_SECONDS = 5.0


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

    def start_run(
        self,
        spec: str,
        *,
        run_id: str | None = None,
        planner_profile: str | None = None,
    ) -> str:
        try:
            selected_run_id = _safe_run_id(run_id) if run_id else generate_run_id()
        except (OrchestrationError, TypeError) as exc:
            raise RunManagerError(str(exc)) from exc

        with self._lock:
            if selected_run_id in self._active_run_ids:
                raise RunCollisionError("run already exists or is active")
            if len(self._active_run_ids) >= self._max_active_runs:
                raise RunCapacityError("maximum active runs reached")
            self._active_run_ids.add(selected_run_id)

        created_event = threading.Event()
        finished_event = threading.Event()
        signal = threading.Condition()

        def notify(event: threading.Event) -> None:
            with signal:
                event.set()
                signal.notify_all()

        def worker() -> None:
            try:
                orchestrator = self._orchestrator_factory(self._config)
                kwargs = {
                    "run_id": selected_run_id,
                    "on_created": lambda _run_dir: notify(created_event),
                }
                if planner_profile is not None:
                    kwargs["planner_profile"] = planner_profile
                orchestrator.run_text(spec, **kwargs)
            except BaseException:
                # The orchestrator records business failures after durable
                # initialization. Factory/thread failures must not escape the
                # daemon thread or strand capacity forever.
                pass
            finally:
                # Capacity is released before the caller is woken, so a
                # failure reported by start_run() has already restored it.
                with self._lock:
                    self._active_run_ids.discard(selected_run_id)
                notify(finished_event)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        with signal:
            signal.wait_for(
                lambda: created_event.is_set() or finished_event.is_set(),
                timeout=_CREATION_TIMEOUT_SECONDS,
            )
        if created_event.is_set():
            return selected_run_id
        if finished_event.is_set():
            raise RunManagerError("run worker failed before durable creation")
        raise RunManagerError("run creation timed out")


__all__ = ["RunCapacityError", "RunCollisionError", "RunManager", "RunManagerError"]
