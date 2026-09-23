import { useCallback, useEffect, useRef, useState } from 'react';

export const MAX_RUN_PROGRESS_EVENTS = 2000;

export type RunProgressState = {
  events: string[];
  droppedCount: number;
  nextOffset: number;
  loading: boolean;
  error: string;
};

type BackendCall = (toolName: string, args?: Record<string, unknown>) => Promise<unknown>;

function object(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function progressError(value: unknown): string {
  const result = object(value);
  if (result.ok === false) {
    const error = object(result.error);
    throw new Error(typeof error.message === 'string' ? error.message : 'MetaHarness progress request failed.');
  }
  return '';
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Unexpected MetaHarness progress error.';
}

export function useRunProgress({
  runId,
  enabled,
  intervalMs,
  callBackendTool,
}: {
  runId: string;
  enabled: boolean;
  intervalMs: number;
  callBackendTool?: BackendCall;
}) {
  const [state, setState] = useState<RunProgressState>({
    events: [], droppedCount: 0, nextOffset: 0, loading: false, error: '',
  });
  const stateRef = useRef(state);
  const [reloadSequence, setReloadSequence] = useState(0);
  const offsetRef = useRef(0);
  const inFlightRef = useRef(false);
  const generationRef = useRef(0);
  const runIdRef = useRef(runId);

  useEffect(() => {
    if (runIdRef.current === runId) return;
    runIdRef.current = runId;
    generationRef.current += 1;
    offsetRef.current = 0;
    const reset = { events: [], droppedCount: 0, nextOffset: 0, loading: false, error: '' };
    stateRef.current = reset;
    setState(reset);
  }, [runId]);

  const reloadFromBeginning = useCallback(() => {
    generationRef.current += 1;
    offsetRef.current = 0;
    const reset = { events: [], droppedCount: 0, nextOffset: 0, loading: false, error: '' };
    stateRef.current = reset;
    setState(reset);
    setReloadSequence((value) => value + 1);
  }, []);

  useEffect(() => {
    if (!enabled || !callBackendTool) return undefined;
    let active = true;
    const generation = generationRef.current;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;

    const poll = async () => {
      if (!active) return;
      if (inFlightRef.current) {
        timer = setTimeout(() => { void poll(); }, 100);
        return;
      }
      inFlightRef.current = true;
      stateRef.current = { ...stateRef.current, loading: true, error: '' };
      setState(stateRef.current);
      const requestedOffset = offsetRef.current;
      try {
        const raw = await callBackendTool('metaharness.progress', { runId, offset: requestedOffset });
        progressError(raw);
        const response = object(raw);
        const nextOffset = response.next_offset;
        if (!Number.isInteger(nextOffset) || (nextOffset as number) < 0) {
          throw new Error('MetaHarness returned an invalid progress offset.');
        }
        if (!Array.isArray(response.events) || !response.events.every((event) => typeof event === 'string')) {
          throw new Error('MetaHarness returned invalid progress events.');
        }
        failures = 0;
        if (active && generation === generationRef.current) {
          // A response at the same/older byte offset must never append again.
          const appended = (nextOffset as number) > requestedOffset ? response.events as string[] : [];
          const combined = [...stateRef.current.events, ...appended];
          const removed = Math.max(0, combined.length - MAX_RUN_PROGRESS_EVENTS);
          const events = removed ? combined.slice(removed) : combined;
          offsetRef.current = nextOffset as number;
          stateRef.current = {
            events,
            droppedCount: stateRef.current.droppedCount + removed,
            nextOffset: nextOffset as number,
            loading: false,
            error: '',
          };
          setState(stateRef.current);
        }
      } catch (error) {
        failures += 1;
        if (active && generation === generationRef.current) {
          stateRef.current = { ...stateRef.current, loading: false, error: errorMessage(error) };
          setState(stateRef.current);
        }
      } finally {
        inFlightRef.current = false;
        if (active) {
          const delay = Math.max(100, intervalMs) * (failures ? Math.min(4, 2 ** (failures - 1)) : 1);
          timer = setTimeout(() => { void poll(); }, delay);
        }
      }
    };

    void poll();
    return () => {
      active = false;
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [callBackendTool, enabled, intervalMs, reloadSequence, runId]);

  return { ...state, reloadFromBeginning };
}
