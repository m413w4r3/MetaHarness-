import { useEffect, useMemo, useRef, useState } from 'react';
import { classifyRunStatus } from '../runStatus';
import { useRunProgress } from '../../hooks/useRunProgress';
import { EventRow, eventMatchesFilter, type ProgressFilter } from './EventRow';

type BackendCall = (toolName: string, args?: Record<string, unknown>) => Promise<unknown>;

const FILTERS: ProgressFilter[] = ['All', 'Planner', 'Steps', 'Checks', 'Review', 'Recovery', 'Errors'];

export function ProgressView({ runId, status, callBackendTool, intervalMs = 1000 }: {
  runId: string;
  status?: string;
  callBackendTool?: BackendCall;
  intervalMs?: number;
}) {
  const category = classifyRunStatus(status ?? '');
  const terminal = category === 'completed' || category === 'failed';
  const slow = category === 'awaiting-action';
  const progress = useRunProgress({ runId, enabled: Boolean(callBackendTool), terminal, intervalMs: slow ? Math.max(5000, intervalMs) : intervalMs, callBackendTool });
  const [filter, setFilter] = useState<ProgressFilter>('All');
  const [autoScroll, setAutoScroll] = useState(true);
  const [copyMessage, setCopyMessage] = useState('');
  const listRef = useRef<HTMLUListElement>(null);
  const visibleEvents = useMemo(
    () => progress.events.filter((event) => eventMatchesFilter(event, filter)),
    [filter, progress.events],
  );

  useEffect(() => {
    if (autoScroll && listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [autoScroll, visibleEvents]);

  const copyVisible = async () => {
    try {
      await navigator.clipboard.writeText(visibleEvents.join('\n'));
      setCopyMessage('Copied');
    } catch {
      setCopyMessage('Copy unavailable');
    }
  };

  return <section className="metaharness-detail-section metaharness-progress" aria-labelledby="metaharness-progress-title">
    <div className="metaharness-progress__header">
      <h2 id="metaharness-progress-title">LIVE PROGRESS</h2>
      <label><input type="checkbox" checked={autoScroll} onChange={(event) => setAutoScroll(event.currentTarget.checked)} /> Auto-scroll</label>
    </div>
    <div className="metaharness-progress__actions">
      <div className="metaharness-progress__filters" role="group" aria-label="Filter progress events">
        {FILTERS.map((item) => <button type="button" key={item} aria-pressed={filter === item} onClick={() => setFilter(item)}>{item}</button>)}
      </div>
      <button type="button" className="metaharness-secondary-button" onClick={() => void copyVisible()}>Copy visible logs</button>
      <button type="button" className="metaharness-secondary-button" onClick={progress.reloadFromBeginning}>Reload from beginning</button>
      {copyMessage && <span role="status">{copyMessage}</span>}
    </div>
    {progress.droppedCount > 0 && <p className="metaharness-muted" role="status">{progress.droppedCount} older events omitted from this view.</p>}
    {progress.error && <p className="metaharness-error" role="alert">{progress.error}</p>}
    {terminal && <p className="metaharness-muted">Run is terminal; live polling stopped.</p>}
    <ul className="metaharness-progress__list" ref={listRef} aria-live="polite">
      {visibleEvents.length === 0 && <li className="metaharness-progress__row"><span className="metaharness-progress__message">No progress events have been recorded yet.</span></li>}
      {visibleEvents.map((event, index) => <EventRow event={event} key={index} />)}
    </ul>
  </section>;
}
