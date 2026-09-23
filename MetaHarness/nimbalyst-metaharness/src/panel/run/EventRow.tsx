export type ProgressFilter = 'All' | 'Planner' | 'Steps' | 'Checks' | 'Review' | 'Errors';

export function eventMatchesFilter(event: string, filter: ProgressFilter): boolean {
  if (filter === 'All') return true;
  const value = event.toLowerCase();
  if (filter === 'Planner') return /planner|planning|plan/.test(value);
  if (filter === 'Steps') return /\bs\d{1,3}\b|step/.test(value);
  if (filter === 'Checks') return /check|lint|typecheck|test|pass|fail/.test(value);
  if (filter === 'Review') return /review|reviewer/.test(value);
  return /error|failed|failure|exception|reject|\bfail\b/.test(value);
}

export function EventRow({ event }: { event: string }) {
  // The backend supplies a compact summary. Render that exact text as a text node.
  const time = event.match(/^(\d{2}:\d{2}:\d{2})\s+/)?.[1];
  const details = time ? event.slice(time.length).trimStart() : event;
  return <li className="metaharness-progress__row">
    {time && <time className="metaharness-progress__time">{time}</time>}
    <span className="metaharness-progress__message">{details}</span>
  </li>;
}
