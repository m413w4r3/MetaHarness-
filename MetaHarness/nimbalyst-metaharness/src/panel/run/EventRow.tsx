export type ProgressFilter = 'All' | 'Planner' | 'Steps' | 'Checks' | 'Review' | 'Recovery' | 'Errors';

export function eventMatchesFilter(event: string, filter: ProgressFilter): boolean {
  if (filter === 'All') return true;
  const value = event.toLowerCase();
  const category = value.match(/^\d{2}:\d{2}:\d{2}\s+\[([a-z]+)\]/)?.[1];
  if (filter === 'Planner') return category ? category === 'planner' : /planner|planning|plan/.test(value);
  if (filter === 'Steps') return category ? category === 'step' : /\bs\d{1,3}\b|step/.test(value);
  if (filter === 'Checks') return category ? category === 'check' : /check|lint|typecheck|test|pass|fail/.test(value);
  if (filter === 'Review') return category ? category === 'review' : /review|reviewer/.test(value);
  if (filter === 'Recovery') return category ? category === 'recovery' : /repair|recovery|waiting|resume/.test(value);
  return category ? /error|failed|failure|exception|reject|timeout|mismatch|unavailable|http 5\d\d|llm_failure/.test(value) : /error|failed|failure|exception|reject|\bfail\b|http 5\d\d|llm_failure/.test(value);
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
