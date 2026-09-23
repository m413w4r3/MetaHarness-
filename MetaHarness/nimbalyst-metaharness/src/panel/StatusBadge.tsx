import { classifyRunStatus } from './runStatus';

export function StatusBadge({ status }: { status: string }) {
  const category = classifyRunStatus(status);
  return <span className={`metaharness-run-status metaharness-run-status--${category}`}>{status}</span>;
}
