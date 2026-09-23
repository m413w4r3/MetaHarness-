import { StepCard } from './StepCard';

type Data = Record<string, unknown>;
type Step = Data & { id: string };

function object(value: unknown): Data {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Data : {};
}

function entries(value: unknown): Step[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is Step => {
    const record = object(item);
    return typeof record.id === 'string';
  });
}

export function StepsView({ data }: { data: Data }) {
  const bundle = object(data.implementation_bundle);
  const cycles = Array.isArray(data.cycle_artifacts) ? data.cycle_artifacts.map((cycle) => entries(object(cycle).steps)) : [];
  const authoritative = cycles.flat();
  const initial = entries(data.step_artifacts);
  const planSteps = entries(bundle.steps);
  const byId = new Map<string, Step>();
  for (const step of [...planSteps, ...initial, ...authoritative]) byId.set(`${step.cycle ?? 1}:${step.id}`, { ...byId.get(`${step.cycle ?? 1}:${step.id}`), ...step });
  const steps = [...byId.values()];
  return <section className="metaharness-detail-section" aria-labelledby="metaharness-steps-title">
    <h2 id="metaharness-steps-title">Steps</h2>
    {steps.length ? <div className="metaharness-step-list">{steps.map((step) => <StepCard key={`${step.cycle ?? 1}:${step.id}`} step={step} />)}</div> : <p className="metaharness-muted">No implementation steps are available yet.</p>}
  </section>;
}
