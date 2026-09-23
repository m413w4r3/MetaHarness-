type Data = Record<string, unknown>;

function block(value: unknown): string | undefined {
  if (typeof value === 'string') return value;
  if (value === undefined || value === null) return undefined;
  return JSON.stringify(value, null, 2);
}

function Artifact({ title, value }: { title: string; value: unknown }) {
  const content = block(value);
  if (!content) return null;
  return <details className="metaharness-plan-artifact" open>
    <summary>{title}</summary>
    <pre>{content}</pre>
  </details>;
}

export function PlanView({ data }: { data: Data }) {
  const plan = data.plan && typeof data.plan === 'object' ? data.plan as Data : {};
  return <section className="metaharness-detail-section" aria-labelledby="metaharness-plan-title">
    <h2 id="metaharness-plan-title">Plan</h2>
    <Artifact title="Planner output" value={data.planner_raw ?? plan.raw} />
    <Artifact title="Implementation contract" value={data.implementation_contract ?? plan.contract} />
    <Artifact title="Implementation bundle" value={data.implementation_bundle} />
    {!data.planner_raw && !plan.raw && !data.implementation_contract && !plan.contract && !data.implementation_bundle && <p className="metaharness-muted">Plan has not been produced.</p>}
  </section>;
}
