type Data = Record<string, unknown>;

function object(value: unknown): Data {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Data : {};
}

function label(value: string): string {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function valueText(value: unknown): string {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (value === null || value === undefined) return '';
  return JSON.stringify(value, null, 2);
}

function Field({ name, value }: { name: string; value: unknown }) {
  if (value === undefined || value === null || value === '') return null;
  if (typeof value === 'object' && Object.keys(value).length === 0) return null;
  const rendered = valueText(value);
  return <div><dt>{label(name)}</dt><dd>{rendered.includes('\n') ? <pre>{rendered}</pre> : rendered}</dd></div>;
}

export function RunSummary({ data }: { data: Data }) {
  const options = object(data.run_options);
  const profiles = object(options.profiles);
  const planning = object(options.planning);
  const pipeline = object(options.pipeline);
  const recommendation = object(data.execution_recommendation);
  const selection = object(data.execution_selection);
  const publish = object(data.publish);
  const revisionEnabled = options.semantic_revision_enabled ?? pipeline.semantic_revision_enabled;
  const budgets: Data = {};
  for (const [key, value] of Object.entries({ ...planning, ...pipeline })) {
    if (key.includes('budget') || key.includes('attempt') || key.includes('cycle') || key.includes('max_')) budgets[key] = value;
  }
  return <section className="metaharness-detail-section" aria-labelledby="metaharness-summary-title">
    <h2 id="metaharness-summary-title">Summary</h2>
    <dl className="metaharness-detail-grid">
      <Field name="SPEC" value={data.spec} />
      <Field name="run options" value={options} />
      <Field name="execution recommendation" value={recommendation} />
      <Field name="execution selection" value={selection} />
      <Field name="publish mode" value={publish.mode ?? options.publish_mode} />
      <Field name="publish status" value={publish.status ?? publish.state} />
      <Field name="revision enabled" value={revisionEnabled} />
      <Field name="budgets" value={Object.keys(budgets).length ? budgets : undefined} />
      {Object.entries(profiles).map(([name, value]) => <Field key={name} name={`${name} profile`} value={value} />)}
    </dl>
  </section>;
}
