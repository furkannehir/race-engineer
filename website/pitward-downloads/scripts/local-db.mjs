import { DatabaseSync } from 'node:sqlite';
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function createLocalDatabase(filename = ':memory:') {
  const sqlite = new DatabaseSync(filename);
  for (const file of readdirSync(path.join(project, 'drizzle')).filter(name => name.endsWith('.sql')).sort()) sqlite.exec(readFileSync(path.join(project, 'drizzle', file), 'utf8'));
  class Statement {
    constructor(sql, values = []) { this.sql = sql; this.values = values; }
    bind(...values) { return new Statement(this.sql, values); }
    first() { return Promise.resolve(sqlite.prepare(this.sql).get(...this.values) || null); }
    all() { return Promise.resolve(this.execute()); }
    run() { return Promise.resolve(this.execute()); }
    execute() {
      if (/^\s*SELECT/i.test(this.sql)) return { success: true, results: sqlite.prepare(this.sql).all(...this.values) };
      const result = sqlite.prepare(this.sql).run(...this.values);
      return { success: true, results: [], meta: { changes: Number(result.changes) } };
    }
  }
  return {
    prepare(sql) { return new Statement(sql); },
    async batch(statements) {
      sqlite.exec('BEGIN');
      try { const results = statements.map(statement => statement.execute()); sqlite.exec('COMMIT'); return results; }
      catch (error) { sqlite.exec('ROLLBACK'); throw error; }
    },
    close() { sqlite.close(); },
  };
}
