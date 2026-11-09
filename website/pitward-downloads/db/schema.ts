import { index, integer, sqliteTable, text } from 'drizzle-orm/sqlite-core';

export const analyticsEvents = sqliteTable('pitward_analytics_events', {
  id: text('id').primaryKey(),
  visitId: text('visit_id').notNull(),
  kind: text('kind').notNull(),
  visitStartedAt: integer('visit_started_at').notNull(),
  recordedAt: integer('recorded_at').notNull(),
  day: text('day').notNull(),
  source: text('source').notNull(),
  device: text('device').notNull(),
}, table => [
  index('pitward_analytics_day_kind').on(table.day, table.kind),
  index('pitward_analytics_visit').on(table.visitId),
  index('pitward_analytics_retention').on(table.visitStartedAt),
]);

export const analyticsKeys = sqliteTable('pitward_analytics_keys', {
  key: text('key').primaryKey(),
  value: text('value').notNull(),
  createdAt: integer('created_at').notNull(),
});
