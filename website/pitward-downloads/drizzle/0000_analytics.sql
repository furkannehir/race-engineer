CREATE TABLE `pitward_analytics_events` (
	`id` text PRIMARY KEY NOT NULL,
	`visit_id` text NOT NULL,
	`kind` text NOT NULL,
	`visit_started_at` integer NOT NULL,
	`recorded_at` integer NOT NULL,
	`day` text NOT NULL,
	`source` text NOT NULL,
	`device` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `pitward_analytics_day_kind` ON `pitward_analytics_events` (`day`,`kind`);--> statement-breakpoint
CREATE INDEX `pitward_analytics_visit` ON `pitward_analytics_events` (`visit_id`);--> statement-breakpoint
CREATE INDEX `pitward_analytics_retention` ON `pitward_analytics_events` (`visit_started_at`);--> statement-breakpoint
CREATE TABLE `pitward_analytics_keys` (
	`key` text PRIMARY KEY NOT NULL,
	`value` text NOT NULL,
	`created_at` integer NOT NULL
);
