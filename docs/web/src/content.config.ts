import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

export const GROUPS = [
  "Overview and history",
  "Architecture and pipeline",
  "Benchmarks and demos",
  "Guides and reference",
] as const;

const docs = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/docs" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    group: z.enum(GROUPS),
    order: z.number(),
  }),
});

export const collections = { docs };
