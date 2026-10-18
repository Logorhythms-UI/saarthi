import { getCollection, type CollectionEntry } from "astro:content";
import { GROUPS } from "../content.config";

export type Doc = CollectionEntry<"docs">;

export interface NavGroup {
  title: string;
  docs: Doc[];
}

/** Return all pages in reading order, in the groups of the sidebar. */
export async function getNav(): Promise<{ groups: NavGroup[]; flat: Doc[] }> {
  const all: Doc[] = await getCollection("docs");
  const groups: NavGroup[] = GROUPS.map((title) => ({
    title,
    docs: all
      .filter((d) => d.data.group === title)
      .sort((a, b) => a.data.order - b.data.order),
  }));
  return { groups, flat: groups.flatMap((g) => g.docs) };
}

export const docUrl = (id: string): string => `/docs/${id}/`;
