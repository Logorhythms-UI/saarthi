import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Group {
  title: string;
  items: { title: string; href: string }[];
}

export default function MobileNav({ groups, current }: { groups: Group[]; current: string }) {
  return (
    <Sheet>
      <SheetTrigger render={<Button variant="ghost" size="icon" className="lg:hidden" aria-label="Open the menu" />}>
        <Menu />
      </SheetTrigger>
      <SheetContent side="left" className="w-80 p-0">
        <SheetHeader>
          <SheetTitle>Documentation</SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-5rem)] px-4 pb-8">
          {groups.map((g) => (
            <div key={g.title} className="mb-5">
              <p className="mb-1 px-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                {g.title}
              </p>
              {g.items.map((i) => (
                <a
                  key={i.href}
                  href={i.href}
                  className={`block rounded-md px-2 py-1.5 text-sm hover:bg-accent ${
                    i.href === current ? "bg-accent font-medium" : "text-muted-foreground"
                  }`}
                >
                  {i.title}
                </a>
              ))}
            </div>
          ))}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
