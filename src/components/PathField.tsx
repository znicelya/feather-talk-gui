import { useId } from "react";
import { FolderOpenIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Field, FieldDescription, FieldTitle } from "@/components/ui/field";

export function PathField({ label, path, action, empty, disabled, onPick }: {
  label: string;
  path: string | null;
  action: string;
  empty: string;
  disabled?: boolean;
  onPick: () => void;
}) {
  const description = useId();
  return <Field data-disabled={disabled || undefined}>
    <FieldTitle>{label}</FieldTitle>
    <div className="path-field">
      <FieldDescription id={description} className={path ? "path-text" : undefined}>{path ?? empty}</FieldDescription>
      <Button type="button" variant="outline" size="lg" aria-describedby={description} disabled={disabled} onClick={onPick}>
        <FolderOpenIcon data-icon="inline-start" />{action}
      </Button>
    </div>
  </Field>;
}
