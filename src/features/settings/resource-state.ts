import { useCallback, useEffect, useState } from "react";
import { api } from "../../lib/api";
import type { AppError, DevicePolicy, ResourceInventory, ResourceSelection } from "../../lib/contracts";
import { toAppError } from "../../lib/errors";

export interface ResourceCheck {
  selection: ResourceSelection;
  inventory: ResourceInventory | null;
  error: AppError | null;
}

export function useResourceChecks() {
  const [checks, setChecks] = useState<ResourceCheck[]>([]);
  const [error, setError] = useState<AppError | null>(null);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => {
    setLoading(true);
    setRevision((n) => n + 1);
  }, []);

  useEffect(() => {
    let alive = true;
    void api.listResourceSelections().then(async (selections) => {
      const next = await Promise.all(selections.map(async (selection): Promise<ResourceCheck> => {
        try {
          return { selection, inventory: await api.getResources(selection.target), error: null };
        } catch (cause) {
          return { selection, inventory: null, error: toAppError(cause, "校验本地资源") };
        }
      }));
      if (alive) { setChecks(next); setError(null); }
    }).catch((cause) => {
      if (alive) { setChecks([]); setError(toAppError(cause, "读取资源清单")); }
    }).finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [revision]);

  return { checks, error, loading, refresh };
}

export function hasReadyResources(checks: ResourceCheck[], policy: DevicePolicy) {
  return checks.some(({ selection, inventory }) => inventory?.ready && (
    policy.policy === "auto" || selection.target.backend === (policy.request.backend === "flex" ? "cpu" : policy.request.backend)
  ));
}

export function missingSelection({ selection, inventory }: ResourceCheck): ResourceSelection | null {
  if (!inventory || inventory.ready) return null;
  const resources = selection.resources.filter((pin) => pin.version !== "unconfigured" && inventory.items.some(({ resource, ready }) =>
    !ready && resource.id === pin.id && resource.version === pin.version));
  return resources.length ? { target: selection.target, resources } : null;
}
