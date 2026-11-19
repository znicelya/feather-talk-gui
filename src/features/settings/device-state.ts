import { useCallback, useRef, useState } from "react";
import { api } from "../../lib/api";
import type { AppError, DeviceDescriptor } from "../../lib/contracts";
import { toAppError } from "../../lib/errors";

export function useDevices() {
  const [items, setItems] = useState<DeviceDescriptor[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<AppError | null>(null);
  const inFlight = useRef<Promise<void> | null>(null);

  const refresh = useCallback(() => {
    if (inFlight.current) return;
    setLoading(true);
    setError(null);
    inFlight.current = api.listDevices().then(setItems, (cause) => {
      setItems([]);
      setError(toAppError(cause, "检测计算设备"));
    }).finally(() => { inFlight.current = null; setLoading(false); });
  }, []);

  return { items, loading, error, refresh };
}
