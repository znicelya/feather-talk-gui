import type { BackendKind, DeviceDescriptor, TargetPlatform } from "../../lib/contracts";

export const backendLabel: Record<BackendKind, string> = {
  flex: "CPU · Flex", cpu: "CPU", vulkan: "Vulkan", cuda: "CUDA", metal: "Metal",
};

export function deviceName(device: DeviceDescriptor) {
  // CUDA's current worker reports a Debug-shaped native identity. Keep it available in details.
  return /^DeviceIdentity \{ name: "([^"\r\n]+)"/.exec(device.name)?.[1] ?? device.name;
}

export function targetKey(target: TargetPlatform) {
  return `${target.os}/${target.architecture}/${target.backend}`;
}

export function deviceKey(device: Pick<DeviceDescriptor, "id" | "backend">) {
  return JSON.stringify([device.backend, device.id]);
}
