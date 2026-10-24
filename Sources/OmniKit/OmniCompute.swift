import Foundation
import MLX

/// WHERE MLX WORK OUTSIDE THE ENCODERS RUNS: the store's scans, masks and replicas and the folder
/// map's projection. On the GPU, as they always have; in Neural Engine mode on the CPU, so that mode
/// leaves the GPU to other apps (Han, 2026-10-23: "Omni never races against GPU"). Set when the
/// Neural Engine is chosen (AppModel) or takes over (OmniEngine.setComputeUnit), cleared when the
/// GPU is.
///
/// A SCOPE, NOT A GLOBAL DEFAULT. mlx-swift reads its process default device once, at first use
/// (a task-local's initial value), so `Device.setDefault` cannot move later work; every entry
/// point runs its MLX inside `Device.withDefaultDevice(OmniCompute.device)` instead - the store
/// through its queue (ComputeQueue), the map in ProjectionEngine.project.
///
/// NEVER A COMPILED GRAPH ON THE CPU IN THE APP. MLX builds a compiled graph's CPU kernels as a
/// C++ library at run time and dlopens it, which a hardened, signed app may not: "Could not load
/// C++ shared library" and the app is gone (measured: the model-stamp probe, the backbone's
/// compiled B==1 forward, on the CPU right after the switch). So the encoders stay on their own
/// path - in this mode the stamp probe, one text forward a launch, is their only GPU work - and
/// the map runs its epochs uncompiled here. Tests and tools are not hardened and would not see it.
public enum OmniCompute {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var _offGPU = false

    /// True in Neural Engine mode: nothing of Omni's runs on the GPU.
    public static var offGPU: Bool {
        get { lock.withLock { _offGPU } }
        set { lock.withLock { _offGPU = newValue } }
    }

    /// The device for MLX work outside the encoders.
    public static var device: Device { offGPU ? .cpu : .gpu }

    /// `work` with its MLX on the CPU when `off` (Neural Engine mode); as it was otherwise, so a
    /// caller's own scope still holds on the GPU path.
    public static func scoped<T>(off: Bool = offGPU, _ work: () throws -> T) rethrows -> T {
        off ? try Device.withDefaultDevice(.cpu, work) : try work()
    }
}

/// A serial queue whose every block runs its MLX on `OmniCompute.device`, decided when the block
/// runs: the store's one queue, so a scan started after the switch runs where the switch says.
struct ComputeQueue: @unchecked Sendable {
    let base: DispatchQueue
    init(label: String) { base = DispatchQueue(label: label) }

    func sync<T>(execute work: () throws -> T) rethrows -> T {
        try base.sync { try OmniCompute.scoped(work) }
    }

    func async(execute work: @escaping @Sendable () -> Void) {
        base.async { OmniCompute.scoped(work) }
    }
}
