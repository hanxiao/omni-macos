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

extension OmniCompute {
    /// The store's exact scan on the CPU: each bf16 row (row-major [count, dim]) against the query,
    /// summed in fp32 and rounded to bf16 - what MLX's bf16 matmul returns - with SIMD on every core.
    /// MLX's own CPU matmul and scatter-reduce cost 6.3 ms a search on a 46k-row store; this and the
    /// host reducer, 1 ms or so (index.md, "Search on the CPU, faster"). `pick`: the rows to score,
    /// gathered in place (the funnel's candidates); every row otherwise.
    static func bf16Scores(_ base: UnsafePointer<UInt16>, rows: Int, dim: Int, query: [Float],
                           pick: [Int32]? = nil) -> [Float] {
        let count = pick?.count ?? rows
        guard count > 0, dim > 0, query.count == dim else { return [] }
        // The query as MLX rounds it to bf16 (qv), round to nearest even.
        let q = query.map { Float(bitPattern: bf16Round($0.bitPattern)) }
        let lanes = 16, vectors = dim / lanes
        let qv = (0 ..< vectors).map { k in SIMD16<Float>(q[k * lanes ..< (k + 1) * lanes]) }
        var out = [Float](repeating: 0, count: count)
        // Picked rows are few and scattered (7,680 candidates): small blocks, so every core reads.
        let block = pick == nil ? 2048 : 128, blocks = (count + block - 1) / block
        qv.withUnsafeBufferPointer { qb in
            out.withUnsafeMutableBufferPointer { o in
                DispatchQueue.concurrentPerform(iterations: blocks) { t in
                    for i in t * block ..< Swift.min(count, (t + 1) * block) {
                        let r = base + (pick.map { Int($0[i]) } ?? i) * dim
                        let raw = UnsafeRawPointer(r)
                        var acc = SIMD16<Float>()
                        for k in 0 ..< vectors {
                            let h = raw.loadUnaligned(fromByteOffset: k * lanes * 2, as: SIMD16<UInt16>.self)
                            let f = unsafeBitCast(SIMD16<UInt32>(truncatingIfNeeded: h) &<< 16, to: SIMD16<Float>.self)
                            acc.addProduct(f, qb[k])
                        }
                        var s = acc.sum()
                        for j in vectors * lanes ..< dim { s += Float(bitPattern: UInt32(r[j]) << 16) * q[j] }
                        o[i] = Float(bitPattern: bf16Round(s.bitPattern))
                    }
                }
            }
        }
        return out
    }

    /// fp32 bits rounded to the nearest bf16 (ties to even), as fp32 bits; NaN kept.
    @inline(__always) static func bf16Round(_ b: UInt32) -> UInt32 {
        if b & 0x7F80_0000 == 0x7F80_0000, b & 0x007F_FFFF != 0 { return b | 0x0040_0000 }
        return (b &+ 0x7FFF &+ ((b >> 16) & 1)) & 0xFFFF_0000
    }
}
