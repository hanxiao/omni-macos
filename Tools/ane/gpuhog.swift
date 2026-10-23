// Build: swiftc -O Tools/ane/gpuhog.swift -o gpuhog
// A separate process that keeps the GPU busy, as a renderer or another ML app would: back-to-back
// fp16 4096^3 matrix multiplies (MPS), one command buffer each. Prints its own throughput every
// second (TFLOPS), so the cost a co-running Omni imposes on it can be read from its log.
//   gpuhog <seconds> [fraction 0..1]   fraction < 1 idles part of each 100 ms (a lighter load)
import Foundation
import Metal
import MetalPerformanceShaders
let seconds = Double(CommandLine.arguments.dropFirst().first ?? "30") ?? 30
let duty = Double(CommandLine.arguments.dropFirst(2).first ?? "1") ?? 1
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let n = 4096
let desc = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * 2, dataType: .float16)
func mat() -> MPSMatrix { MPSMatrix(buffer: dev.makeBuffer(length: n * n * 2, options: .storageModePrivate)!, descriptor: desc) }
let a = mat(), b = mat(), c = mat()
let mm = MPSMatrixMultiplication(device: dev, transposeLeft: false, transposeRight: false, resultRows: n, resultColumns: n,
                                 interiorColumns: n, alpha: 1, beta: 0)
let start = Date()
var tick = Date(), done = 0, total = 0
while -start.timeIntervalSinceNow < seconds {
    let t0 = Date()
    while -t0.timeIntervalSinceNow < 0.1 * duty {
        let cb = q.makeCommandBuffer()!
        mm.encode(commandBuffer: cb, leftMatrix: a, rightMatrix: b, resultMatrix: c)
        cb.commit(); cb.waitUntilCompleted()
        done += 1; total += 1
    }
    if duty < 1 { Thread.sleep(forTimeInterval: 0.1 * (1 - duty)) }
    if -tick.timeIntervalSinceNow >= 1 {
        let per: Double = 2 * Double(n) * Double(n) * Double(n)
        let tf: Double = Double(done) * per / -tick.timeIntervalSinceNow / 1e12
        print(String(format: "%.1f %.2f TFLOPS", -start.timeIntervalSinceNow, tf)); fflush(stdout)
        tick = Date(); done = 0
    }
}
let flops: Double = 2 * Double(n) * Double(n) * Double(n)
let elapsed: Double = -start.timeIntervalSinceNow
print(String(format: "TOTAL %.2f TFLOPS avg", Double(total) * flops / elapsed / 1e12))
