import SwiftUI
import MetalKit
import simd

/// GPU point-cloud renderer for the folder embedding map.
///
/// Why Metal and not SwiftUI `Canvas`: Canvas is immediate-mode CoreGraphics on the CPU, so every
/// redraw re-runs a per-point fill loop. At tens of thousands of files that loop dominates, and it
/// re-runs on every pan/zoom frame (and, if hover is drawn in it, on every mouse move). Here the
/// positions + colors are uploaded to a GPU buffer ONCE; pan/zoom are a tiny uniform update, so the
/// GPU re-renders 50k+ premultiplied-alpha point sprites in well under a millisecond. The view only
/// redraws when something actually changes (`enableSetNeedsDisplay` + `isPaused`), never on a timer.
/// Memory is a flat positions buffer (8 B/point) + colors buffer (16 B/point) - ~1.3 MB at 56k.
struct MetalScatterView: NSViewRepresentable {
    var points: [SIMD2<Float>]      // model-space positions, row-aligned with colors
    var colors: [SIMD4<Float>]      // straight RGBA 0...1 (alpha carries the per-dot density alpha)
    var dataVersion: Int            // bumps only when the point SET changes -> re-upload the buffer
    var zoom: CGFloat
    var pan: CGSize
    var dotRadius: CGFloat          // logical points
    var inset: CGFloat

    func makeCoordinator() -> Renderer { Renderer() }

    func makeNSView(context: Context) -> MTKView {
        let v = MTKView()
        v.device = MetalScatterResources.shared?.device
        v.colorPixelFormat = .bgra8Unorm
        v.framebufferOnly = true
        v.isPaused = true                 // no display-link; we drive redraws explicitly
        v.enableSetNeedsDisplay = true
        v.layer?.isOpaque = false         // transparent drawable composited over the content background
        v.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        v.delegate = context.coordinator
        context.coordinator.sync(view: v, points: points, colors: colors, version: dataVersion,
                                 zoom: zoom, pan: pan, dotRadius: dotRadius, inset: inset)
        return v
    }

    func updateNSView(_ v: MTKView, context: Context) {
        context.coordinator.sync(view: v, points: points, colors: colors, version: dataVersion,
                                 zoom: zoom, pan: pan, dotRadius: dotRadius, inset: inset)
    }

    // MARK: - Renderer (MTKViewDelegate)

    final class Renderer: NSObject, MTKViewDelegate {
        private var count = 0
        private var posBuffer: MTLBuffer?
        private var colBuffer: MTLBuffer?
        private var center = SIMD2<Float>(0, 0)
        private var ext = SIMD2<Float>(1, 1)
        private var uploadedVersion = Int.min
        private var zoom: Float = 1, pan = SIMD2<Float>(0, 0), dotRadius: Float = 2.6, inset: Float = 24

        /// Re-upload the GPU buffers only when the point set actually changed; redraw only when the
        /// buffer or the transform changed. SwiftUI calls updateNSView on every surrounding state
        /// change (e.g. hover), so this guard keeps an unrelated re-render from forcing a GPU pass.
        func sync(view: MTKView, points: [SIMD2<Float>], colors: [SIMD4<Float>], version: Int,
                  zoom: CGFloat, pan: CGSize, dotRadius: CGFloat, inset: CGFloat) {
            let z = Float(zoom), p = SIMD2(Float(pan.width), Float(pan.height))
            let r = Float(dotRadius), ins = Float(inset)
            let changed = version != uploadedVersion || z != self.zoom || p != self.pan || r != self.dotRadius || ins != self.inset
            self.zoom = z; self.pan = p; self.dotRadius = r; self.inset = ins
            if version != uploadedVersion {
                upload(points: points, colors: colors)
                uploadedVersion = version
            }
            if changed { view.setNeedsDisplay(view.bounds) }
        }

        private func upload(points: [SIMD2<Float>], colors: [SIMD4<Float>]) {
            guard let device = MetalScatterResources.shared?.device else { return }
            let n = min(points.count, colors.count)
            count = n
            guard n > 0 else { posBuffer = nil; colBuffer = nil; return }
            posBuffer = points.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: n * 8, options: .storageModeShared) }
            colBuffer = colors.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: n * 16, options: .storageModeShared) }
            // Bounding box (model space) for aspect-preserving normalization.
            var mn = SIMD2<Float>(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
            var mx = SIMD2<Float>(-.greatestFiniteMagnitude, -.greatestFiniteMagnitude)
            for i in 0 ..< n where points[i].x.isFinite && points[i].y.isFinite {
                mn = simd.min(mn, points[i]); mx = simd.max(mx, points[i])
            }
            center = (mn + mx) * 0.5
            ext = simd.max(mx - mn, SIMD2<Float>(1e-5, 1e-5))
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { view.setNeedsDisplay(view.bounds) }

        func draw(in view: MTKView) {
            // Always run a render pass so the drawable is CLEARED (loadAction = .clear -> transparent)
            // even with no points - otherwise presenting an undrawn drawable shows GPU garbage
            // (magenta). Points are drawn only when present (e.g. blank during a fit).
            guard let res = MetalScatterResources.shared,
                  let pass = view.currentRenderPassDescriptor,
                  let drawable = view.currentDrawable,
                  let cmd = res.queue.makeCommandBuffer(),
                  let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return }
            if count > 0, let posBuffer, let colBuffer {
                let drawableSize = view.drawableSize
                let scaleFactor = Float(drawableSize.width) / Float(max(view.bounds.width, 1))   // retina (points -> pixels)
                let vpW = Float(drawableSize.width), vpH = Float(drawableSize.height)
                let insetPx = inset * scaleFactor
                // Aspect-preserving fit of the cloud's bbox into the inset rect, then user zoom.
                let baseScale = min((vpW - 2 * insetPx) / ext.x, (vpH - 2 * insetPx) / ext.y)
                var u = ScatterUniforms(
                    center: center,
                    offset: SIMD2(vpW * 0.5 + pan.x * scaleFactor, vpH * 0.5 + pan.y * scaleFactor),
                    viewport: SIMD2(vpW, vpH),
                    scale: baseScale * zoom,
                    pointSize: max(dotRadius * 2 * scaleFactor, 1)
                )
                enc.setRenderPipelineState(res.pipeline)
                enc.setVertexBuffer(posBuffer, offset: 0, index: 0)
                enc.setVertexBuffer(colBuffer, offset: 0, index: 1)
                enc.setVertexBytes(&u, length: MemoryLayout<ScatterUniforms>.stride, index: 2)
                enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: count)
            }
            enc.endEncoding()
            cmd.present(drawable)
            cmd.commit()
        }
    }
}

/// Uniforms shared with the Metal shader (same field order/layout).
private struct ScatterUniforms {
    var center: SIMD2<Float>
    var offset: SIMD2<Float>
    var viewport: SIMD2<Float>
    var scale: Float
    var pointSize: Float
}

/// Process-wide Metal device/queue/pipeline, compiled once (not per view appearance). The shader is
/// built from source at runtime so no .metal build step is needed. Immutable after init and the
/// Metal objects are thread-safe, so the shared instance is safe to reach from any thread.
final class MetalScatterResources: @unchecked Sendable {
    static let shared = MetalScatterResources()
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLRenderPipelineState

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;
    struct Uniforms { float2 center; float2 offset; float2 viewport; float scale; float pointSize; };
    struct VOut { float4 position [[position]]; float psize [[point_size]]; float4 color; };
    vertex VOut scatter_vertex(uint vid [[vertex_id]],
                               const device float2* pos [[buffer(0)]],
                               const device float4* col [[buffer(1)]],
                               constant Uniforms& u [[buffer(2)]]) {
        float2 px = (pos[vid] - u.center) * u.scale + u.offset;
        float2 ndc = float2(px.x / u.viewport.x * 2.0 - 1.0, 1.0 - px.y / u.viewport.y * 2.0);
        VOut o;
        o.position = float4(ndc, 0.0, 1.0);
        o.psize = max(u.pointSize, 1.0);
        o.color = col[vid];
        return o;
    }
    fragment float4 scatter_fragment(VOut in [[stage_in]], float2 pc [[point_coord]]) {
        float d = length(pc - float2(0.5)) * 2.0;          // 0 center .. 1 edge
        float a = (1.0 - smoothstep(0.8, 1.0, d)) * in.color.a;   // soft round edge
        return float4(in.color.rgb * a, a);                // premultiplied alpha
    }
    """

    private init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = try? device.makeLibrary(source: Self.shaderSource, options: nil),
              let vfn = library.makeFunction(name: "scatter_vertex"),
              let ffn = library.makeFunction(name: "scatter_fragment") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        let att = desc.colorAttachments[0]!
        att.pixelFormat = .bgra8Unorm
        att.isBlendingEnabled = true                        // premultiplied over-compositing so
        att.rgbBlendOperation = .add                        // overlapping translucent dots stack
        att.alphaBlendOperation = .add                      // (density), correct over a clear drawable
        att.sourceRGBBlendFactor = .one
        att.sourceAlphaBlendFactor = .one
        att.destinationRGBBlendFactor = .oneMinusSourceAlpha
        att.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: desc) else { return nil }
        self.device = device; self.queue = queue; self.pipeline = pipeline
    }
}
