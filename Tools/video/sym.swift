// sym <name> <out.png>: an SF Symbol as a black-on-transparent PNG at 240 pt, medium weight
import AppKit
let a = CommandLine.arguments
_ = NSApplication.shared
let cfg = NSImage.SymbolConfiguration(pointSize: 240, weight: .medium)
guard let img = NSImage(systemSymbolName: a[1], accessibilityDescription: nil)?.withSymbolConfiguration(cfg) else { print("no symbol \(a[1])"); exit(1) }
let size = img.size
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size.width * 2), pixelsHigh: Int(size.height * 2), bitsPerSample: 8,
                           samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
rep.size = size
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
img.draw(in: NSRect(origin: .zero, size: size))
NSGraphicsContext.restoreGraphicsState()
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: a[2]))
