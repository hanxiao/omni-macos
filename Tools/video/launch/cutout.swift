// cutout <in.png> <out.png>: lift the foreground subjects of a generated plate into an RGBA sprite
// with Apple Vision's subject mask (the same one Photos uses for "Lift Subject").
import Foundation
import Vision
import CoreImage
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
guard args.count == 3 else { print("usage: cutout <in> <out>"); exit(2) }
let input = CIImage(contentsOf: URL(fileURLWithPath: args[1]))!
let request = VNGenerateForegroundInstanceMaskRequest()
let handler = VNImageRequestHandler(ciImage: input)
try handler.perform([request])
guard let result = request.results?.first else { print("no subject"); exit(1) }
let buffer = try result.generateMaskedImage(ofInstances: result.allInstances, from: handler,
                                            croppedToInstancesExtent: false)
let out = CIImage(cvPixelBuffer: buffer)
let ctx = CIContext()
let cg = ctx.createCGImage(out, from: out.extent, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)!
let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: args[2]) as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(dest, cg, nil)
CGImageDestinationFinalize(dest)
print("ok \(result.allInstances.count) instances")
