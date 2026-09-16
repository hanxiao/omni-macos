import AppKit
for n in ["doc.badge.gearshape","doc.badge.ellipsis","exclamationmark.magnifyingglass"] {
  print("\(NSImage(systemSymbolName: n, accessibilityDescription: nil) != nil ? "OK  " : "MISS") \(n)")
}
