import Foundation

/// One page of a document read a page at a time: a slide, or a sheet of a workbook.
public struct DocumentPage: Sendable {
    /// What the page is called where it is shown: "Slide 3", or the sheet's name.
    public let label: String
    /// A slide's words, its speaker notes after them; a Numbers sheet's text. Empty for a sheet
    /// read as cells.
    public let text: String
    /// A sheet's cells, by row, in sheet order. Empty for a slide.
    public let rows: [SheetRow]
    public init(label: String, text: String = "", rows: [SheetRow] = []) {
        self.label = label; self.text = text; self.rows = rows
    }
}

public struct SheetRow: Sendable {
    /// 1-based, as the sheet numbers it.
    public let row: Int
    /// The non-empty cells, left to right; `column` is 1-based (A = 1).
    public let cells: [SheetCell]
}

public struct SheetCell: Sendable {
    public let column: Int
    public let value: String
}

/// The text of a PowerPoint deck or an Excel workbook, read straight from its archive: a page a
/// slide or a sheet, so every piece of it is found at its slide or its cells. Both are zips of XML
/// (Office Open XML); the zip reader is IWorkText's and the XML is Foundation's parser. Only the
/// parts that hold words are inflated: a deck's pictures are never touched.
enum OfficeDocument {

    static func extract(_ url: URL, maxBytes: Int = FileExtractor.maxTextBytes) -> [DocumentPage]? {
        guard let z = try? Data(contentsOf: url, options: .alwaysMapped) else { return nil }
        switch url.pathExtension.lowercased() {
        case "pptx": return slides(z, maxBytes: maxBytes)
        case "xlsx": return sheets(z, maxBytes: maxBytes)
        default: return nil
        }
    }

    // MARK: - Slides

    static func slides(_ z: Data, maxBytes: Int) -> [DocumentPage]? {
        let parts = Dictionary(Zip.entries(z) { n in
            n == "ppt/presentation.xml" || n == "ppt/_rels/presentation.xml.rels"
                || (n.hasPrefix("ppt/slides/") || n.hasPrefix("ppt/notesSlides/")) && (n.hasSuffix(".xml") || n.hasSuffix(".rels"))
        }.map { ($0.name, $0.data) }, uniquingKeysWith: { a, _ in a })
        guard let pres = parts["ppt/presentation.xml"], let presRels = parts["ppt/_rels/presentation.xml.rels"] else { return nil }
        let targets = Relationships.parse(presRels)
        var out: [DocumentPage] = []
        var total = 0
        for (n, rid) in SlideOrder.parse(pres).enumerated() {
            guard total < maxBytes, let target = targets[rid]?.target else { continue }
            let path = Relationships.resolve(target, from: "ppt/")
            guard let xml = parts[path] else { continue }
            var text = DrawingText.parse(xml)
            // The notes, through the slide's own relationships.
            let name = (path as NSString).lastPathComponent
            if let rels = parts["ppt/slides/_rels/\(name).rels"],
               let notes = Relationships.parse(rels).values.first(where: { $0.type.hasSuffix("/notesSlide") }),
               let nxml = parts[Relationships.resolve(notes.target, from: "ppt/slides/")] {
                let said = DrawingText.parse(nxml)
                if !said.isEmpty { text += text.isEmpty ? said : "\n\n" + said }
            }
            total += text.utf8.count
            out.append(DocumentPage(label: "Slide \(n + 1)", text: text))
        }
        return out
    }

    /// The slides in the order the deck shows them: `p:sldIdLst`, each a relationship id.
    private final class SlideOrder: NSObject, XMLParserDelegate {
        var ids: [String] = []
        static func parse(_ data: Data) -> [String] {
            let d = SlideOrder()
            let p = XMLParser(data: data)
            p.shouldProcessNamespaces = true
            p.delegate = d
            p.parse()
            return d.ids
        }
        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                    qualifiedName: String?, attributes: [String: String]) {
            guard name == "sldId" else { return }
            // r:id, whatever the prefix is bound to.
            if let rid = attributes.first(where: { $0.key == "r:id" || $0.key.hasSuffix(":id") })?.value { ids.append(rid) }
        }
    }

    /// A slide's or notes page's words: each `a:p` a line, its runs and fields in order, a break a
    /// newline. The slide number and date placeholders are left out (they hold "‹#›" or a field).
    private final class DrawingText: NSObject, XMLParserDelegate {
        var lines: [String] = []
        var line = ""
        var inText = false
        var skipDepth = 0        // inside a placeholder shape whose text is not the slide's own
        var shapeDepth = 0
        var shapeSkipped = false
        static func parse(_ data: Data) -> String {
            let d = DrawingText()
            let p = XMLParser(data: data)
            p.shouldProcessNamespaces = true
            p.delegate = d
            p.parse()
            return d.lines.joined(separator: "\n")
        }
        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                    qualifiedName: String?, attributes: [String: String]) {
            switch name {
            case "sp": shapeDepth += 1; if shapeDepth == 1 { shapeSkipped = false }
            case "ph": if let t = attributes["type"], ["sldNum", "dt", "sldImg", "hdr"].contains(t) { shapeSkipped = true }
            case "p": line = ""
            case "t": inText = true
            case "br": line += "\n"
            case "tab": line += "\t"
            default: break
            }
        }
        func parser(_ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?) {
            switch name {
            case "sp": shapeDepth = max(0, shapeDepth - 1)
            case "t": inText = false
            case "p":
                let s = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if !s.isEmpty, !shapeSkipped || shapeDepth == 0 { lines.append(s) }
                line = ""
            default: break
            }
        }
        func parser(_ parser: XMLParser, foundCharacters string: String) {
            if inText { line += string }
        }
    }

    // MARK: - Sheets

    static func sheets(_ z: Data, maxBytes: Int) -> [DocumentPage]? {
        // The worksheets are inflated one at a time, below: a workbook's sheets can each be large.
        let parts = Dictionary(Zip.entries(z) { n in
            n == "xl/workbook.xml" || n == "xl/_rels/workbook.xml.rels" || n == "xl/sharedStrings.xml" || n == "xl/styles.xml"
        }.map { ($0.name, $0.data) }, uniquingKeysWith: { a, _ in a })
        guard let wbData = parts["xl/workbook.xml"], let rels = parts["xl/_rels/workbook.xml.rels"] else { return nil }
        let wb = Workbook.parse(wbData)
        let targets = Relationships.parse(rels)
        let strings = parts["xl/sharedStrings.xml"].map(SharedStrings.parse) ?? []
        let dates = parts["xl/styles.xml"].map(DateStyles.parse) ?? []
        var out: [DocumentPage] = []
        var budget = maxBytes
        for sheet in wb.sheets {
            guard budget > 0, let target = targets[sheet.rid]?.target else { continue }
            let path = Relationships.resolve(target, from: "xl/")
            guard let xml = Zip.entries(z, where: { $0 == path }).first?.data else { continue }
            let rows = SheetCells.parse(xml, strings: strings, dateStyles: dates, date1904: wb.date1904, budget: &budget)
            if !rows.isEmpty { out.append(DocumentPage(label: sheet.name, rows: rows)) }
        }
        return out
    }

    private final class Workbook: NSObject, XMLParserDelegate {
        var sheets: [(name: String, rid: String)] = []
        var date1904 = false
        static func parse(_ data: Data) -> Workbook {
            let d = Workbook()
            let p = XMLParser(data: data)
            p.shouldProcessNamespaces = true
            p.delegate = d
            p.parse()
            return d
        }
        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                    qualifiedName: String?, attributes: [String: String]) {
            if name == "sheet", let n = attributes["name"],
               let rid = attributes.first(where: { $0.key == "r:id" || $0.key.hasSuffix(":id") })?.value {
                sheets.append((n, rid))
            } else if name == "workbookPr", let v = attributes["date1904"] {
                date1904 = v == "1" || v == "true"
            }
        }
    }

    /// `xl/sharedStrings.xml`: each `si` one string, its runs joined; phonetic runs (`rPh`) left out.
    private final class SharedStrings: NSObject, XMLParserDelegate {
        var out: [String] = []
        var cur = ""
        var inT = false, inPhonetic = false
        static func parse(_ data: Data) -> [String] {
            let d = SharedStrings()
            let p = XMLParser(data: data)
            p.shouldProcessNamespaces = true
            p.delegate = d
            p.parse()
            return d.out
        }
        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                    qualifiedName: String?, attributes: [String: String]) {
            switch name {
            case "si": cur = ""
            case "t": inT = true
            case "rPh": inPhonetic = true
            default: break
            }
        }
        func parser(_ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?) {
            switch name {
            case "si": out.append(cur)
            case "t": inT = false
            case "rPh": inPhonetic = false
            default: break
            }
        }
        func parser(_ parser: XMLParser, foundCharacters string: String) {
            if inT, !inPhonetic { cur += string }
        }
    }

    /// Which cell styles show a date. A cell holds a date as a day count; only its number format says
    /// so: the built-in date formats (14-22, 45-47) or a custom one whose code has a day, month,
    /// year or hour outside quotes and brackets.
    private final class DateStyles: NSObject, XMLParserDelegate {
        var custom: [Int: String] = [:]
        var xfFormats: [Int] = []
        var inCellXfs = false
        static func parse(_ data: Data) -> [Bool] {
            let d = DateStyles()
            let p = XMLParser(data: data)
            p.shouldProcessNamespaces = true
            p.delegate = d
            p.parse()
            return d.xfFormats.map { id in
                if (14 ... 22).contains(id) || (45 ... 47).contains(id) { return true }
                guard let code = d.custom[id] else { return false }
                return isDateFormat(code)
            }
        }
        static func isDateFormat(_ code: String) -> Bool {
            var inQuote = false, inBracket = false, prev: Character = " "
            for ch in code {
                if inQuote { if ch == "\"" { inQuote = false }; continue }
                if inBracket { if ch == "]" { inBracket = false }; continue }
                if prev == "\\" { prev = " "; continue }
                switch ch {
                case "\"": inQuote = true
                case "[": inBracket = true
                case "d", "D", "m", "M", "y", "Y", "h", "H": return true
                default: break
                }
                prev = ch
            }
            return false
        }
        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                    qualifiedName: String?, attributes: [String: String]) {
            switch name {
            case "numFmt":
                if let id = attributes["numFmtId"].flatMap(Int.init), let code = attributes["formatCode"] { custom[id] = code }
            case "cellXfs": inCellXfs = true
            case "xf": if inCellXfs { xfFormats.append(attributes["numFmtId"].flatMap(Int.init) ?? 0) }
            default: break
            }
        }
        func parser(_ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?) {
            if name == "cellXfs" { inCellXfs = false }
        }
    }

    /// A worksheet's cells as the sheet shows their values: shared and inline strings, booleans,
    /// errors, formula results, numbers without float noise and dates as dates. Stops once
    /// `budget` bytes have been read.
    private final class SheetCells: NSObject, XMLParserDelegate {
        let strings: [String], dateStyles: [Bool], date1904: Bool
        var budget: Int
        var rows: [SheetRow] = []
        var cells: [SheetCell] = []
        var rowNumber = 0
        var nextColumn = 1
        var cellColumn = 0, cellType = "", cellStyle = 0
        var value = "", inline = ""
        var inV = false, inInlineT = false, inIs = false
        weak var parser: XMLParser?
        init(strings: [String], dateStyles: [Bool], date1904: Bool, budget: Int) {
            self.strings = strings; self.dateStyles = dateStyles; self.date1904 = date1904; self.budget = budget
        }
        static func parse(_ data: Data, strings: [String], dateStyles: [Bool], date1904: Bool, budget: inout Int) -> [SheetRow] {
            let d = SheetCells(strings: strings, dateStyles: dateStyles, date1904: date1904, budget: budget)
            let p = XMLParser(data: data)
            p.shouldProcessNamespaces = true
            p.delegate = d
            d.parser = p
            p.parse()
            budget = d.budget
            return d.rows
        }
        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                    qualifiedName: String?, attributes: [String: String]) {
            switch name {
            case "row":
                rowNumber = attributes["r"].flatMap(Int.init) ?? rowNumber + 1
                cells = []; nextColumn = 1
            case "c":
                cellColumn = attributes["r"].map(Self.column) ?? nextColumn
                if cellColumn == 0 { cellColumn = nextColumn }
                cellType = attributes["t"] ?? ""
                cellStyle = attributes["s"].flatMap(Int.init) ?? 0
                value = ""; inline = ""
            case "v": inV = true
            case "is": inIs = true
            case "t": if inIs { inInlineT = true }
            default: break
            }
        }
        func parser(_ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?) {
            switch name {
            case "v": inV = false
            case "t": inInlineT = false
            case "is": inIs = false
            case "c":
                nextColumn = cellColumn + 1
                let s = shown().trimmingCharacters(in: .whitespacesAndNewlines)
                if !s.isEmpty { cells.append(SheetCell(column: cellColumn, value: s)); budget -= s.utf8.count + 3 }
            case "row":
                if !cells.isEmpty { rows.append(SheetRow(row: rowNumber, cells: cells)) }
                cells = []
                if budget <= 0 { self.parser?.abortParsing() }
            default: break
            }
        }
        func parser(_ parser: XMLParser, foundCharacters string: String) {
            if inV { value += string } else if inInlineT { inline += string }
        }

        private func shown() -> String {
            switch cellType {
            case "s": return Int(value).flatMap { $0 >= 0 && $0 < strings.count ? strings[$0] : nil } ?? ""
            case "inlineStr": return inline
            case "b": return value == "1" ? "TRUE" : value == "0" ? "FALSE" : value
            case "str", "e", "d": return value
            default:
                guard let x = Double(value) else { return value }
                if cellStyle < dateStyles.count, dateStyles[cellStyle] { return Self.date(x, date1904: date1904) ?? value }
                return Self.number(x)
            }
        }

        /// A number as written, without the noise of its binary form (0.1 + 0.2 stored as
        /// 0.30000000000000004 reads 0.3).
        static func number(_ x: Double) -> String {
            if x == x.rounded(), abs(x) < 1e15 { return String(Int64(x)) }
            var s = String(format: "%.15g", x)
            if s.contains("e") { return s }
            if s.contains(".") { while s.hasSuffix("0") { s.removeLast() }; if s.hasSuffix(".") { s.removeLast() } }
            return s
        }

        /// A day count as a date (with the time when it has one): 1900 system by default, where day
        /// 60 is the 29 February 1900 Lotus counted and day 61 is 1 March.
        static func date(_ serial: Double, date1904: Bool) -> String? {
            guard serial >= 0, serial < 2_958_466 else { return nil }
            var days = serial.rounded(.down)
            let seconds = Int(((serial - days) * 86_400).rounded())
            if !date1904, days < 61 { days += 1 }   // before the phantom leap day
            var cal = Calendar(identifier: .gregorian)
            cal.timeZone = TimeZone(identifier: "UTC")!
            let base = date1904 ? DateComponents(year: 1904, month: 1, day: 1) : DateComponents(year: 1899, month: 12, day: 30)
            guard let start = cal.date(from: base),
                  let d = cal.date(byAdding: .second, value: Int(days) * 86_400 + seconds, to: start) else { return nil }
            let c = cal.dateComponents([.year, .month, .day, .hour, .minute, .second], from: d)
            let ymd = String(format: "%04d-%02d-%02d", c.year ?? 0, c.month ?? 0, c.day ?? 0)
            guard seconds > 0 else { return ymd }
            return ymd + String(format: " %02d:%02d", c.hour ?? 0, c.minute ?? 0) + ((c.second ?? 0) > 0 ? String(format: ":%02d", c.second ?? 0) : "")
        }

        /// "BC12" -> 55. 0 when the reference has no letters.
        static func column(_ ref: String) -> Int {
            var n = 0
            for u in ref.unicodeScalars {
                let v = u.value
                if v >= 65, v <= 90 { n = n * 26 + Int(v - 64) }
                else if v >= 97, v <= 122 { n = n * 26 + Int(v - 96) }
                else { break }
            }
            return n
        }
    }

    /// 55 -> "BC".
    static func columnName(_ n: Int) -> String {
        var n = n, s = ""
        while n > 0 { let r = (n - 1) % 26; s = String(UnicodeScalar(UInt8(65 + r))) + s; n = (n - 1) / 26 }
        return s
    }

    // MARK: - Relationships

    /// An Office part's `_rels`: relationship id -> (type, target).
    private final class Relationships: NSObject, XMLParserDelegate {
        var out: [String: (type: String, target: String)] = [:]
        static func parse(_ data: Data) -> [String: (type: String, target: String)] {
            let d = Relationships()
            let p = XMLParser(data: data)
            p.shouldProcessNamespaces = true
            p.delegate = d
            p.parse()
            return d.out
        }
        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                    qualifiedName: String?, attributes: [String: String]) {
            guard name == "Relationship", let id = attributes["Id"], let t = attributes["Target"],
                  attributes["TargetMode"] != "External" else { return }
            out[id] = (attributes["Type"] ?? "", t)
        }

        /// A target relative to the part's folder ("slides/slide1.xml", "../notesSlides/x.xml"), or
        /// absolute from the package root ("/xl/worksheets/sheet1.xml").
        static func resolve(_ target: String, from folder: String) -> String {
            if target.hasPrefix("/") { return String(target.dropFirst()) }
            var parts = folder.split(separator: "/").map(String.init)
            for c in target.split(separator: "/") {
                if c == ".." { if !parts.isEmpty { parts.removeLast() } } else if c != "." { parts.append(String(c)) }
            }
            return parts.joined(separator: "/")
        }
    }
}
