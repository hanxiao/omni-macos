import XCTest
@testable import omni

/// The `omni` command turns any tool's input schema into a subcommand: these pin the mapping, the
/// help layers and the error wording, so a tool added to the app is usable from the command line
/// and the skill without code here.
final class OmniCLITests: XCTestCase {
    let search: [String: Any] = [
        "name": "search", "title": "Search", "description": "Find files. Longer explanation here.",
        "inputSchema": [
            "type": "object",
            "properties": [
                "query": ["type": "string", "description": "What to find."],
                "top_k": ["type": "integer", "description": "How many. At most 50."],
                "min_score": ["type": "number"],
                "group_duplicates": ["type": "boolean"],
                "kinds": ["type": "array", "items": ["type": "string", "enum": ["text", "image"]]],
                "modified_after": ["type": "string"],
            ] as [String: Any],
            "required": ["query"],
        ] as [String: Any],
        "_meta": [CLIProtocol.examplesKey: [["query": "red car", "kinds": ["image"], "top_k": 5],
                                            ["query": "notes", "group_duplicates": false]]],
    ]
    let status: [String: Any] = [
        "name": "file_status", "title": "Status",
        "inputSchema": ["type": "object",
                        "properties": ["paths": ["type": "array", "items": ["type": "string"]]],
                        "required": ["paths"]] as [String: Any],
    ]

    func testTheFirstRequiredParameterTakesTheBareWords() throws {
        let a = try arguments(for: search, ["red", "sports", "car", "--top-k", "5"])
        XCTAssertEqual(a["query"] as? String, "red sports car")
        XCTAssertEqual(a["top_k"] as? Int, 5)
        let s = try arguments(for: status, ["/a.pdf", "/b.png"])
        XCTAssertEqual(s["paths"] as? [String], ["/a.pdf", "/b.png"])
    }

    func testFlagsFollowTheSchemaTypes() throws {
        let a = try arguments(for: search, ["q", "--min-score=0.3", "--no-group-duplicates", "--kinds", "text",
                                            "--kinds", "image", "--modified-after", "2026-10-01"])
        XCTAssertEqual(a["min_score"] as? Double, 0.3)
        XCTAssertEqual(a["group_duplicates"] as? Bool, false)
        XCTAssertEqual(a["kinds"] as? [String], ["text", "image"])
        XCTAssertEqual(a["modified_after"] as? String, "2026-10-01")
        XCTAssertEqual(try arguments(for: search, ["q", "--group-duplicates"])["group_duplicates"] as? Bool, true)
        XCTAssertEqual(try arguments(for: search, ["q", "--top_k", "3"])["top_k"] as? Int, 3)   // the schema's spelling too
    }

    /// One bad command should cost one retry: every mistake names the fix.
    func testMistakesSayHowToFixThem() {
        func message(_ words: [String]) -> String {
            do { _ = try arguments(for: search, words); return "" } catch { return (error as? Failure)?.message ?? "" }
        }
        XCTAssertTrue(message(["q", "--topk", "3"]).contains("Did you mean --top-k?"))
        XCTAssertTrue(message(["q", "--top-k", "five"]).contains("whole number"))
        XCTAssertTrue(message(["q", "--kinds", "imgae"]).contains("Did you mean image?"))
        let missing = message(["--top-k", "3"])
        XCTAssertTrue(missing.contains("missing <query>") && missing.contains("Example: omni search"), missing)
        XCTAssertEqual(suggestion("serach", ["search", "search_inline", "ocr"]), "search")
        XCTAssertNil(suggestion("zzzzzz", ["search", "ocr"]))
    }

    func testServerErrorsAreRewordedAsFlags() {
        let m = inCLITerms("search failed: 'modified_after' must be an ISO 8601 date; 'query' is required", tool: search)
        XCTAssertEqual(m, "omni search: --modified-after must be an ISO 8601 date; <query> is required")
    }

    func testHelpComesInLayers() {
        let short = toolHelp(search, full: false), full = toolHelp(search, full: true)
        XCTAssertTrue(short.contains("Find files.") && !short.contains("Longer explanation"))
        XCTAssertTrue(short.contains("omni search 'red car' --kinds image --top-k 5"), short)
        XCTAssertTrue(short.contains("omni search notes --no-group-duplicates"), short)
        XCTAssertTrue(short.contains("How many.") && !short.contains("At most 50"))
        XCTAssertTrue(short.contains("--help-all"))
        XCTAssertTrue(full.contains("Longer explanation") && full.contains("At most 50"))
    }

    /// The skill names the commands and shows examples; options stay behind --help.
    func testTheSkillIsProgressive() {
        let md = AgentSkill.render(instructions: "Use it well.", tools: [search, status], command: "/X/omni")
        XCTAssertTrue(md.hasPrefix("---\nname: omni-local-search\n"))
        XCTAssertTrue(md.contains("Use it well."))
        XCTAssertTrue(md.contains("  search       Search") && md.contains("  file_status  Status"), md)
        XCTAssertTrue(md.contains("/X/omni search 'red car' --kinds image --top-k 5"))
        XCTAssertTrue(md.contains("--help-all") && md.contains("--json") && md.contains("Exit codes"))
        XCTAssertFalse(md.contains("--min-score"), "options are for --help, not the skill")
    }
}
