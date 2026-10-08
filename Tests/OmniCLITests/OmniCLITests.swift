import XCTest
@testable import omni

/// The `omni` command turns any tool's input schema into a subcommand: these pin the mapping, so
/// a tool added to the app is usable from the command line and the skill without code here.
final class OmniCLITests: XCTestCase {
    let search: [String: Any] = [
        "name": "search", "title": "Search", "description": "Find files.",
        "inputSchema": [
            "type": "object",
            "properties": [
                "query": ["type": "string", "description": "What to find."],
                "top_k": ["type": "integer", "description": "How many."],
                "min_score": ["type": "number"],
                "group_duplicates": ["type": "boolean"],
                "kinds": ["type": "array", "items": ["type": "string", "enum": ["text", "image"]]],
                "modified_after": ["type": "string"],
            ] as [String: Any],
            "required": ["query"],
        ] as [String: Any],
    ]
    let status: [String: Any] = [
        "name": "file_status",
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

    func testMistakesSayWhatIsExpected() {
        XCTAssertThrowsError(try arguments(for: search, ["q", "--topk", "3"])) {
            XCTAssertTrue(($0 as? Failure)?.message.contains("--top-k") ?? false, "lists the real options")
        }
        XCTAssertThrowsError(try arguments(for: search, ["q", "--top-k", "five"])) {
            XCTAssertTrue(($0 as? Failure)?.message.contains("whole number") ?? false)
        }
        XCTAssertThrowsError(try arguments(for: search, ["--top-k", "3"])) {
            XCTAssertTrue(($0 as? Failure)?.message.contains("missing <query>") ?? false)
        }
    }

    func testTheSkillHasEveryToolAndFlag() {
        let md = AgentSkill.render(instructions: "Use it well.", tools: [search, status], command: "/X/omni")
        XCTAssertTrue(md.hasPrefix("---\nname: omni-local-search\n"))
        XCTAssertTrue(md.contains("Use it well."))
        XCTAssertTrue(md.contains("## search") && md.contains("## file_status"))
        XCTAssertTrue(md.contains("/X/omni search <query> [options]"))
        XCTAssertTrue(md.contains("/X/omni file_status <paths>..."))
        for flag in ["--top-k <integer>", "--min-score <number>", "--group-duplicates`", "--kinds <text|image>", "--modified-after <string>"] {
            XCTAssertTrue(md.contains(flag), flag)
        }
    }
}
