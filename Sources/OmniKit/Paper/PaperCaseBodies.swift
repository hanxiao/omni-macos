import Foundation

// Where the runner finds every body this build has.
//
// The two body files are deliberately ignorant of each other: `PaperCasesCompute` owns the cases
// whose cost is the model (p01-p05, p11, p12) and `PaperCasesStore` owns the ones whose cost is the
// vector store (p06-p10). Neither imports the other, so either can be edited, or left out of a
// build, without touching the other. This type is the only place that knows both exist.
//
// Lookup order is compute-then-store, and it is checked rather than assumed: `contestedIDs` names
// any case both files claim. Two providers answering for one id would mean the export's numbers
// came from whichever file happened to be asked first, which is exactly the kind of silent
// substitution the suite refuses everywhere else.
public struct PaperAllCaseBodies: PaperCaseBodies {
    public init() {}

    public func body(for id: PaperCaseID) -> PaperCaseBody? {
        PaperCasesCompute.body(for: id) ?? PaperCasesStore.body(for: id) ?? PaperCasesLive.body(for: id)
    }

    /// Cases with no body in this build. They record `skipped:unimplemented`, never a measured zero.
    public static var missingIDs: [PaperCaseID] {
        PaperCaseID.allCases.filter { providerCount(for: $0) == 0 }
    }

    /// Cases claimed by more than one provider. Must be empty; a runner should say so loudly rather
    /// than quietly measuring one of the implementations.
    public static var contestedIDs: [PaperCaseID] {
        PaperCaseID.allCases.filter { providerCount(for: $0) > 1 }
    }

    private static func providerCount(for id: PaperCaseID) -> Int {
        [PaperCasesCompute.body(for: id), PaperCasesStore.body(for: id), PaperCasesLive.body(for: id)]
            .reduce(0) { $0 + ($1 == nil ? 0 : 1) }
    }
}
