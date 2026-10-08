import Foundation

// Where the runner finds every body this build has.
//
// The two body files are deliberately ignorant of each other: `PaperCasesCompute` owns the cases
// whose cost is the model, `PaperCasesStore` the ones whose cost is the vector store, and
// `BenchCases` the task-table cases on the generated one-million-row store. None imports another,
// so any can be edited, or left out of a build, without touching the others. This type is the only place that knows both exist.
//
// Lookup order is compute-then-store, and it is checked rather than assumed: `contestedIDs` names
// any case both files claim. Two providers answering for one id would mean the export's numbers
// came from whichever file happened to be asked first, which is exactly the kind of silent
// substitution the suite refuses everywhere else.
public struct PaperAllCaseBodies: PaperCaseBodies {
    public init() {}

    public func body(for id: PaperCaseID) -> PaperCaseBody? {
        PaperCasesCompute.body(for: id) ?? PaperCasesStore.body(for: id) ?? BenchCases.body(for: id)
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
        [PaperCasesCompute.body(for: id), PaperCasesStore.body(for: id), BenchCases.body(for: id)]
            .reduce(0) { $0 + ($1 == nil ? 0 : 1) }
    }
}
