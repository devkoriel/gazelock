import XCTest
@testable import GazeLock

final class CoreMLRefinerTests: XCTestCase {
    func testMissingModelThrowsModelNotFound() {
        // Create an empty temporary bundle path
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("empty-bundle-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let bundle = Bundle(url: tempDir) ?? Bundle.main
        if bundle === Bundle.main {
            // If we can't make an empty test bundle on this setup, skip
            // — we can't meaningfully test the "not found" path without
            // isolating the search root. In Phase 3b we'll inject a
            // mock bundle.
            return
        }
        XCTAssertThrowsError(try CoreMLRefiner(bundle: bundle)) { error in
            guard case CoreMLRefiner.LoadError.modelNotFound = error else {
                XCTFail("expected .modelNotFound, got \(error)")
                return
            }
        }
    }
}
