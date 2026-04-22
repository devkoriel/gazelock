import Foundation
import Observation

/// Persists UserCalibration to Application Support. Observable for SwiftUI.
@MainActor
@Observable
public final class CalibrationStore {
    public private(set) var current: UserCalibration?

    private let fileURL: URL

    public init(fileURL: URL? = nil) {
        self.fileURL = fileURL ?? Self.defaultFileURL()
        self.current = Self.load(from: self.fileURL)
    }

    public func save(_ cal: UserCalibration) {
        current = cal
        try? FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if let data = try? JSONEncoder().encode(cal) {
            try? data.write(to: fileURL)
        }
    }

    public func clear() {
        current = nil
        try? FileManager.default.removeItem(at: fileURL)
    }

    private static func defaultFileURL() -> URL {
        let support = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        return support.appendingPathComponent("GazeLock/calibration.json")
    }

    private static func load(from url: URL) -> UserCalibration? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(UserCalibration.self, from: data)
    }
}
