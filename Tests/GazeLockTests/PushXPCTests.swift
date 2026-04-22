import XCTest
@testable import GazeLock

final class PushXPCTests: XCTestCase {
    func testClientPushUpdatesCachedState() throws {
        let client = ControlClient()
        var state = ControlState.default
        state.intensity = 0.33
        state.sensitivity = .tight
        let data = try JSONEncoder().encode(state)

        let expectation = XCTestExpectation(description: "ack")
        client.pushControlState(data) { ok in
            XCTAssertTrue(ok)
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)

        let cached = client.currentState()
        XCTAssertEqual(cached.intensity, 0.33, accuracy: 0.001)
        XCTAssertEqual(cached.sensitivity, .tight)
    }

    func testClientRejectsMalformedPayload() {
        let client = ControlClient()
        let bogus = Data([0xDE, 0xAD])
        let expectation = XCTestExpectation(description: "ack")
        client.pushControlState(bogus) { ok in
            XCTAssertFalse(ok)
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
    }

    func testServerBroadcastDoesNotCrashWithNoConnections() {
        let server = ControlServer { .default }
        server.broadcast(.default)  // no active connections; should be a no-op
    }

    func testClientPingAcks() {
        let client = ControlClient()
        let expectation = XCTestExpectation(description: "ack")
        client.ping { ok in
            XCTAssertTrue(ok)
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
    }

    func testServerRejectsClientPush() {
        let server = ControlServer { .default }
        let expectation = XCTestExpectation(description: "ack")
        server.pushControlState(Data()) { ok in
            XCTAssertFalse(ok, "Server should refuse client-to-server push")
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
    }
}
