import Foundation
import XCTest

@testable import Rapid

/// Issue #17 desktop-half: the bearer mode configuration (random / fixed /
/// off) must behave predictably, and the restart-detection that the
/// Settings panel relies on must never dead-loop.
///
/// Each test gets a fresh, uniquely-named UserDefaults suite injected into
/// ``APIAuthConfig.defaults``; tear-down always restores `.standard` so the
/// developer's real preferences are never touched.
final class APIAuthConfigTests: XCTestCase {

    private var suiteName: String?
    private var suite: UserDefaults?

    override func setUpWithError() throws {
        try super.setUpWithError()
        let name = "APIAuthConfigTests-\(UUID().uuidString)"
        let fresh = try XCTUnwrap(UserDefaults(suiteName: name),
                                  "suite creation must succeed")
        fresh.removePersistentDomain(forName: name)
        suiteName = name
        suite = fresh
        APIAuthConfig.defaults = fresh
    }

    override func tearDownWithError() throws {
        if let suiteName {
            suite?.removePersistentDomain(forName: suiteName)
        }
        APIAuthConfig.defaults = .standard
        suite = nil
        suiteName = nil
        try super.tearDownWithError()
    }

    // MARK: - mode storage

    func testModeDefaultsToRandom() {
        XCTAssertEqual(APIAuthConfig.mode, .random)
    }

    func testModeRoundTrips() {
        APIAuthConfig.mode = .fixed
        XCTAssertEqual(APIAuthConfig.mode, .fixed)
        APIAuthConfig.mode = .off
        XCTAssertEqual(APIAuthConfig.mode, .off)
    }

    // MARK: - effective mode (fixed-without-key => random)

    func testEffectiveModeFallsBackToRandom() {
        APIAuthConfig.mode = .fixed
        XCTAssertEqual(APIAuthConfig.effectiveMode, .random,
                       "fixed with no stored key must never claim to be fixed")
    }

    func testEffectiveModeHonorsStoredKey() {
        APIAuthConfig.mode = .fixed
        APIAuthConfig.defaults.set("abc123", forKey: APIAuthConfig.fixedKeyStorageKey)
        XCTAssertEqual(APIAuthConfig.effectiveMode, .fixed)
    }

    func testEmptyKeyIsNotStored() {
        APIAuthConfig.defaults.set("", forKey: APIAuthConfig.fixedKeyStorageKey)
        XCTAssertFalse(APIAuthConfig.hasStoredFixedKey)
        XCTAssertEqual(APIAuthConfig.defaults.string(forKey: APIAuthConfig.fixedKeyStorageKey), "")
    }

    // MARK: - bearerForSpawn()

    func testBearerForSpawnRandom() throws {
        APIAuthConfig.mode = .random
        let bearer = try XCTUnwrap(APIAuthConfig.bearerForSpawn())
        XCTAssertEqual(bearer.count, 64)
        let hexChars: Set<Character> = Set("0123456789abcdef")
        XCTAssertTrue(bearer.allSatisfy { hexChars.contains($0) })
    }

    func testBearerForSpawnFixed() {
        APIAuthConfig.mode = .fixed
        APIAuthConfig.defaults.set("my-fixed-key", forKey: APIAuthConfig.fixedKeyStorageKey)
        XCTAssertEqual(APIAuthConfig.bearerForSpawn(), "my-fixed-key")
    }

    func testBearerForSpawnFixedFallback() throws {
        APIAuthConfig.mode = .fixed
        let bearer = try XCTUnwrap(APIAuthConfig.bearerForSpawn(),
                                   "fixed without a stored key must still auth (random), not nil")
        XCTAssertEqual(bearer.count, 64)
    }

    func testBearerForSpawnOff() {
        APIAuthConfig.mode = .off
        XCTAssertNil(APIAuthConfig.bearerForSpawn())
    }

    // MARK: - needsRestart (P1-1 dead-loop regression)

    func testNeedsRestartFalseWhenNotServing() {
        XCTAssertFalse(APIAuthConfig.needsRestart(activeMode: .random, activeBearer: "x", isServing: false))
        XCTAssertFalse(APIAuthConfig.needsRestart(activeMode: nil, activeBearer: nil, isServing: true))
    }

    func testNeedsRestartFalseWhenInEffect() {
        APIAuthConfig.mode = .random
        XCTAssertFalse(APIAuthConfig.needsRestart(activeMode: .random, activeBearer: "abc", isServing: true))

        APIAuthConfig.mode = .off
        XCTAssertFalse(APIAuthConfig.needsRestart(activeMode: .off, activeBearer: nil, isServing: true))
    }

    func testNeedsRestartTrueWhenModeDiffers() {
        APIAuthConfig.mode = .off
        XCTAssertTrue(APIAuthConfig.needsRestart(activeMode: .random, activeBearer: "abc", isServing: true))

        APIAuthConfig.mode = .fixed
        APIAuthConfig.defaults.set("abc123", forKey: APIAuthConfig.fixedKeyStorageKey)
        XCTAssertTrue(APIAuthConfig.needsRestart(activeMode: .random, activeBearer: "abc", isServing: true))
    }

    func testNoDeadLoopForUnstoredFixed() {
        // Pre-review bug (handoff P1 #1): mode fixed + no stored key made
        // the engine spawn random, but the panel compared mode (fixed) to
        // activeAuthMode (random) and demanded a restart forever, showing
        // "Applied" every time while never converging.
        APIAuthConfig.mode = .fixed
        XCTAssertEqual(APIAuthConfig.effectiveMode, .random)
        XCTAssertFalse(APIAuthConfig.needsRestart(activeMode: .random, activeBearer: "fresh", isServing: true),
                       "fixed-without-key must read as already-in-effect once the engine runs random")
    }

    func testNeedsRestartTrueWhenFixedKeyChanged() {
        APIAuthConfig.mode = .fixed
        APIAuthConfig.defaults.set("new-key", forKey: APIAuthConfig.fixedKeyStorageKey)
        // Engine still holds the OLD key from its spawn.
        XCTAssertTrue(APIAuthConfig.needsRestart(activeMode: .fixed, activeBearer: "old-key", isServing: true))
        // Engine holds the current key — no restart needed.
        XCTAssertFalse(APIAuthConfig.needsRestart(activeMode: .fixed, activeBearer: "new-key", isServing: true))
    }
}
