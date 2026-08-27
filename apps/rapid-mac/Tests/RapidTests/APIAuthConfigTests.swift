import Foundation
import XCTest

@testable import Rapid

/// In-memory ``KeychainStoring`` double so the suite never touches
/// Security.framework (which would prompt, leak across tests, and require
/// manual cleanup). Upsert semantics mirror ``SystemKeychain.write``:
/// a write replaces any existing item for the same account.
private final class InMemoryKeychain: KeychainStoring {
    private var items: [String: String] = [:]

    /// Test seam: simulate a refused Keychain write (user denied the prompt).
    var failWrites = false

    /// Test seam: simulate an ACL-refused Keychain delete.
    var failDeletes = false

    func read(account: String) -> String? { items[account] }

    func readWithoutUserInteraction(account: String) -> KeychainReadResult {
        if let value = items[account] { return .found(value) }
        return .missing
    }

    @discardableResult
    func write(account: String, secret: String) -> Bool {
        if failWrites { return false }
        items[account] = secret
        return true
    }

    @discardableResult
    func delete(account: String) -> Bool {
        if failDeletes { return false }
        items.removeValue(forKey: account)
        // Idempotent, mirroring SystemKeychain.remove: an absent item is a
        // successful no-op (SecItemDelete returns errSecItemNotFound, which
        // the production code treats as success).
        return true
    }

    /// Test seam: plant a raw payload (valid or corrupt).
    func seed(account: String, payload: String) {
        items[account] = payload
    }
}

/// Issue #17 desktop-half: the bearer key-lifetime configuration (launch /
/// 24h / permanent) must behave predictably, and the restart-detection the
/// Settings panel relies on must never dead-loop.
///
/// Storage split per review: only the *mode* lives in UserDefaults; the
/// secret lives in the Keychain. Tests inject both a throwaway UserDefaults
/// suite and an in-memory Keychain, and restore the real stores in
/// tear-down so the developer's preferences/Keychain are never touched.
final class APIAuthConfigTests: XCTestCase {

    private var suiteName: String?
    private var suite: UserDefaults?
    private var keychain: InMemoryKeychain!

    override func setUpWithError() throws {
        try super.setUpWithError()
        let name = "APIAuthConfigTests-\(UUID().uuidString)"
        let fresh = try XCTUnwrap(UserDefaults(suiteName: name),
                                  "suite creation must succeed")
        fresh.removePersistentDomain(forName: name)
        suiteName = name
        suite = fresh
        APIAuthConfig.defaults = fresh

        keychain = InMemoryKeychain()
        APIAuthConfig.keychain = keychain
    }

    override func tearDownWithError() throws {
        if let suiteName {
            suite?.removePersistentDomain(forName: suiteName)
        }
        APIAuthConfig.defaults = .standard
        APIAuthConfig.keychain = SystemKeychain()
        suite = nil
        suiteName = nil
        keychain = nil
        try super.tearDownWithError()
    }

    // MARK: - mode storage

    func testModeDefaultsToLaunch() {
        XCTAssertEqual(APIAuthConfig.mode, .launch)
    }

    func testModeRoundTrips() {
        APIAuthConfig.mode = .hours24
        XCTAssertEqual(APIAuthConfig.mode, .hours24)
        APIAuthConfig.mode = .permanent
        XCTAssertEqual(APIAuthConfig.mode, .permanent)
    }

    // MARK: - storage split (review: no secret in defaults)

    func testNoSecretInUserDefaults() throws {
        // The secret must never land in the plaintext preference store —
        // only the mode metadata may.
        APIAuthConfig.mode = .hours24
        let bearer = try XCTUnwrap(APIAuthConfig.bearerForSpawn(now: Date()))
        let all = APIAuthConfig.defaults.dictionaryRepresentation()
        for (key, value) in all {
            let stringValue = String(describing: value)
            XCTAssertFalse(stringValue.contains(bearer),
                           "Secret must not leak into defaults (key \(key))")
            XCTAssertFalse(stringValue.contains(APIAuthConfig.keychainAccount),
                           "Keychain account must not leak into defaults")
            XCTAssertNotEqual(key, APIAuthConfig.keychainAccount)
        }
    }

    // MARK: - launch mode

    func testLaunchMintsFreshEachCall() throws {
        APIAuthConfig.mode = .launch
        let first = try XCTUnwrap(APIAuthConfig.bearerForSpawn())
        let second = try XCTUnwrap(APIAuthConfig.bearerForSpawn())
        XCTAssertEqual(first.count, 64)
        XCTAssertNotEqual(first, second, "launch mode must mint a fresh secret per spawn")
    }

    func testLaunchDoesNotPersist() throws {
        APIAuthConfig.mode = .launch
        _ = try XCTUnwrap(APIAuthConfig.bearerForSpawn())
        XCTAssertNil(APIAuthConfig.storedBearer(),
                     "launch mode must never write the secret to the Keychain")
        XCTAssertNil(APIAuthConfig.persistedKey)
    }

    // MARK: - 24h mode

    func testHours24ReusesFreshSecret() throws {
        APIAuthConfig.mode = .hours24
        let now = Date()
        let first = try XCTUnwrap(APIAuthConfig.bearerForSpawn(now: now))
        let later = now.addingTimeInterval(60 * 60) // 1h later, still fresh
        let second = try XCTUnwrap(APIAuthConfig.bearerForSpawn(now: later))
        XCTAssertEqual(first, second, "fresh persisted secret must be reused")
    }

    func testHours24RotatesAfterLifetime() throws {
        APIAuthConfig.mode = .hours24
        let now = Date()
        let first = try XCTUnwrap(APIAuthConfig.bearerForSpawn(now: now))
        let expired = now.addingTimeInterval(25 * 60 * 60) // past 24h
        let second = try XCTUnwrap(APIAuthConfig.bearerForSpawn(now: expired))
        XCTAssertNotEqual(first, second, "expired persisted secret must rotate")
        XCTAssertEqual(APIAuthConfig.persistedKey, second, "rotated secret must be persisted")
    }

    // MARK: - permanent mode

    func testPermanentReusesIndefinitely() throws {
        APIAuthConfig.mode = .permanent
        let now = Date()
        let first = try XCTUnwrap(APIAuthConfig.bearerForSpawn(now: now))
        let farFuture = now.addingTimeInterval(365 * 24 * 60 * 60)
        let second = try XCTUnwrap(APIAuthConfig.bearerForSpawn(now: farFuture))
        XCTAssertEqual(first, second, "permanent secret must never expire")
    }

    func testPermanentKeyExpiryNil() throws {
        APIAuthConfig.mode = .permanent
        _ = try XCTUnwrap(APIAuthConfig.bearerForSpawn())
        XCTAssertNil(APIAuthConfig.keyExpiry, "permanent mode has no expiry")
    }

    func testHours24KeyExpirySet() throws {
        APIAuthConfig.mode = .hours24
        let now = Date()
        _ = try XCTUnwrap(APIAuthConfig.bearerForSpawn(now: now))
        let expiry = try XCTUnwrap(APIAuthConfig.keyExpiry)
        XCTAssertEqual(expiry.timeIntervalSince(now), 24 * 60 * 60, accuracy: 1)
    }

    // MARK: - rotate

    func testRotatePersistedKey() throws {
        APIAuthConfig.mode = .permanent
        let first = try XCTUnwrap(APIAuthConfig.bearerForSpawn())
        XCTAssertTrue(APIAuthConfig.rotatePersistedKey(),
                      "rotate must report Keychain write failure instead of fabricating a key")
        let rotated = try XCTUnwrap(APIAuthConfig.persistedKey)
        XCTAssertNotEqual(first, rotated)
        XCTAssertEqual(APIAuthConfig.persistedKey, rotated)
    }

    func testRotateReportsKeychainFailure() throws {
        APIAuthConfig.mode = .permanent
        _ = APIAuthConfig.bearerForSpawn()
        keychain.failWrites = true
        XCTAssertFalse(APIAuthConfig.rotatePersistedKey(),
                       "refused Keychain write must surface as false so the UI does not claim a new key")
    }

    // MARK: - corrupt / missing fail-safe (review: never unauthenticated)

    func testCorruptKeychainFallsBackToFresh() throws {
        APIAuthConfig.mode = .hours24
        keychain.seed(account: APIAuthConfig.keychainAccount, payload: "not-a-valid-payload")
        let bearer = try XCTUnwrap(APIAuthConfig.bearerForSpawn(),
                                   "corrupt Keychain entry must resolve to a fresh random key, not nil")
        XCTAssertEqual(bearer.count, 64)
        XCTAssertEqual(APIAuthConfig.persistedKey, bearer,
                       "fail-safe must also repair the corrupt entry")
    }

    func testMissingKeychainFallsBackToFresh() throws {
        APIAuthConfig.mode = .permanent
        let bearer = try XCTUnwrap(APIAuthConfig.bearerForSpawn())
        XCTAssertEqual(bearer.count, 64)
    }

    // MARK: - clear on launch (sleeping-bomb prevention)

    func testClearPersistedKeyRemovesSecret() throws {
        APIAuthConfig.mode = .permanent
        _ = try XCTUnwrap(APIAuthConfig.bearerForSpawn())
        XCTAssertNotNil(APIAuthConfig.persistedKey, "precondition: key is stored")
        XCTAssertTrue(APIAuthConfig.clearPersistedKey(),
                      "clear must report success when the item was removed")
        XCTAssertNil(APIAuthConfig.persistedKey,
                     "switching to launch must purge the persisted secret")
    }

    func testClearPersistedKeyIdempotentWhenAbsent() {
        // SystemKeychain.remove treats an absent item as a successful no-op
        // (errSecItemNotFound); the mock mirrors that. Clearing with nothing
        // stored must NOT surface a spurious Keychain-denied warning.
        XCTAssertTrue(APIAuthConfig.clearPersistedKey(),
                      "clearing an absent key must succeed without a prompt")
        XCTAssertNil(APIAuthConfig.persistedKey)
    }

    func testClearPersistedKeyReportsFailure() throws {
        APIAuthConfig.mode = .permanent
        _ = try XCTUnwrap(APIAuthConfig.bearerForSpawn())
        keychain.failDeletes = true
        XCTAssertFalse(APIAuthConfig.clearPersistedKey(),
                       "ACL-refused delete must surface as false so the UI warns the old key may resurface")
    }
}
