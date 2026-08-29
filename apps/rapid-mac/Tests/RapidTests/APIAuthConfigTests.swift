import Foundation
import Testing

@testable import Rapid

/// In-memory ``KeychainStoring`` double so the suite never touches
/// Security.framework (which would prompt, leak across tests, and require
/// manual cleanup). Upsert semantics mirror ``SystemKeychain.write``:
/// a write replaces any existing item for the same account.
///
/// ``@unchecked Sendable``: the double is exercised only from this suite's
/// main-actor test methods, one at a time, so its mutable backing store
/// never crosses an isolation boundary concurrently. The compiler cannot
/// prove that (``KeychainStoring: Sendable`` requires a value type or an
/// immutable class), hence the explicit opt-out — the standard shape for
/// an isolated in-memory test double.
private final class InMemoryKeychain: KeychainStoring, @unchecked Sendable {
    private var items: [String: String] = [:]
    private(set) var readCount = 0
    private(set) var lastReadWasOnMainThread: Bool?

    /// Test seam: simulate a refused Keychain write (user denied the prompt).
    var failWrites = false

    /// Test seam: simulate an ACL-refused Keychain delete.
    var failDeletes = false

    func read(account: String) -> String? {
        readCount += 1
        lastReadWasOnMainThread = Thread.isMainThread
        return items[account]
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
/// teardown so the developer's preferences/Keychain are never touched.
/// `@MainActor` keeps each mutation actor-safe; `.serialized` additionally
/// prevents a test that awaits the persistence actor from yielding to another
/// case that swaps the same global stores.
@MainActor
@Suite("APIAuthConfig — key lifetime (launch / 24h / permanent)", .serialized)
final class APIAuthConfigTests {
    /// TestDefaultsScope pattern: name suites per-test, clean up on deinit so
    /// ``~/Library/Preferences`` never accumulates straggler plists.
    nonisolated(unsafe) private var createdSuiteNames: [String] = []
    deinit { TestDefaultsScope.cleanup(suiteNames: createdSuiteNames) }

    private var suite: UserDefaults!
    private var keychain: InMemoryKeychain!

    private func freshSuite() -> UserDefaults {
        let name = TestDefaultsScope.mintSuiteName(prefix: "rapid-apiauth-test-")
        createdSuiteNames.append(name)
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    /// Every test starts with an isolated store + keychain and restores the
    /// real ones afterwards so the developer's preferences are untouched.
    private func setUpStores() {
        suite = freshSuite()
        APIAuthConfig.defaults = suite
        keychain = InMemoryKeychain()
        APIAuthConfig.keychain = keychain
    }

    private func tearDownStores() {
        APIAuthConfig.defaults = .standard
        APIAuthConfig.keychain = SystemKeychain()
        suite = nil
        keychain = nil
    }

    // MARK: - mode storage

    @Test("mode defaults to launch")
    func testModeDefaultsToLaunch() {
        setUpStores()
        defer { tearDownStores() }
        #expect(APIAuthConfig.mode == .launch)
    }

    @Test("mode round-trips through UserDefaults")
    func testModeRoundTrips() {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        #expect(APIAuthConfig.mode == .hours24)
        APIAuthConfig.mode = .permanent
        #expect(APIAuthConfig.mode == .permanent)
    }

    @Test("auth settings are locked during startup and lifecycle operations")
    func testAuthMutationGateCoversLifecycleRaces() {
        #expect(!SettingsAPIAuthPanel.canMutateAuth(
            for: .starting(alias: "model"), isOperating: false, isRestarting: false
        ))
        #expect(!SettingsAPIAuthPanel.canMutateAuth(
            for: .ready(alias: "model"), isOperating: true, isRestarting: false
        ))
        #expect(!SettingsAPIAuthPanel.canMutateAuth(
            for: .stopped, isOperating: false, isRestarting: true
        ))
        #expect(SettingsAPIAuthPanel.canMutateAuth(
            for: .ready(alias: "model"), isOperating: false, isRestarting: false
        ))
        #expect(SettingsAPIAuthPanel.canMutateAuth(
            for: .idle, isOperating: false, isRestarting: false
        ))
        #expect(SettingsAPIAuthPanel.canMutateAuth(
            for: .stopped, isOperating: false, isRestarting: false
        ))
    }

    @Test("restart refuses to race an existing lifecycle operation")
    func testRestartRejectsConcurrentLifecycleOperation() async {
        let manager = ServerManager(
            testingState: .ready(alias: "model"),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            activeBearer: "existing-bearer",
            isOperating: true
        )

        let restarted = await manager.restart(alias: "model")

        #expect(!restarted)
        #expect(manager.state == .ready(alias: "model"))
        #expect(manager.activeBearer == "existing-bearer")
    }

    @Test("restart preserves the current model path only for the same alias")
    func testRestartHFPathResolution() {
        #expect(ServerManager.resolveRestartHFPath(
            requestedAlias: "custom-model",
            explicitHFPath: nil,
            launchedAlias: "CUSTOM-MODEL",
            launchedHFPath: "/models/custom-model"
        ) == "/models/custom-model")
        #expect(ServerManager.resolveRestartHFPath(
            requestedAlias: "different-model",
            explicitHFPath: nil,
            launchedAlias: "custom-model",
            launchedHFPath: "/models/custom-model"
        ) == nil)
        #expect(ServerManager.resolveRestartHFPath(
            requestedAlias: "custom-model",
            explicitHFPath: "owner/explicit-repo",
            launchedAlias: "custom-model",
            launchedHFPath: "/models/custom-model"
        ) == "owner/explicit-repo")
    }

    @Test("restart reservation blocks an interleaved model start")
    func testRestartReservationOwnsStopStartGap() async {
        let manager = ServerManager(
            testingState: .ready(alias: "custom-model"),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            launchedHFPath: "/models/custom-model"
        )
        var competingStartSucceeded: Bool?
        var reservedLaunch: (alias: String, hfPath: String?)?
        manager.restartAfterStopHook = {
            competingStartSucceeded = await manager.ensureServing(alias: "other-model")
        }
        manager.restartLaunchOverride = { alias, hfPath in
            reservedLaunch = (alias, hfPath)
            return true
        }

        let restarted = await manager.restart(alias: "custom-model")

        #expect(restarted)
        #expect(competingStartSucceeded == false)
        #expect(reservedLaunch?.alias == "custom-model")
        #expect(reservedLaunch?.hfPath == "/models/custom-model")
    }

    // MARK: - storage split (review: no secret in defaults)

    @Test("secret never lands in UserDefaults")
    func testNoSecretInUserDefaults() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        let bearer = try #require(APIAuthConfig.bearerForSpawn(now: Date()))
        let all = APIAuthConfig.defaults.dictionaryRepresentation()
        for (key, value) in all {
            let stringValue = String(describing: value)
            #expect(!stringValue.contains(bearer),
                    "Secret must not leak into defaults (key \(key))")
            #expect(!stringValue.contains(APIAuthConfig.keychainAccount),
                    "Keychain account must not leak into defaults")
            #expect(key != APIAuthConfig.keychainAccount)
        }
    }

    // MARK: - launch mode

    @Test("launch mints a fresh secret per spawn")
    func testLaunchMintsFreshEachCall() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .launch
        let first = try #require(APIAuthConfig.bearerForSpawn())
        let second = try #require(APIAuthConfig.bearerForSpawn())
        #expect(first.count == 64)
        #expect(first != second, "launch mode must mint a fresh secret per spawn")
    }

    @Test("launch never persists to Keychain")
    func testLaunchDoesNotPersist() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .launch
        _ = try #require(APIAuthConfig.bearerForSpawn())
        #expect(APIAuthConfig.storedBearer() == nil,
                "launch mode must never write the secret to the Keychain")
        #expect(APIAuthConfig.persistedKey == nil)
    }

    // MARK: - 24h mode

    @Test("24h reuses a fresh persisted secret")
    func testHours24ReusesFreshSecret() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        let now = Date()
        let first = try #require(APIAuthConfig.bearerForSpawn(now: now))
        let later = now.addingTimeInterval(60 * 60) // 1h later, still fresh
        let second = try #require(APIAuthConfig.bearerForSpawn(now: later))
        #expect(first == second, "fresh persisted secret must be reused")
    }

    @Test("daily mode rotates on the first spawn after 24h")
    func testHours24RotatesAfterLifetime() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        let now = Date()
        let first = try #require(APIAuthConfig.bearerForSpawn(now: now))
        let expired = now.addingTimeInterval(25 * 60 * 60) // past 24h
        let second = try #require(APIAuthConfig.bearerForSpawn(now: expired))
        #expect(first != second, "expired persisted secret must rotate")
        #expect(APIAuthConfig.persistedKey == second, "rotated secret must be persisted")
    }

    // MARK: - permanent mode

    @Test("permanent reuses the secret indefinitely")
    func testPermanentReusesIndefinitely() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .permanent
        let now = Date()
        let first = try #require(APIAuthConfig.bearerForSpawn(now: now))
        let farFuture = now.addingTimeInterval(365 * 24 * 60 * 60)
        let second = try #require(APIAuthConfig.bearerForSpawn(now: farFuture))
        #expect(first == second, "permanent secret must never expire")
    }

    @Test("permanent has no scheduled rotation")
    func testPermanentKeyExpiryNil() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .permanent
        _ = try #require(APIAuthConfig.bearerForSpawn())
        #expect(
            APIAuthConfig.persistenceSnapshot(for: .permanent).rotationDate == nil,
            "permanent mode has no scheduled rotation"
        )
    }

    @Test("daily mode reports the earliest next-start rotation moment")
    func testHours24KeyRotationDateSet() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        let now = Date()
        _ = try #require(APIAuthConfig.bearerForSpawn(now: now))
        let rotation = try #require(
            APIAuthConfig.persistenceSnapshot(for: .hours24).rotationDate
        )
        #expect(abs(rotation.timeIntervalSince(now) - 24 * 60 * 60) < 1)
    }

    @Test("persistence snapshot reads Keychain once")
    func testPersistenceSnapshotUsesSingleKeychainRead() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        let now = Date()
        let bearer = String(repeating: "a", count: 64)
        keychain.seed(
            account: APIAuthConfig.keychainAccount,
            payload: "\(bearer)\n\(Int(now.timeIntervalSince1970))"
        )
        let readsBefore = keychain.readCount

        let snapshot = APIAuthConfig.persistenceSnapshot(for: .hours24)

        #expect(keychain.readCount == readsBefore + 1)
        #expect(snapshot.persistedKey == bearer)
        #expect(snapshot.rotationDate != nil)
    }

    // MARK: - rotate

    @Test("rotate mints and persists a new secret")
    func testRotatePersistedKey() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .permanent
        let first = try #require(APIAuthConfig.bearerForSpawn())
        #expect(APIAuthConfig.rotatePersistedKey(),
                "rotate must report Keychain write failure instead of fabricating a key")
        let rotated = try #require(APIAuthConfig.persistedKey)
        #expect(first != rotated)
        #expect(APIAuthConfig.persistedKey == rotated)
    }

    @Test("rotate surfaces a refused Keychain write")
    func testRotateReportsKeychainFailure() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .permanent
        _ = try #require(APIAuthConfig.bearerForSpawn())
        keychain.failWrites = true
        #expect(!APIAuthConfig.rotatePersistedKey(),
                "refused Keychain write must surface as false so the UI does not claim a new key")
    }

    // MARK: - corrupt / missing fail-safe (review: never unauthenticated)

    @Test("corrupt Keychain entry falls back to fresh and repairs")
    func testCorruptKeychainFallsBackToFresh() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        keychain.seed(account: APIAuthConfig.keychainAccount, payload: "not-a-valid-payload")
        let bearer = try #require(APIAuthConfig.bearerForSpawn(),
                                  "corrupt Keychain entry must resolve to a fresh random key, not nil")
        #expect(bearer.count == 64)
        #expect(APIAuthConfig.persistedKey == bearer,
                "fail-safe must also repair the corrupt entry")
    }

    @Test("weak but parseable Keychain secret is rejected and repaired")
    func testWeakKeychainSecretFallsBackToFresh() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .permanent
        keychain.seed(
            account: APIAuthConfig.keychainAccount,
            payload: "weak\n\(Int(Date().timeIntervalSince1970))"
        )
        let bearer = try #require(APIAuthConfig.bearerForSpawn())
        #expect(bearer.count == 64)
        #expect(bearer != "weak")
        #expect(APIAuthConfig.persistedKey == bearer)
    }

    @Test("non-hex Keychain secret is rejected and repaired")
    func testNonHexKeychainSecretFallsBackToFresh() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .permanent
        let nonHex = String(repeating: "z", count: 64)
        keychain.seed(
            account: APIAuthConfig.keychainAccount,
            payload: "\(nonHex)\n\(Int(Date().timeIntervalSince1970))"
        )
        let bearer = try #require(APIAuthConfig.bearerForSpawn())
        #expect(bearer.count == 64)
        #expect(bearer != nonHex)
        #expect(APIAuthConfig.persistedKey == bearer)
    }

    @Test("persistent spawn stays authenticated when Keychain write fails")
    func testPersistentSpawnReportsSafePersistenceDegradation() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        keychain.failWrites = true

        let resolved = try #require(APIAuthConfig.resolvedBearerForSpawn())

        #expect(resolved.secret.count == 64)
        #expect(resolved.persistenceDegraded)
        #expect(APIAuthConfig.persistedKey == nil)
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: resolved.secret,
            ambient: ["PATH": "/usr/bin:/bin"]
        )
        #expect(env["RAPID_MLX_API_KEY"] == resolved.secret)
    }

    @Test("expired stale key is never displayed after replacement write fails")
    func testExpiredStaleKeyDoesNotReplaceLiveDegradedBearer() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        let old = String(repeating: "a", count: 64)
        let expiredAt = Date().addingTimeInterval(-25 * 60 * 60)
        keychain.seed(
            account: APIAuthConfig.keychainAccount,
            payload: "\(old)\n\(Int(expiredAt.timeIntervalSince1970))"
        )
        keychain.failWrites = true

        let resolved = try #require(APIAuthConfig.resolvedBearerForSpawn())
        let displayed = SettingsAPIAuthPanel.displayedKey(
            activeBearer: resolved.secret,
            persistedKey: APIAuthConfig.persistedKey
        )

        #expect(resolved.persistenceDegraded)
        #expect(resolved.persistence == .empty)
        #expect(resolved.secret != old)
        #expect(APIAuthConfig.persistedKey == old, "refused upsert leaves the stale item in Keychain")
        #expect(displayed == resolved.secret, "Settings must copy the bearer accepted by the live child")
    }

    @Test("running child bearer stays displayed until rotation restart succeeds")
    func testActiveBearerWinsDuringRotationRestartRace() {
        let active = String(repeating: "a", count: 64)
        let newlyPersisted = String(repeating: "b", count: 64)

        #expect(SettingsAPIAuthPanel.displayedKey(
            activeBearer: active,
            persistedKey: newlyPersisted
        ) == active)
        #expect(SettingsAPIAuthPanel.displayedKey(
            activeBearer: nil,
            persistedKey: newlyPersisted
        ) == newlyPersisted)
    }

    @Test("server publishes persisted auth from an asynchronous snapshot")
    func testServerRefreshesCachedAuthPersistence() async throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        let bearer = String(repeating: "b", count: 64)
        let now = Date()
        keychain.seed(
            account: APIAuthConfig.keychainAccount,
            payload: "\(bearer)\n\(Int(now.timeIntervalSince1970))"
        )
        let manager = ServerManager(
            testingState: .stopped,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )

        await manager.refreshAuthPersistence()

        #expect(manager.authPersistence.persistedKey == bearer)
        #expect(manager.authPersistence.rotationDate != nil)
        #expect(keychain.lastReadWasOnMainThread == false)
    }

    @Test("missing Keychain entry falls back to fresh")
    func testMissingKeychainFallsBackToFresh() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .permanent
        let bearer = try #require(APIAuthConfig.bearerForSpawn())
        #expect(bearer.count == 64)
    }

    // MARK: - clear on launch (sleeping-bomb prevention)

    @Test("clear removes the persisted secret")
    func testClearPersistedKeyRemovesSecret() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .permanent
        _ = try #require(APIAuthConfig.bearerForSpawn())
        #expect(APIAuthConfig.persistedKey != nil, "precondition: key is stored")
        #expect(APIAuthConfig.clearPersistedKey(),
                "clear must report success when the item was removed")
        #expect(APIAuthConfig.persistedKey == nil,
                "switching to launch must purge the persisted secret")
    }

    @Test("clear of an absent key succeeds silently")
    func testClearPersistedKeyIdempotentWhenAbsent() {
        setUpStores()
        defer { tearDownStores() }
        // SystemKeychain.remove treats an absent item as a successful no-op
        // (errSecItemNotFound); the mock mirrors that. Clearing with nothing
        // stored must NOT surface a spurious Keychain-denied warning.
        #expect(APIAuthConfig.clearPersistedKey(),
                "clearing an absent key must succeed without a prompt")
        #expect(APIAuthConfig.persistedKey == nil)
    }

    @Test("clear surfaces an ACL-refused delete")
    func testClearPersistedKeyReportsFailure() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .permanent
        _ = try #require(APIAuthConfig.bearerForSpawn())
        keychain.failDeletes = true
        #expect(!APIAuthConfig.clearPersistedKey(),
                "ACL-refused delete must surface as false so the UI warns the old key may resurface")
    }

    // MARK: - spawn/client path contracts (owner review)

    @Test("spawn env carries the current effective bearer (fresh 24h key)")
    func testSpawnEnvCarriesEffectiveBearer() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        let bearer = try #require(APIAuthConfig.bearerForSpawn())
        // Real spawn-path contract: ServerManager.serveEnvironmentAdditions
        // is exactly what the spawn uses to build the sidecar env (Layer 2).
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: bearer,
            ambient: ["PATH": "/usr/bin:/bin"]
        )
        #expect(env["RAPID_MLX_API_KEY"] == bearer,
                "spawn env must carry the bearer the spawn actually resolved")
        #expect(env["RAPID_MLX_API_KEY"] != nil,
                "a live spawn must never be unauthenticated")
    }

    @Test("a model start after 24h rotates the spawn bearer")
    func testSpawnEnvRotatesAfterExpiry() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        let now = Date()
        let first = try #require(APIAuthConfig.bearerForSpawn(now: now))
        // A later spawn past the 24h lifetime must resolve a NEW key and
        // carry it into the spawn env — the engine never starts with an
        // expired secret.
        let expired = now.addingTimeInterval(25 * 60 * 60)
        let rotated = try #require(APIAuthConfig.bearerForSpawn(now: expired))
        #expect(rotated != first, "expired 24h key must rotate at spawn")
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: rotated,
            ambient: ["PATH": "/usr/bin:/bin"]
        )
        #expect(env["RAPID_MLX_API_KEY"] == rotated)
        #expect(env["RAPID_MLX_API_KEY"] != first,
                "spawn env must never carry the expired key")
    }

    @Test("client path uses the same bearer the spawn resolved")
    func testClientPathMatchesSpawnBearer() throws {
        setUpStores()
        defer { tearDownStores() }
        APIAuthConfig.mode = .hours24
        let bearer = try #require(APIAuthConfig.bearerForSpawn())
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: bearer,
            ambient: ["PATH": "/usr/bin:/bin"]
        )
        let spawnValue = env["RAPID_MLX_API_KEY"]
        #expect(spawnValue == bearer)
        // ChatStreamClient is driven with the manager's activeBearer; the
        // contract is that this is the SAME value the spawn injected. The
        // full header-on-wire assertion lives in ChatStreamBearerAuthTests;
        // this pins the value identity between spawn and client.
        let header = "Bearer \(spawnValue ?? "")"
        #expect(header == "Bearer \(bearer)")
    }
}
