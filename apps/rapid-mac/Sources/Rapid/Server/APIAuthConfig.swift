import Foundation

/// Configurable API authentication for the embedded engine.
///
/// Issue #17 desktop-half ships a fresh random bearer per spawn, which is
/// safe but annoying for local tooling: every restart rotates the key and
/// external callers (curl, scripts, the menubar) must re-fetch it.
///
/// This config exposes three key-lifetime modes, replacing the earlier
/// random / fixed / off tri-state after review (a plaintext preference key
/// and an unauthenticated mode both violate the security boundary):
///
///   * ``launch`` (default) — a fresh 64-hex secret per spawn
///     (``BearerSecret.generate()``). Most secure: a leaked key is
///     bounded to the current session.
///   * ``hours24`` — one secret reused for up to 24 hours, persisted in
///     the system Keychain so local tooling can hard-code it across
///     restarts for a day.
///   * ``permanent`` — one secret reused indefinitely until the user
///     explicitly rotates it. Persisted in the system Keychain.
///
/// Storage split (per review): only the *mode* is plaintext metadata in
/// ``UserDefaults``; the secret itself always lives in the Keychain. A
/// missing, empty, or corrupt Keychain value resolves to a fresh random
/// key — never to an unauthenticated engine.
enum APIAuthMode: String, CaseIterable, Identifiable {
    case launch
    case hours24
    case permanent

    var id: String { rawValue }

    var title: String {
        switch self {
        case .launch: return "Per launch"
        case .hours24: return "24 hours"
        case .permanent: return "Permanent"
        }
    }

    /// The lifetime of a persisted secret. ``nil`` means "no expiry"
    /// (launch mints per spawn anyway; permanent never expires).
    var lifetime: TimeInterval? {
        switch self {
        case .launch: return nil
        case .hours24: return 24 * 60 * 60
        case .permanent: return nil
        }
    }
}

enum APIAuthConfig {
    static let storageKey = "apiAuthMode"

    /// Keychain account for the persisted engine bearer. The payload is a
    /// single line: ``"<secret>\n<unix-epoch>"`` so the generation time
    /// travels with the secret and the item is self-contained.
    static let keychainAccount = "rapid.engine.bearer"

    /// Pluggable for tests: the default store is ``UserDefaults.standard``,
    /// but ``APIAuthConfigTests`` injects a throwaway suite instance so the
    /// modes and the fail-safe fallback can be exercised without touching
    /// the real preferences (or polluting the developer's prefs).
    ///
    /// ``UserDefaults`` is itself thread-safe; the mutable reference is only
    /// swapped by serialized tests and always restored to ``.standard`` in
    /// tear-down, so the `nonisolated(unsafe)` escape hatch (SE-0412) is the
    /// minimal way to keep the test seam under Swift 6 strict concurrency.
    nonisolated(unsafe) static var defaults: UserDefaults = .standard

    /// Pluggable for tests: the default store is the real system Keychain.
    /// Tests swap in an in-memory ``KeychainStoring`` double (the protocol
    /// exists precisely so the suite never touches Security.framework) and
    /// restore ``SystemKeychain()`` in tear-down. Same SE-0412 reasoning as
    /// ``defaults``.
    nonisolated(unsafe) static var keychain: any KeychainStoring = SystemKeychain()

    static var mode: APIAuthMode {
        get {
            let raw = defaults.string(forKey: storageKey)
            return APIAuthMode(rawValue: raw ?? APIAuthMode.launch.rawValue) ?? .launch
        }
        set {
            defaults.set(newValue.rawValue, forKey: storageKey)
        }
    }

    // MARK: - Keychain persistence

    /// The persisted bearer secret and its generation time, or ``nil`` when
    /// missing / unreadable / corrupt. Never falls back to a fresh value —
    /// callers decide what to do with a miss.
    static func storedBearer() -> (secret: String, generatedAt: Date)? {
        guard let stored = keychain.read(account: keychainAccount), !stored.isEmpty else {
            return nil
        }
        let parts = stored.split(separator: "\n", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2,
              !parts[0].isEmpty,
              let epoch = TimeInterval(parts[1])
        else {
            return nil
        }
        return (String(parts[0]), Date(timeIntervalSince1970: epoch))
    }

    /// Persist a bearer with its generation time to the Keychain.
    /// Replace/delete semantics come from ``KeychainStoring.write``
    /// (upsert). Returns false when the Keychain write is refused (user
    /// denied the access prompt) so callers can surface the degradation
    /// instead of silently falling back to a fresh random key on every
    /// spawn.
    @discardableResult
    static func persistBearer(_ secret: String, generatedAt: Date = Date()) -> Bool {
        let payload = "\(secret)\n\(Int(generatedAt.timeIntervalSince1970))"
        return keychain.write(account: keychainAccount, secret: payload)
    }

    /// Whether a persisted secret is still within its configured lifetime.
    static func isFresh(_ generatedAt: Date, now: Date = Date()) -> Bool {
        guard let lifetime = mode.lifetime else { return true }
        return now.timeIntervalSince(generatedAt) < lifetime
    }

    /// Remove the persisted secret from the Keychain. Called when the user
    /// switches back to ``launch`` so a stale key cannot resurface later
    /// ("sleeping bomb"): a key the user thought was gone must not start
    /// authenticating again if they re-select a persistent mode.
    @discardableResult
    static func clearPersistedKey() -> Bool {
        keychain.delete(account: keychainAccount)
    }

    // MARK: - Spawn resolution

    /// The bearer to hand to a freshly spawned engine, or ``nil`` only when
    /// the OS RNG fails (pathological; callers abort the start rather than
    /// spawn unauthenticated — never a "no auth" outcome).
    ///
    /// * ``launch`` — a fresh secret every call.
    /// * ``hours24`` / ``permanent`` — reuse the persisted Keychain secret
    ///   while fresh; otherwise mint, persist, and return a new one. A
    ///   missing / corrupt entry also mints fresh (fail-safe to random,
    ///   never to unauthenticated).
    ///
    /// - Parameter now: injectable clock for tests (expiry arithmetic).
    static func bearerForSpawn(now: Date = Date()) -> String? {
        switch mode {
        case .launch:
            return BearerSecret.generate()
        case .hours24, .permanent:
            if let stored = storedBearer(), isFresh(stored.generatedAt, now: now) {
                return stored.secret
            }
            guard let fresh = BearerSecret.generate() else { return nil }
            persistBearer(fresh, generatedAt: now)
            return fresh
        }
    }

    /// Force a new persisted secret now ("Rotate" / revoke). The old key
    /// stops working on the next spawn (the running engine keeps its current
    /// secret until it is restarted). Returns false if generation or the
    /// Keychain write fails; callers must not treat a false result as a
    /// usable new key (the engine would not be able to read it back).
    @discardableResult
    static func rotatePersistedKey(now: Date = Date()) -> Bool {
        guard let fresh = BearerSecret.generate() else { return false }
        return persistBearer(fresh, generatedAt: now)
    }

    /// When the currently-persisted key expires, for the settings/launch
    /// UI. ``nil`` means it never expires (launch has no persisted key;
    /// permanent is indefinite). Returns nil when nothing is stored.
    static var keyExpiry: Date? {
        guard let stored = storedBearer() else { return nil }
        guard let lifetime = mode.lifetime else { return nil }
        return stored.generatedAt.addingTimeInterval(lifetime)
    }

    /// The persisted key for UI display, or ``nil`` when nothing is stored.
    /// Launch mode never persists, so this is only meaningful for
    /// ``hours24`` / ``permanent``.
    static var persistedKey: String? {
        storedBearer()?.secret
    }

}
