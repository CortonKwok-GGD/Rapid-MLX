import Foundation

/// Configurable API authentication for the embedded engine.
///
/// Issue #17 desktop-half ships a fresh random bearer per spawn, which is
/// safe but annoying for local tooling: every restart rotates the key and
/// external callers (curl, scripts, the menubar) must re-fetch it.
///
/// This config exposes three modes, stored in ``UserDefaults``:
///
///   * ``random`` (default) — the existing per-launch
///     ``BearerSecret.generate()`` behaviour. Most secure: a leaked key is
///     bounded to the current session.
///   * ``fixed`` — a user-supplied key kept in app preferences (the same
///     store every other user setting uses; deliberately NOT the system
///     Keychain, which the official convention reserves for third-party
///     service secrets like Brave/Tavily). Stable across launches so
///     local API clients can hard-code it.
///   * ``off`` — no auth. The engine already allows this natively
///     (``cfg.api_key is None`` short-circuits in
///     ``vllm_mlx/middleware/auth.py``); the desktop simply stops injecting
///     ``RAPID_MLX_API_KEY``. Most convenient, least secure: ANY local
///     process (sandbox-escaped browser tab, unrelated script) can drive
///     inference on the loopback port.
enum APIAuthMode: String, CaseIterable, Identifiable {
    case random
    case fixed
    case off

    var id: String { rawValue }

    var title: String {
        switch self {
        case .random: return "Random key (per launch)"
        case .fixed: return "Fixed key"
        case .off: return "Off (no auth)"
        }
    }
}

enum APIAuthConfig {
    static let storageKey = "apiAuthMode"

    /// UserDefaults key for the user-supplied fixed key.
    /// Deliberately NOT the system Keychain: official convention stores
    /// third-party service secrets (Brave/Tavily) in Keychain, but the
    /// engine bearer protects only a loopback server whose threat model
    /// matches BearerSecret's env delivery (same-UID readers win either
    /// way). A fixed key must survive restarts, so it lives in the same
    /// preferences store every other user setting uses.
    static let fixedKeyStorageKey = "apiFixedBearer"

    /// Pluggable for tests: the default store is ``UserDefaults.standard``,
    /// but ``APIAuthConfigTests`` injects a throwaway suite instance so the
    /// three modes and the fixed-key fallback can be exercised without
    /// touching the real preferences (or polluting the developer's prefs).
    ///
    /// ``UserDefaults`` is itself thread-safe; the mutable reference is only
    /// swapped by serialized tests and always restored to ``.standard`` in
    /// tear-down, so the `nonisolated(unsafe)` escape hatch (SE-0412) is the
    /// minimal way to keep the test seam under Swift 6 strict concurrency.
    nonisolated(unsafe) static var defaults: UserDefaults = .standard

    static var mode: APIAuthMode {
        get {
            let raw = defaults.string(forKey: storageKey)
            return APIAuthMode(rawValue: raw ?? APIAuthMode.random.rawValue) ?? .random
        }
        set {
            defaults.set(newValue.rawValue, forKey: storageKey)
        }
    }

    /// The mode that will actually take effect for the next spawn, once
    /// the fixed-key fallback is accounted for.
    ///
    /// ``fixed`` with an empty stored key degrades to ``random`` — never to
    /// ``off`` — so the effective mode is what the UI should display and
    /// what restart detection should compare against.
    static var effectiveMode: APIAuthMode {
        let selected = mode
        switch selected {
        case .fixed:
            return hasStoredFixedKey ? .fixed : .random
        case .random, .off:
            return selected
        }
    }

    /// True when a non-empty fixed key is currently stored.
    static var hasStoredFixedKey: Bool {
        guard let fixed = defaults.string(forKey: fixedKeyStorageKey) else { return false }
        return !fixed.isEmpty
    }

    /// The bearer to hand to a freshly spawned engine, or ``nil`` when auth
    /// is off. Callers should pass ``nil`` through to ``activeBearer`` and
    /// skip injecting ``RAPID_MLX_API_KEY`` (the engine then runs open).
    ///
    /// ``fixed`` falls back to random when no fixed key is stored — never
    /// silently downgrades to ``off``.
    static func bearerForSpawn() -> String? {
        switch mode {
        case .random:
            return BearerSecret.generate()
        case .fixed:
            if hasStoredFixedKey {
                return defaults.string(forKey: fixedKeyStorageKey)
            }
            return BearerSecret.generate()
        case .off:
            return nil
        }
    }

    /// Pure restart-detection: whether the currently-running engine needs a
    /// restart for the configured settings to take effect.
    ///
    /// Compares in *effective* terms so the fixed-key fallback is handled:
    /// ``fixed`` without a stored key is already running ``random``, so it
    /// never demands a restart (the pre-review code dead-looped — the panel
    /// showed "Applied" yet flagged a pending restart forever, P1 #1).
    ///
    /// The fixed-key comparison is against the *stored* key, not the mode:
    /// editing a fixed key while the engine runs fixed must demand a
    /// restart even though the mode never changed.
    ///
    /// - Parameters:
    ///   - activeMode: the effective mode recorded at spawn time (nil when
    ///     no engine is running / no spawn has ever completed).
    ///   - activeBearer: the bearer the running engine was spawned with.
    ///   - isServing: whether an engine is currently serving.
    static func needsRestart(activeMode: APIAuthMode?, activeBearer: String?, isServing: Bool) -> Bool {
        guard isServing, let activeMode else { return false }
        if effectiveMode != activeMode { return true }
        switch effectiveMode {
        case .fixed:
            return activeBearer != defaults.string(forKey: fixedKeyStorageKey)
        case .random, .off:
            return false
        }
    }
}
