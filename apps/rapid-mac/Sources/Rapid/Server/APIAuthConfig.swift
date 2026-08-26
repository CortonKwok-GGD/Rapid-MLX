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
///   * ``fixed`` — a user-supplied key kept in the system Keychain. Stable
///     across launches so local API clients can hard-code it.
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

    /// Keychain account for the user-supplied fixed key.
    static let fixedKeyAccount = "api.fixed-bearer"

    static var mode: APIAuthMode {
        get {
            let raw = UserDefaults.standard.string(forKey: storageKey)
            return APIAuthMode(rawValue: raw ?? APIAuthMode.random.rawValue) ?? .random
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: storageKey)
        }
    }

    /// The bearer to hand to a freshly spawned engine, or ``nil`` when auth
    /// is off. Callers should pass ``nil`` through to ``activeBearer`` and
    /// skip injecting ``RAPID_MLX_API_KEY`` (the engine then runs open).
    ///
    /// ``fixed`` falls back to random when the Keychain holds no value —
    /// never silently downgrades to ``off``.
    static func bearerForSpawn() -> String? {
        switch mode {
        case .random:
            return BearerSecret.generate()
        case .fixed:
            let keychain = SystemKeychain()
            if let fixed = keychain.read(account: fixedKeyAccount), !fixed.isEmpty {
                return fixed
            }
            return BearerSecret.generate()
        case .off:
            return nil
        }
    }
}
