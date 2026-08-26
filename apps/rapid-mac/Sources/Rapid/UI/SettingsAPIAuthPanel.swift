import SwiftUI

/// API authentication configuration for the embedded engine.
///
/// The engine binds loopback-only, so the bearer is NOT a network firewall —
/// it gates *local* callers (see Issue #17). Three modes:
///
///   * Random key (default) — fresh 64-hex secret per launch; safest, but
///     external local callers (curl, scripts) must re-fetch it each start.
///   * Fixed key — user-supplied, stored in the system Keychain; stable
///     across launches so tooling can hard-code it.
///   * Off — no auth; the engine serves every local request unauthenticated.
///     Most convenient, least safe: ANY local process (a sandbox-escaped
///     browser tab, an unrelated script) can drive inference.
struct SettingsAPIAuthPanel: View {
    @AppStorage(APIAuthConfig.storageKey) private var modeRaw = APIAuthMode.random.rawValue

    @State private var fixedKeyInput = ""
    @State private var savedNotice = false
    @State private var clearNotice = false

    private var mode: APIAuthMode {
        APIAuthMode(rawValue: modeRaw) ?? .random
    }

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            SectionHeader(
                "API Authentication",
                subtitle: "Controls the bearer key the embedded engine requires on its local port.",
                emphasis: .page
            )

            SettingsSection(
                "Auth mode",
                subtitle: "Applies on the next model start."
            ) {
                Picker("Mode", selection: $modeRaw) {
                    ForEach(APIAuthMode.allCases) { m in
                        Text(m.title).tag(m.rawValue)
                    }
                }
                .pickerStyle(.segmented)
                .accessibilityIdentifier("Settings.APIAuth.Mode")

                switch mode {
                case .random:
                    InlineNotice(
                        message: "A new random key is minted each launch and shown in the status pill. Safest — a leaked key expires with the session.",
                        tone: .info
                    )
                case .fixed:
                    fixedKeySection
                case .off:
                    InlineNotice(
                        message: "No auth: any local process can call the engine on 127.0.0.1 and drive inference. Use only on a trusted, single-user Mac.",
                        tone: .warning
                    )
                }
            }
        }
    }

    private var fixedKeySection: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            HStack(spacing: RapidTheme.Space.md) {
                SecureField("Fixed API key", text: $fixedKeyInput)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("Settings.APIAuth.FixedKey")
                Button("Save to Keychain") {
                    let trimmed = fixedKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { return }
                    if SystemKeychain().write(account: APIAuthConfig.fixedKeyAccount, secret: trimmed) {
                        savedNotice = true
                        clearNotice = false
                    }
                }
                .buttonStyle(.borderedProminent)
            }
            if savedNotice {
                InlineNotice(message: "Fixed key saved. It takes effect on the next model start.", tone: .success)
            }
            if clearNotice {
                InlineNotice(message: "Fixed key removed — falls back to a random key per launch.", tone: .info)
            }
            if let existing = SystemKeychain().read(account: APIAuthConfig.fixedKeyAccount), !existing.isEmpty {
                Button("Remove saved key") {
                    if SystemKeychain().delete(account: APIAuthConfig.fixedKeyAccount) {
                        clearNotice = true
                        savedNotice = false
                        fixedKeyInput = ""
                    }
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("Settings.APIAuth.RemoveFixedKey")
            }
            InlineNotice(
                message: "The key is stored in the macOS Keychain (not in app preferences). Keep it stable: this is the value external tools should send as the Bearer token.",
                tone: .info
            )
        }
    }
}
