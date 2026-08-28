import SwiftUI

/// API authentication configuration for the embedded engine.
///
/// A single "key lifetime" picker replaces the old random/fixed/off matrix:
///
/// * ``launch`` — a fresh 64-hex secret per engine spawn.
/// * ``hours24`` — one secret reused for 24 hours, persisted in the system
///   Keychain; tooling can hard-code it across restarts for a day.
/// * ``permanent`` — one secret reused indefinitely until rotated manually.
///
/// Switching to a persistent mode persists the engine's current key
/// immediately (or mints one if the engine isn't running), so the key on
/// screen today IS the key that survives. Mode changes never force a
/// restart; Rotate key is the one action that does — it is gated behind
/// a confirmation so the user can back out of an in-flight service.
struct SettingsAPIAuthPanel: View {
    @Environment(ServerManager.self) private var server
    @AppStorage(APIAuthConfig.storageKey) private var modeRaw = APIAuthMode.launch.rawValue
    @State private var keyRevision = 0
    @State private var confirmingRotate = false
    @State private var rotating = false
    @State private var rotateError: String?
    @State private var keychainDenied = false

    private var mode: APIAuthMode {
        APIAuthMode(rawValue: modeRaw) ?? .launch
    }

    /// Picker binding that normalizes stale values left by earlier builds
    /// (the storage key predates the launch/hours24/permanent enum, so a
    /// user's old ``fixed``/``off`` choice would otherwise render as an
    /// empty selection). Reading a legacy value shows ``launch``; picking
    /// any option writes back a canonical value.
    private var normalizedModeBinding: Binding<String> {
        Binding(
            get: { APIAuthMode(rawValue: modeRaw)?.rawValue ?? APIAuthMode.launch.rawValue },
            set: { modeRaw = $0 }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            SectionHeader(
                "API Authentication",
                subtitle: "Controls the bearer key the embedded engine requires on its local port. Changes apply on the next model start.",
                emphasis: .page
            )

            SettingsSection("Key lifetime") {
                VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                    Picker("Lifetime", selection: normalizedModeBinding) {
                        ForEach(APIAuthMode.allCases) { m in
                            Text(m.title).tag(m.rawValue)
                        }
                    }
                    .pickerStyle(.segmented)
                    .accessibilityIdentifier("Settings.APIAuth.Mode")
                    .onChange(of: modeRaw) { _, newRaw in
                        guard let newMode = APIAuthMode(rawValue: newRaw) else {
                            modeRaw = APIAuthMode.launch.rawValue
                            return
                        }
                        guard newMode != .launch else {
                            // Switching back to session-only: purge the
                            // persisted key so it cannot resurface later.
                            keychainDenied = !APIAuthConfig.clearPersistedKey()
                            return
                        }
                        keychainDenied = !persistCurrentKey()
                        keyRevision += 1
                    }

                    switch mode {
                    case .launch:
                        InlineNotice(
                            message: "A new random key is minted on each start. Safest — a leaked key expires with the session.",
                            tone: .info
                        )
                        if keychainDenied {
                            InlineNotice(
                                message: "Keychain access was denied, so the previous persistent key was not removed. It could be reused if you select a persistent mode again — allow Keychain access to purge it.",
                                tone: .warning
                            )
                        }
                    case .hours24:
                        persistentKeySection(
                            notice: "One key, reused for 24 hours, stored in the system Keychain.",
                            expiryHint: APIAuthConfig.keyExpiry
                        )
                    case .permanent:
                        persistentKeySection(
                            notice: "One key, reused until you rotate it, stored in the system Keychain.",
                            expiryHint: nil
                        )
                    }
                }
            }
        }
        .padding(RapidTheme.Space.xl)
    }

    /// Key display + rotate for the persistent modes. `expiryHint` is the
    /// date the current key stops being fresh (nil = never).
    private func persistentKeySection(notice: String, expiryHint: Date?) -> some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            InlineNotice(message: notice, tone: .info)
            InlineNotice(
                message: "macOS asks for Keychain permission on first use — choose “Always Allow” so the key persists.",
                tone: .info
            )
            if keychainDenied {
                InlineNotice(
                    message: "Keychain access was denied, so the key was not saved. Every launch will use a fresh random key — switch away from a persistent mode or allow Keychain access to fix.",
                    tone: .warning
                )
            }

            if let key = displayedKey {
                HStack(spacing: RapidTheme.Space.md) {
                    Text(masked(key))
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .accessibilityIdentifier("Settings.APIAuth.KeyDisplay")
                    Spacer()
                    Button("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(key, forType: .string)
                    }
                    .buttonStyle(.bordered)
                    .accessibilityIdentifier("Settings.APIAuth.CopyKey")
                }
            } else {
                Text("No key stored yet — one is generated on the next model start.")
                    .font(RapidFont.secondary)
                    .foregroundStyle(.secondary)
            }

            if let expiry = expiryHint {
                InlineNotice(
                    message: "Expires \(expiry.formatted(date: .abbreviated, time: .shortened)). A fresh key is minted on the next start.",
                    tone: .warning
                )
            }

            Button(rotating ? "Rotating…" : "Rotate key") {
                confirmingRotate = true
            }
            .buttonStyle(.bordered)
            .disabled(rotating)
            .accessibilityIdentifier("Settings.APIAuth.RotateKey")
        }
        .confirmationDialog(
            "Rotate API key?",
            isPresented: $confirmingRotate,
            titleVisibility: .visible
        ) {
            Button("Rotate & Restart", role: .destructive) {
                rotateKeyAndRestart()
            }
            .accessibilityIdentifier("Settings.APIAuth.ConfirmRotate")
            Button("Cancel", role: .cancel) {}
            .accessibilityIdentifier("Settings.APIAuth.CancelRotate")
        } message: {
            Text("A new key is generated and the engine restarts immediately. In-flight requests are interrupted.")
        }
        .alert("Rotation failed", isPresented: Binding(
            get: { rotateError != nil },
            set: { if !$0 { rotateError = nil } }
        )) {
            Button("OK", role: .cancel) {}
            .accessibilityIdentifier("Settings.APIAuth.RotationFailedOK")
        } message: {
            Text(rotateError ?? "")
        }
    }

    /// Rotate the persisted key, then restart the engine so the new key is
    /// live immediately (the engine pins the key it is spawned with). If no
    /// model is serving, the new key simply takes effect on the next start.
    private func rotateKeyAndRestart() {
        guard APIAuthConfig.rotatePersistedKey() else {
            rotateError = "Couldn't store the new key — Keychain access may be denied. No restart was performed; the previous key is still in effect."
            return
        }
        keychainDenied = false
        keyRevision += 1
        guard let alias = server.servingAlias else { return }
        rotating = true
        Task {
            let ok = await server.restart(alias: alias)
            rotating = false
            if !ok {
                rotateError = "Engine restart failed — the new key is stored and applies on the next model start."
            }
        }
    }

    /// Persist the engine's current key when the user switches to a
    /// persistent mode, so the key they see today IS the key that survives
    /// (24 hours / until rotated). If the engine isn't running, reuse an
    /// existing fresh persisted key — switching modes must not churn the
    /// secret (hard-coded tooling keeps working); mint only when nothing
    /// fresh is stored. Returns false when the Keychain write was refused.
    private func persistCurrentKey() -> Bool {
        if let current = server.activeBearer {
            return APIAuthConfig.persistBearer(current)
        }
        if let stored = APIAuthConfig.storedBearer(),
           APIAuthConfig.isFresh(stored.generatedAt) {
            return true
        }
        return APIAuthConfig.rotatePersistedKey()
    }

    /// Re-read the Keychain so Rotate is reflected immediately (`keyRevision`
    /// forces SwiftUI to recompute the body).
    private var displayedKey: String? {
        _ = keyRevision
        return APIAuthConfig.persistedKey
    }

    private func masked(_ key: String) -> String {
        String(repeating: "•", count: min(key.count, 16))
    }
}
