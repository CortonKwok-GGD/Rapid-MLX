import SwiftUI

/// API authentication configuration for the embedded engine.
///
/// The engine binds loopback-only, so the bearer is NOT a network firewall —
/// it gates *local* callers (see Issue #17). Three modes:
///
///   * Random key (default) — fresh 64-hex secret per launch; safest, but
///     external local callers (curl, scripts) must re-fetch it each start.
///   * Fixed key — user-supplied, stored in app preferences; stable
///     across launches so tooling can hard-code it.
///   * Off — no auth; the engine serves every local request unauthenticated.
///     Most convenient, least safe: ANY local process (a sandbox-escaped
///     browser tab, an unrelated script) can drive inference.
struct SettingsAPIAuthPanel: View {
    @AppStorage(APIAuthConfig.storageKey) private var modeRaw = APIAuthMode.random.rawValue

    @Environment(ServerManager.self) private var server
    @State private var fixedKeyInput = ""
    @State private var savedNotice = false
    @State private var restarting = false
    @State private var restartFeedback: String?
    @State private var revealInput = false
    @FocusState private var keyFieldFocused: Bool
    @State private var noticeDismissTask: Task<Void, Never>?

    private var mode: APIAuthMode {
        APIAuthMode(rawValue: modeRaw) ?? .random
    }

    private var hasFixedKey: Bool {
        !(APIAuthConfig.defaults.string(forKey: APIAuthConfig.fixedKeyStorageKey) ?? "").isEmpty
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
                .onChange(of: modeRaw) { _, _ in
                    restartFeedback = nil
                    savedNotice = false
                }

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

                if isServing {
                    Divider()
                    HStack(spacing: RapidTheme.Space.md) {
                        VStack(alignment: .leading, spacing: 2) {
                            if authNeedsRestart {
                                Text("A model is currently running")
                                    .font(RapidFont.secondary)
                                    .foregroundStyle(.secondary)
                                Text("Auth changes apply on the next engine start.")
                                    .font(RapidFont.caption)
                                    .foregroundStyle(.tertiary)
                            } else {
                                Text("Auth is already in effect")
                                    .font(RapidFont.secondary)
                                    .foregroundStyle(.secondary)
                                Text("No restart needed for the current settings.")
                                    .font(RapidFont.caption)
                                    .foregroundStyle(.tertiary)
                            }
                        }
                        Spacer()
                        Button(restarting ? "Restarting…" : (authNeedsRestart ? "Restart model to apply" : "Restart anyway")) {
                            restartToApply()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(restarting)
                    }
                    if let feedback = restartFeedback {
                        InlineNotice(
                            message: feedback,
                            tone: feedback.hasPrefix("Applied") ? .success : (feedback.hasPrefix("No changes") ? .info : .error)
                        )
                    }
                }
            }
        }
    }

    private var fixedKeySection: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            HStack(spacing: RapidTheme.Space.md) {
                HStack(spacing: RapidTheme.Space.xs) {
                    if revealInput {
                        TextField("Fixed API key", text: $fixedKeyInput)
                            .textFieldStyle(.roundedBorder)
                            .focused($keyFieldFocused)
                            .accessibilityIdentifier("Settings.APIAuth.FixedKey")
                    } else {
                        SecureField("Fixed API key", text: $fixedKeyInput)
                            .textFieldStyle(.roundedBorder)
                            .focused($keyFieldFocused)
                            .accessibilityIdentifier("Settings.APIAuth.FixedKey")
                    }
                    QuietIconButton(
                        symbol: revealInput ? "eye.slash" : "eye",
                        label: revealInput ? "Hide key" : "Show key",
                        size: RapidTheme.ControlHeight.mini
                    ) {
                        revealInput.toggle()
                        // P2-5: the TextField/SecureField swap rebuilds the
                        // field, which drops first responder. Re-assert
                        // focus on the next runloop tick so a half-typed
                        // key survives the reveal toggle.
                        DispatchQueue.main.async {
                            keyFieldFocused = true
                        }
                    }
                    .accessibilityIdentifier("Settings.APIAuth.RevealInput")
                }
                .onAppear {
                    if fixedKeyInput.isEmpty,
                       let existing = APIAuthConfig.defaults.string(forKey: APIAuthConfig.fixedKeyStorageKey) {
                        fixedKeyInput = existing
                    }
                }
                Button("Save") {
                    saveFixedKey()
                }
                .buttonStyle(.borderedProminent)
            }
            if savedNotice {
                InlineNotice(
                    message: isServing
                        ? "Saved — takes effect after you restart the model."
                        : "Saved — will be used on the next model start.",
                    tone: .success
                )
                .transition(.opacity)
            }
            if !hasFixedKey {
                // P1-1: without a stored key the engine silently falls back
                // to a fresh random key every launch. Surface it instead of
                // letting the panel claim "Applied" for a fixed mode that
                // never actually fixed anything.
                InlineNotice(
                    message: "No fixed key stored yet — the engine falls back to a fresh random key until you save one.",
                    tone: .warning
                )
            }
            InlineNotice(
                message: "The key is stored in app preferences. Keep it stable: this is the value external tools should send as the Bearer token.",
                tone: .info
            )
        }
    }

    private var isServing: Bool {
        if case .ready = server.state { return true }
        return false
    }

    /// True when the running engine's auth differs from what the current
    /// settings would spawn (mode or fixed key), so a restart would
    /// actually change something. Delegates to the pure helper so the
    /// effective-mode fallback (fixed-without-key ⇒ random) is handled in
    /// exactly one place.
    private var authNeedsRestart: Bool {
        APIAuthConfig.needsRestart(
            activeMode: server.activeAuthMode,
            activeBearer: server.activeBearer,
            isServing: isServing
        )
    }

    private func saveFixedKey() {
        let trimmed = fixedKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        APIAuthConfig.defaults.set(trimmed, forKey: APIAuthConfig.fixedKeyStorageKey)
        savedNotice = true
        restartFeedback = nil
        // P2-4: auto-dismiss the confirmation so it doesn't linger forever.
        // Cancel any previous dismissal so a rapid re-save doesn't get
        // cleared early by the first save's timer.
        noticeDismissTask?.cancel()
        noticeDismissTask = Task {
            try? await Task.sleep(nanoseconds: 4_000_000_000)
            if !Task.isCancelled {
                savedNotice = false
            }
        }
    }

    private func restartToApply() {
        guard case .ready(let alias) = server.state else { return }
        guard authNeedsRestart else {
            restartFeedback = "No changes — auth is already in effect."
            return
        }
        // P1-1 hardening: never claim "Applied" for a fixed mode with no
        // stored key — the restart would just mint another random key.
        if mode == .fixed && !hasFixedKey {
            restartFeedback = "Save a fixed key first — without one the engine falls back to a random key."
            return
        }
        restarting = true
        restartFeedback = nil
        Task {
            let ok = await server.restart(alias: alias)
            restarting = false
            if ok {
                restartFeedback = "Applied — the engine restarted with the new auth mode."
            } else {
                restartFeedback = "Restart failed — check the server logs and try again."
            }
        }
    }
}
