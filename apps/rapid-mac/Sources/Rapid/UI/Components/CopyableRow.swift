import SwiftUI

/// A labelled value row with a mask toggle (for secrets), a Copy control,
/// and an optional trailing action. Shared by the Connect (launch) page and
/// the Settings API-auth panel so the two surfaces can never drift.
///
/// * The value renders masked behind ``•`` when ``masked`` is true, with an
///   eye toggle to reveal the plaintext (a ``QuietIconButton``).
/// * Copy places the REAL value on the clipboard; the masked form is the
///   only thing painted on screen unless revealed.
/// * ``placeholder`` is shown instead of the value when it is empty, and its
///   presence disables Copy — an empty clipboard write is a silent failure
///   the user only discovers in their editor.
struct CopyableRow: View {
    let label: String
    let value: String
    var masked: Bool = false
    /// Shown instead of the value when there is nothing real to show.
    /// Its presence also disables Copy — an empty clipboard write is a
    /// silent failure the user only discovers in their editor.
    var placeholder: String? = nil
    /// Optional trailing action (e.g. "Configure…" next to the API key
    /// row). When non-nil, a small gear button renders after the Copy
    /// button and calls this closure.
    var onConfigure: (() -> Void)? = nil
    /// AX-identifier namespace. Callers in different surfaces pass their
    /// own prefix (e.g. "ConnectTools", "Settings.APIAuth") so the
    /// identifiers stay unique and greppable.
    var identifierPrefix: String = "ConnectTools"
    @State private var reveal = false
    @State private var copied = false

    private var hasValue: Bool { !value.isEmpty }

    private var shown: String {
        guard hasValue else { return placeholder ?? "—" }
        guard masked, !reveal else { return value }
        return String(repeating: "•", count: min(value.count, 16))
    }

    var body: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Text(label)
                .font(RapidFont.secondary)
                .foregroundStyle(.secondary)
                .frame(width: 132, alignment: .leading)
            // Monospaced is correct here — this is an endpoint / key /
            // model id, one of the four sanctioned mono uses. The
            // not-yet placeholder drops to the prose font and tertiary,
            // so it can't be mistaken for a value worth copying.
            // Values render at full contrast. A not-yet placeholder is
            // one step down — clearly secondary, still comfortably
            // readable. It is NOT `.tertiary`: that made whole rows look
            // switched off when the row's own label and the sentence it
            // carries are both perfectly legitimate information.
            Text(shown)
                .font(hasValue ? RapidFont.code : RapidFont.secondary)
                .foregroundStyle(hasValue ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                .lineLimit(1)
                .truncationMode(.middle)
                // Always selectable: the two `textSelection` states are
                // different types so they can't share a ternary, and
                // the real guard against pasting a placeholder is the
                // disabled Copy button below, not selection.
                .textSelection(.enabled)
            Spacer(minLength: RapidTheme.Space.xs)
            if masked, hasValue {
                QuietIconButton(
                    symbol: reveal ? "eye.slash" : "eye",
                    label: reveal ? "Hide key" : "Show key",
                    size: RapidTheme.ControlHeight.mini
                ) {
                    reveal.toggle()
                }
                .accessibilityIdentifier("\(identifierPrefix).Reveal.\(label)")
            }
            QuietIconButton(
                symbol: copied ? "checkmark" : "doc.on.doc",
                label: "Copy \(label)",
                help: hasValue
                    ? "Copy \(label)"
                    : "Start a model to generate a valid key and configuration.",
                tint: copied ? RapidTheme.utilityActionSuccess : nil,
                size: RapidTheme.ControlHeight.mini
            ) {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(value, forType: .string)
                copied = true
                Task {
                    try? await Task.sleep(nanoseconds: 1_200_000_000)
                    copied = false
                }
            }
            .disabled(!hasValue)
            .accessibilityIdentifier("\(identifierPrefix).Copy.\(label)")
            if let onConfigure {
                QuietIconButton(
                    symbol: "gearshape",
                    label: "Configure \(label)",
                    help: "Open Settings to configure the API key mode.",
                    size: RapidTheme.ControlHeight.mini
                ) {
                    onConfigure()
                }
                .accessibilityIdentifier("\(identifierPrefix).Configure.\(label)")
            }
        }
        .padding(.horizontal, RapidTheme.Space.md)
        .frame(height: RapidTheme.ControlHeight.medium)
    }
}
