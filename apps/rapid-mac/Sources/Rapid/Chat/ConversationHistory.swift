import Foundation

/// A saved conversation — the unit the sidebar "Older" list shows and the
/// on-disk history persists. ``ChatMessage`` is already ``Codable``, so a
/// conversation serialises as-is.
struct ChatConversation: Identifiable, Codable, Equatable {
    let id: UUID
    var title: String
    var messages: [ChatMessage]
    let createdAt: Date
    var updatedAt: Date
}

/// On-disk store for the conversation history. One JSON file under
/// Application Support, keyed by bundle identifier so a dogfood-isolated
/// instance (rewritten bundle id) keeps its own history separate from the
/// real app's. Writes are atomic; a missing / unreadable file reads as an
/// empty history (first run).
enum ConversationStore {
    static func fileURL() -> URL {
        let base = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first ?? FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent(
            Bundle.main.bundleIdentifier ?? "Rapid",
            isDirectory: true
        )
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        return dir.appendingPathComponent("conversations.json")
    }

    static func load() -> [ChatConversation] {
        let url = fileURL()
        // A MISSING file is a normal first run → empty history. A file that
        // EXISTS but fails to decode (schema change, corruption) must NOT be
        // silently discarded — the next save would atomically overwrite the
        // user's whole history with a single new conversation. Side it to
        // `.corrupt-<t>` so it's recoverable, then start empty.
        guard let data = try? Data(contentsOf: url) else { return [] }
        if let decoded = try? JSONDecoder().decode([ChatConversation].self, from: data) {
            return decoded
        }
        let backup = url.deletingLastPathComponent()
            .appendingPathComponent(
                "conversations.corrupt-\(Int(Date().timeIntervalSince1970)).json"
            )
        try? FileManager.default.moveItem(at: url, to: backup)
        return []
    }

    /// Persist off the main actor — the caller (``ChatViewModel``) is
    /// ``@MainActor`` and every save re-encodes the whole history, so
    /// encoding + disk I/O must not run on the main thread or history
    /// growth would stall the UI. The snapshot is passed by value (Codable
    /// value types), so the background write sees a stable copy.
    static func save(_ conversations: [ChatConversation]) {
        DispatchQueue.global(qos: .utility).async {
            guard let data = try? JSONEncoder().encode(conversations) else { return }
            try? data.write(to: fileURL(), options: .atomic)
        }
    }

    /// Derive a one-line title from the transcript — the first user
    /// message, whitespace-collapsed and length-capped (Ollama shows the
    /// opening message as the row label). Falls back to "New chat".
    static func title(from messages: [ChatMessage]) -> String {
        guard let first = messages.first(where: { $0.role == .user }) else {
            return "New chat"
        }
        let collapsed = first.content
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        if collapsed.isEmpty { return "New chat" }
        return collapsed.count > 42
            ? String(collapsed.prefix(42)) + "…"
            : collapsed
    }
}
