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
        guard let data = try? Data(contentsOf: fileURL()) else { return [] }
        return (try? JSONDecoder().decode([ChatConversation].self, from: data)) ?? []
    }

    static func save(_ conversations: [ChatConversation]) {
        guard let data = try? JSONEncoder().encode(conversations) else { return }
        try? data.write(to: fileURL(), options: .atomic)
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
