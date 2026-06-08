import enum

class ChunkingStrategy(str, enum.Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    DOCUMENT_STRUCTURE = "document_structure"
    RECURSIVE = "recursive"
    HIERARCHICAL = "hierarchical"