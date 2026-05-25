# RAG Failure Modes in Research Assistants

Retrieval-augmented generation systems fail in ways that look deceptively
reasonable. The most dangerous failure is not an empty answer; it is a fluent
answer grounded in the wrong passage. A research assistant must therefore carry
source identity, section metadata, and retrieval scores forward into answer
synthesis.

Chunking strategy is a quality-control decision. Large chunks preserve local
argument structure, but they make retrieval less precise and can waste context
window budget. Tiny chunks improve nearest-neighbor specificity, but they often
strip claims from the assumptions and definitions that make them true.

Operationally, embedding drift is a silent migration problem. If a corpus is
embedded with one model and queries are embedded with another, the system may
continue to return results while relevance degrades. Production systems should
version embedding models and rebuild or partition indexes when the model changes.

