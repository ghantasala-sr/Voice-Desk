"""Retrieval: corpus -> chunks -> embeddings -> hybrid search.

Retrieval has two jobs here, and the second is the one people forget:

  1. Find the chunk that answers the question.
  2. Produce a score you can THRESHOLD on — "the top score is too low" is the
     first soft handoff signal and the main defence against confident nonsense.

Job 2 is why this is hybrid search with a calibrated threshold rather than
`ORDER BY embedding <=> query LIMIT 4`. See docs/05-retrieval.md.
"""
