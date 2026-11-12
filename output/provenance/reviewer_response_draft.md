# Draft response on ASCB question provenance

Thank you for this suggestion. We agree that “a novel handcrafted benchmark” can overstate the originality of individual questions. We will revise it to “a manually curated benchmark of relation-constrained question quadruples.” As described in [Section 3.2](../../45436_Answer_Set_Consistency_o.pdf#page=4) and [Appendix G](../../45436_Answer_Set_Consistency_o.pdf#page=19), existing QA datasets supplied candidate questions rather than ready-made ASCB quadruples. The construction required an equivalent Q1–Q2 pair, a Q3 whose answer set is a strict subset of Q1’s, and a Q4 denoting the complementary subset. Three authors participated in curation; each retained quadruple was independently checked by at least two authors.

To quantify text reuse, we compared the 1,800 questions in the three source-based subsets against the original raw datasets, including available source paraphrases. **338 (18.8%) match source text verbatim**, and another **23 (1.3%) differ only in formatting**. Thus, 361 (20.1%) are verbatim or formatting-only copies. A provisional review of source wording and answer conditions classifies 248 (13.8%) as close rephrasings and 1,191 (66.2%) as substantially altered or lacking a close source counterpart. We used semantic similarity to retrieve candidate source questions for inspection; we did not assign these categories with a fixed cosine threshold. Only 35 of the 900 Q3 and Q4 questions match a raw source question verbatim. The separate synthetic subset contains 600 final questions selected and revised from LLM-generated candidates. We will clarify this provenance and add before-and-after examples to Appendix G.

If the reviewer requests the operational method: We searched the raw English question and paraphrase fields for literal matches, then used `BAAI/bge-small-en-v1.5` sentence embeddings and cosine similarity to retrieve the five closest raw questions within each source subset. We inspected changed Q1 wording against its direct source anchor where available, or against the retrieved candidate, and distinguished apparent answer-preserving rephrasing from changes in entity, time, scope, or filter. Q2 inherited the Q1 judgment because ASCB intends Q1 and Q2 to have the same answer set. We treated nonverbatim Q3/Q4 as substantial changes when they added the designed subset or complement condition, except for eight close QAWiki rephrasings identified in review. The latter two counts are provisional editorial estimates.

For each ASCB question, the retrieval script L2-normalized its 384-dimensional
embedding and compared it with every raw-source question embedding in the same
source subset using `cos(q,s) = (q · s) / (||q|| ||s||)`. After normalization,
this is the vector dot product. A numerical cutoff alone does not reproduce
the reported categories: among 1,439 nonverbatim source-derived questions,
1,438 have a nearest-candidate cosine above 0.60 and 1,374 above 0.70. A
QAWiki question containing “not as first author” scores 0.9839 against a raw
question containing “as first author,” despite the changed answer condition.
Do not describe 248 as the number above a 0.60/0.70 cosine threshold.

## Audit caveats for the authors (not for submission without resolution)

- The 248/1,191 split is a reproducible **provisional estimate**. It is not an
  author-adjudicated provenance count: the current release lacks per-row
  LC-QuAD source IDs, and some QAWiki rows likewise lack source links. Review
  the row-level labels before treating them as final paper statistics.
- The `spinach` subset is not drawn from SPINACH's own 320-question dataset.
  Its 150 directly linked Q1 questions trace to 94 QALD and 56
  WikiWebQuestions questions bundled in SPINACH. All 46 literal matches in
  this 600-slot subset trace to QALD; all 22 case/space-only matches trace to
  WikiWebQuestions. Reconcile the paper's `QALD` label with this provenance.
- A provisional review of these 150 directly linked Q1 pairs yields 66 copies
  or formatting changes, 39 answer-preserving paraphrases, 32 changes to the
  answer condition, and 13 uncertain cases. For the full provisional split,
  those 13 cases were resolved using the stated conservative editorial rule.
- The 600 synthetic questions were produced from LLM-generated candidates and
  author revision, as Appendix G states; phrase their origin accordingly.

The [audit README](README.md) gives the source files and exact-match method;
[`exact_matches.tsv`](exact_matches.tsv) gives every slot's text-match result;
[`spinach_q1_semantic_review.tsv`](spinach_q1_semantic_review.tsv) and the README
give the provisionally reviewed source-to-Q1 pairs. The complete provisional
labels and their basis are in
[`provisional_similarity_split.tsv`](provisional_similarity_split.tsv).
