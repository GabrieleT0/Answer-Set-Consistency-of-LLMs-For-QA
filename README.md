# ANSWER SET CONSISTENCY OF LLMS FOR QUESTION ANSWERING

---
## Summary Table Column Descriptions

This table reports summary statistics aggregated over all question–answer combinations (e.g., 450 combos) for each `(dataset, llm, action)` triple.  
Values are reported as **ratios** (accuracy, “idk” rate) or **averages** (Jaccard similarities).


### Identifiers
- **dataset** – dataset used for evaluation  
- **action** – strategy applied (e.g., `zero-shot`, `wikidata`, `fixing`, `classification`)  
- **llm** – the large language model used  


### Relation Accuracies
Fraction of cases where the logical relation is correctly satisfied.
- **?A1=A2** – A1 equals A2  
- **?A1=A3+A4** – A1 equals the union of A3 and A4  
- **?A1>A3** – A1 strictly contains A3  
- **?A1>A4** – A1 strictly contains A4  
- **?A3∅A4** – A3 and A4 are disjoint  


### Average Jaccard Similarities
Mean Jaccard similarity scores across all combinations.
- **J(A1-A2)** – similarity between A1 and A2  
- **J(A1-A34)** – similarity between A1 and (A3 ∪ A4)  
- **J(A1-A1\*)** – similarity between A1 and A1\*  
- **J(A1-A1\*\*)** – similarity between A1 and A1\*\*  
- **J(A1\*-A1\*\*)** – similarity between A1\* and A1\*\*  
- **J_A1_ave** – average similarity across all A1 comparisons  


### "I don't know" Ratios
Fraction of answers that are empty or "I don't know".
- **idk_A1** – ratio for A1  
- **idk_A2** – ratio for A2  
- **idk_A3** – ratio for A3  
- **idk_A4** – ratio for A4  


### Stratified Metrics
Performance separated into positive and negative cases.
- **?A1=A2(+) / ?A1=A2(-)** – accuracy when the relation holds (+) vs when it does not (−)  
- **J(1-2)+ / J(1-2)-** – mean Jaccard similarity for positive vs negative cases  
- **?A1>A3(+/−), ?A1>A4(+/−), ?A3∅A4(+/−), ?A1=A3+A4(+/−)** – stratified accuracies  
- **J(1-34)+ / J(1-34)-** – stratified similarities for A1 vs A3∪A4  


### Statistical Significance (vs. zero-shot)
P-values from McNemar’s exact test comparing each action against the zero-shot baseline.  
Smaller values indicate stronger evidence of a difference.
- **p(A1=A2)** – for relation A1=A2  
- **p(A1=A3+A4)** – for relation A1=A3+A4  
- **p(A1>A3)** – for relation A1>A3  
- **p(A1>A4)** – for relation A1>A4  
- **p(A3∅A4)** – for relation A3∅A4
