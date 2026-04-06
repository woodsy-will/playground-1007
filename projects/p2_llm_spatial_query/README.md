# P2 — Local LLM-Powered Spatial Query Interface

## Objective

Natural language interface translating user queries into spatial SQL/ArcPy operations against a file geodatabase. Locally hosted LLM with RAG over schema metadata and few-shot examples.

## Data Requirements

- **Geodatabase:** FGDB containing harvest units, classified streams, roads, sensitive habitats, ownership parcels, LiDAR tile index
- **LLM:** Quantized open-source model (Llama 3 8B or Mistral 7B) on Proxmox infrastructure
- **CRS:** EPSG:3310

## Architecture

```
User query (natural language)
        │
        ▼
RAG retrieval (schema metadata + few-shot examples)
        │
        ▼
LLM generates SQL/ArcPy expression
        │
        ▼
Validation layer (whitelist SELECT, reject destructive ops)
        │
        ▼
Execution against FGDB
        │
        ▼
Map layer or summary table returned
        │
        ▼
Zero-result? → Reformulation loop
```

## Example Queries

```
"Show me all proposed harvest units within 200 feet of a Class I watercourse
 that intersect with known sensitive habitat."

"How many acres of high-severity burn overlap with the Crane Flat timber sale area?"

"List all road segments with slope greater than 15% within the project boundary."
```

## Safety Constraints

- **Allowed:** SELECT, ST_Buffer, ST_Intersects, ST_Within, ST_Contains
- **Blocked:** DELETE, DROP, UPDATE, INSERT, ALTER, TRUNCATE
- Schema injected into every prompt — LLM never infers field names from context alone

## Key References

- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS* 33.
- Zhong, V., et al. (2017). Seq2SQL: Generating Structured Queries from Natural Language. arXiv:1709.00103.

## Status

🔲 Not started
