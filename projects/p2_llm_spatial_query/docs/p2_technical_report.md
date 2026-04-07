# P2 Technical Report: Local LLM-Powered Spatial Query Interface

## 1. Objective

Develop a natural language interface for querying a Sierra Nevada forestry
GeoPackage database using a locally hosted large language model (LLM).  The
system translates plain-English questions into validated SpatiaLite SQL
queries, executes them against the GeoPackage, and returns human-readable
results.  All inference runs on-premises with no data leaving the local
network, addressing data sovereignty requirements common in forestry and
land management contexts.

## 2. Motivation

Geospatial databases in forestry operations contain rich spatial and
attribute data across harvest units, stream networks, road systems, and
sensitive habitat areas.  However, querying these databases requires
knowledge of SQL and spatial functions that many field foresters and
planners lack.  A natural language interface lowers this barrier while
maintaining the precision and auditability of SQL-based queries.

Key motivations:

- **Accessibility**: Enable non-technical users to query spatial data.
- **Data privacy**: Local LLM inference keeps sensitive operational data
  on-premises.
- **Safety**: A strict validation layer prevents any data modification.
- **Auditability**: Every query produces an inspectable SQL statement.

## 3. Architecture

The pipeline follows a Retrieval-Augmented Generation (RAG) pattern with
five stages:

```
User Question
     |
     v
[Few-Shot Selection] -- keyword overlap retrieval from example bank
     |
     v
[Prompt Builder] -- system prompt (schema + rules) + user prompt (examples + question)
     |
     v
[Local LLM] -- Llama 3 8B Instruct (GPTQ 4-bit) via vLLM / OpenAI-compatible API
     |
     v
[SQL Validator] -- whitelist/blocklist safety checks
     |
     v
[SpatiaLite Executor] -- GeoPackage query via sqlite3 + mod_spatialite
     |
     v
[Formatter] -- human-readable result summary
```

### 3.1 Components

| Module              | Responsibility                                          |
|---------------------|---------------------------------------------------------|
| `schema_extractor`  | Introspect GeoPackage tables, columns, geometry types   |
| `prompt_builder`    | Assemble system/user prompts with schema and few-shots  |
| `sql_generator`     | Call local LLM endpoint, parse SQL from response        |
| `sql_validator`     | Enforce safety rules on generated SQL                   |
| `executor`          | Execute validated SQL via SpatiaLite                    |
| `formatter`         | Produce human-readable output summaries                 |
| `pipeline`          | Orchestrate all stages via `run_query()`                |

### 3.2 Technology Stack

- **LLM**: Meta Llama 3 8B Instruct, GPTQ 4-bit quantized
- **Inference server**: vLLM or text-generation-inference (OpenAI-compatible API)
- **Database**: GeoPackage (SQLite + OGC extensions)
- **Spatial engine**: SpatiaLite (mod_spatialite)
- **CRS**: EPSG:3310 (California Albers NAD83)

## 4. Safety Design

The SQL validator is the critical safety component.  It enforces a
defence-in-depth strategy with four independent checks applied to every
generated SQL statement before execution:

### 4.1 SELECT-Only Constraint

The sanitized SQL must begin with `SELECT` (case-insensitive).  Any other
leading keyword is rejected immediately.

### 4.2 Blocked Operation Detection

A blocklist of destructive operations is checked via word-boundary regex
matching: `DELETE`, `DROP`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`.  This
catches attempts to embed destructive operations in subqueries or UNION
clauses.

### 4.3 Spatial Function Whitelist

Only explicitly approved spatial functions are permitted:
`ST_Buffer`, `ST_Intersects`, `ST_Within`, `ST_Contains`, `ST_Area`,
`ST_Length`.  Any other `ST_*` function call is rejected.

### 4.4 Multi-Statement Injection Prevention

Semicolons are detected after removing string literal content.  Any
statement containing a semicolon (whether followed by additional SQL or
trailing) is rejected.  This prevents classic injection attacks of the
form `SELECT ...; DROP TABLE ...`.

### 4.5 Comment Stripping

SQL comments (`--` single-line and `/* */` multi-line) are removed before
validation to prevent attackers from hiding blocked operations inside
comments.

## 5. Few-Shot Learning Approach

The system uses a bank of 15 curated natural-language-to-SQL example pairs
covering common forestry spatial queries.  At query time, the most relevant
examples are selected by keyword overlap scoring:

1. Tokenize the user query and each example question into lowercase words
   (length > 2 characters).
2. Score each example by the size of the token set intersection.
3. Return the top-K examples (default K=5) as in-context demonstrations.

This lightweight retrieval approach avoids the complexity and latency of
embedding-based retrieval while providing effective example selection for
the domain-specific query vocabulary.

## 6. Example Queries and Results

### 6.1 Simple Attribute Query

**Question**: "Show all harvest units prescribed for clearcut"

**Generated SQL**:
```sql
SELECT unit_name, acres, prescription
FROM harvest_units
WHERE prescription = 'clearcut'
```

### 6.2 Spatial Buffer Query

**Question**: "Find harvest units within 200 feet of Class I streams"

**Generated SQL**:
```sql
SELECT h.*
FROM harvest_units h, streams s
WHERE s.stream_class = 'I'
  AND ST_Intersects(h.geometry, ST_Buffer(s.geometry, 60.96))
```

### 6.3 Aggregate with Spatial Join

**Question**: "Calculate total stream length by stream class"

**Generated SQL**:
```sql
SELECT stream_class, SUM(ST_Length(geometry)) AS total_length_m
FROM streams
GROUP BY stream_class
```

### 6.4 Cross-Layer Overlap Analysis

**Question**: "Find sensitive habitats that overlap with any harvest unit"

**Generated SQL**:
```sql
SELECT sh.*
FROM sensitive_habitats sh, harvest_units h
WHERE ST_Intersects(sh.geometry, h.geometry)
```

## 7. Database Schema

The GeoPackage contains six layers representing the spatial data model
for Sierra Nevada forestry operations:

| Layer               | Geometry    | Key Attributes                           |
|---------------------|-------------|------------------------------------------|
| harvest_units       | Polygon     | unit_id, unit_name, acres, prescription  |
| streams             | LineString  | stream_id, stream_class, name            |
| roads               | LineString  | road_id, road_class, surface             |
| sensitive_habitats  | Polygon     | habitat_id, species, status              |
| ownership_parcels   | Polygon     | parcel_id, owner, ownership_type         |
| lidar_tile_index    | Polygon     | tile_id, acquisition_date, point_density |

All layers use EPSG:3310 (California Albers NAD83) with units in meters.

## 8. Limitations and Future Work

- **Keyword retrieval**: The current few-shot selection uses simple keyword
  overlap.  Embedding-based retrieval (e.g. sentence-transformers) would
  improve semantic matching for paraphrased queries.
- **Schema coverage**: Complex multi-table joins across more than two
  layers may require additional few-shot examples or chain-of-thought
  prompting.
- **Quantization trade-offs**: GPTQ 4-bit quantization reduces memory
  requirements but may degrade SQL generation accuracy for complex queries.
  Evaluation against a held-out query benchmark is planned.
- **Streaming results**: Large result sets are currently loaded into memory.
  Pagination or streaming would improve scalability.

## 9. References

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
  ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-
  intensive NLP tasks. *Advances in Neural Information Processing Systems*,
  33, 9459--9474.
- Zhong, V., Xiong, C., & Socher, R. (2017). Seq2SQL: Generating
  structured queries from natural language using reinforcement learning.
  *arXiv preprint arXiv:1709.00103*.
- Meta AI (2024). Llama 3: Open foundation and fine-tuned chat models.
  https://llama.meta.com/llama3/
- SpatiaLite. https://www.gaia-gis.it/fossil/libspatialite/
- OGC GeoPackage Standard. https://www.geopackage.org/
- California Forest Practice Rules. California Department of Forestry and
  Fire Protection (CAL FIRE).
