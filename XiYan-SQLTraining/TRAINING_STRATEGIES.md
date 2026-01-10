# XiYan-SQL Training Strategies: Complete Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture: Paper vs Implementation](#architecture-paper-vs-implementation)
3. [Multi-Task Training Strategy](#multi-task-training-strategy)
4. [Data Augmentation Pipeline](#data-augmentation-pipeline)
5. [Multi-Format Training](#multi-format-training)
6. [Multi-Generator Strategy](#multi-generator-strategy)
7. [MOMQ MOE Architecture](#momq-moe-architecture)
8. [Creating the Four Generators (SQLG₁-₄)](#creating-the-four-generators)
9. [Practical Training Examples](#practical-training-examples)
10. [Implementation Checklist](#implementation-checklist)

---

## Overview

**XiYan-SQL** is a multi-generator ensemble framework for Text-to-SQL conversion. This repository (**XiYan-SQLTraining**) implements the **training component** that creates diverse SQL generators using multi-task and multi-format fine-tuning strategies.

### Key Innovation

The paper describes a system with **4 SQL generators (SQLG₁-₄)** trained using:
- **Different strategies**: Varying augmentation, model sizes, architectures
- **Different data formats**: Schema variations, prompt formats, dialects
- **Multi-task training**: NL2SQL, self-refinement, candidate selection

**This repository provides the framework to train these generators.**

---

## Architecture: Paper vs Implementation

### Paper's Complete XiYan-SQL Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                     XiYan-SQL Full System                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Inputs                                                │
│    • Question (natural language query)                          │
│    • Evidence (optional hints/context)                          │
│    • Full Database Schema (complete M-Schema)                   │
│                         ↓                                        │
│  Stage 2: Schema Filter                                         │
│    • Filter schema to relevant tables/columns                   │
│    • Uses M-Schema metadata for intelligent filtering           │
│    • Output: Focused subset of schema                           │
│                         ↓                                        │
│  Stage 3: Multiple SQL Generation (4 generators)                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ SQLG₁   │  │ SQLG₂   │  │ SQLG₃   │  │ SQLG₄   │           │
│  │(Heavy   │  │(Moderate│  │(Light   │  │(No      │           │
│  │ Aug)    │  │ Aug)    │  │ Aug)    │  │ Aug)    │           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
│       │            │            │            │                  │
│       └────────────┴────────────┴────────────┘                  │
│                         ↓                                        │
│  Stage 4: SQL Selection (Ensemble Voting)                       │
│    • Compare candidates from all 4 generators                   │
│    • Vote or select best SQL                                    │
│                         ↓                                        │
│  Final SQL Output                                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### What This Repository Implements

| Component | Status | Location |
|-----------|--------|----------|
| **Stage 1: M-Schema Generation** | ✅ **Fully Implemented** | `data/data_utils/m_schema.py`, `data/data_utils/schema_engine.py` |
| **Stage 2: Schema Filter Training** | ✅ As data augmentation | `data/data_utils/aug_ops/schema_filter_aug.py` |
| **Stage 3: Generator Training** | ✅ **Fully Implemented** | This entire repository |
| **Multi-Task Templates** | ✅ Implemented | `data/data_utils/prompt_utils.py` |
| **MOMQ MOE Training** | ✅ Implemented | `train/xiyan_momq_sft.sh` |
| **Stage 4: Ensemble Inference** | ❌ Not included | Separate system required |

**Focus**: This repository trains the 4 generators (SQLG₁-₄) used in **Stage 3** of the XiYan-SQL pipeline.

**Important Note**: The Schema Filter appears in two contexts:
1. **Runtime (Stage 2)**: In the full XiYan-SQL system, Schema Filter runs BEFORE SQL generation to reduce schema complexity
2. **Training (Data Aug)**: During training, SchemaFilter augmentation simulates this filtering to make generators robust to schema variations

---

## Multi-Task Training Strategy

The paper mentions three main tasks in the multi-task fine-tuning strategy. Here's how they're implemented:

### Task 1: Question Inference (Basic NL2SQL)

**Purpose**: Generate SQL from natural language question, database schema, and optional evidence.

**Templates Available**:

| Dialect | Template Name | Language | Status |
|---------|--------------|----------|--------|
| SQLite | `NL2SQLITE_TEMPLATE` | Chinese | ✅ |
| SQLite | `NL2SQLITE_TEMPLATE_EN` | English | ✅ |
| PostgreSQL | `NL2PGSQL_TEMPLATE` | Chinese | ✅ |
| PostgreSQL | `NL2PGSQL_TEMPLATE_EN` | English | ✅ |
| MySQL | `NL2MYSQL_TEMPLATE` | Chinese | ✅ |
| Cypher (Neo4j) | `NL2CYPHER_TEMPLATE` | Chinese | ✅ |
| NGQL (NebulaGraph) | `NL2NGQL_TEMPLATE` | Chinese | ✅ |
| Generic SQL | `NL2SQL_TEMPLATE` | Chinese | ✅ |

**Example Prompt Structure** (SQLite, Chinese):
```
你是一名SQLite专家，现在需要阅读并理解下面的【数据库schema】描述，
以及可能用到的【参考信息】，并运用SQLite知识生成sql语句回答【用户问题】。

【用户问题】
Name movie titles released in year 1945. Sort by popularity.

【数据库schema】
【DB_ID】 movie_platform
【Schema】
# Table: movies
[
  (movie_id:INTEGER, Examples: [1, 2, 3]),
  (movie_title:TEXT, Examples: ['The Great Adventure', 'War Stories']),
  (movie_release_year:INTEGER, Examples: [1945, 1946, 1947]),
  (movie_popularity:REAL, Examples: [8.5, 7.2, 6.8])
]

【参考信息】
released in the year 1945 refers to movie_release_year = 1945;

【用户问题】
Name movie titles released in year 1945. Sort by popularity.

```sql
```

**Expected Output**:
```sql
SELECT movie_title FROM movies
WHERE movie_release_year = 1945
ORDER BY movie_popularity DESC LIMIT 1
```

**Configuration**:
```json
{
  "task_name": "nl2sqlite",
  "data_aug": true
}
```

**Code Location**: `data/data_utils/prompt_utils.py:16-29, 180-181`

---

### Task 2: Evidence Inference

**Status**: ⚠️ **Not implemented as a separate generation task**

**Current Implementation**: Evidence is treated as **input context**, not a generation target.

```python
# From data_processing.py
evidence = data_item.get("evidence", "")  # From dataset
prompt = template.format(..., evidence=evidence)  # Used as input
```

**How It's Used**:
- Evidence is extracted from datasets like BIRD (pre-labeled)
- Embedded in prompts as contextual information
- Helps guide SQL generation but not generated by the model

**Paper's Intent**: Evidence might have been a separate generation task in some experiments, but this codebase treats it as given contextual information.

---

### Task 3: Self-Refinement

**Purpose**: Correct failed SQL queries based on execution errors.

**Template**: `SQLITE_SELF_REFINE_TEMPLATE`

**Input Components**:
1. Database schema
2. Original question
3. Failed SQL query
4. Execution error message

**Example Prompt**:
```
你是一名SQLite专家，之前回复用户问题的【SQL】查询未能产生正确的结果，
你需要根据提供的【数据库schema】描述，可能用到的【参考信息】和
不正确SQL的【执行结果】来进行纠正，请提供一个能够正确回复【用户问题】的更正SQL。

【数据库schema】
[... schema information ...]

【参考信息】
[... evidence ...]

【用户问题】
What is the average rating for movies?

【SQL】
SELECT AVG(rating) FROM movies

【执行结果】
Error: no such column: rating
Did you mean: rating_score?

【更正SQL】
```sql
```

**Expected Output**:
```sql
SELECT AVG(rating_score) FROM movies
```

**How to Create Self-Refine Training Data**:

```python
# Step 1: Run inference with an initial model
initial_predictions = model.generate(test_data)

# Step 2: Execute SQL and capture errors
failed_samples = []
for pred in initial_predictions:
    result, error = execute_sql(pred['sql'], pred['db'])
    if error:
        failed_samples.append({
            "db_name": pred['db_name'],
            "question": pred['question'],
            "evidence": pred['evidence'],
            "db_schema": pred['db_schema'],
            "sql": pred['ground_truth_sql'],  # Correct SQL
            "pred_sql_res": [pred['failed_sql'], error]
        })

# Step 3: Save and configure for training
save_json("self_refine_data.json", failed_samples)

# In dataset config:
{
  "self_refine_task": {
    "data_path": "path/to/self_refine_data.json",
    "task_name": "self_refine",
    "data_aug": false,
    "sample_num": -1
  }
}
```

**Configuration**:
```json
{
  "task_name": "self_refine",
  "data_aug": false
}
```

**Code Location**: `data/data_utils/prompt_utils.py:94-111, 186-190`

---

### Task 4: Candidate Selection (Bonus)

**Purpose**: Select the best SQL from multiple candidates based on execution results.

**Template**: `SQL2SELECT_TEMPLATE`

**Input Components**:
1. Database schema
2. Question and evidence
3. Multiple SQL candidates with execution results
4. Candidate labels (候选1, 候选2, ...)

**Example Prompt**:
```
你是一名SQLite专家，针对【用户问题】，下面有3条候选【SQL】及该sql在数据库上的
【执行结果】（展示前10行）；你需要比较这些候选，分析不同的候选【SQL】之间的差异。
基于给出的【数据库schema】、【参考信息】和【用户问题】选择一个正确合理的结果。

【数据库schema】
[...]

【用户问题】
What is the highest rated movie?

==========

候选1
【SQL】
SELECT movie_title FROM movies ORDER BY rating_score DESC LIMIT 1
【执行结果】
movie_title
"The Shawshank Redemption"

候选2
【SQL】
SELECT MAX(movie_title) FROM movies WHERE rating_score = MAX(rating_score)
【执行结果】
Error: misuse of aggregate function MAX()

候选3
【SQL】
SELECT movie_title FROM movies ORDER BY movie_popularity DESC LIMIT 1
【执行结果】
movie_title
"Avengers: Endgame"

请输出所选择的候选是候选1, 候选2, 候选3
```

**Expected Output**:
```
候选1
```

**Configuration**:
```json
{
  "task_name": "sql2select",
  "data_aug": false
}
```

**Note**: Template exists but requires custom data preparation. You need to generate multiple SQL candidates and their execution results.

**Code Location**: `data/data_utils/prompt_utils.py:114-134`

---

## Data Augmentation Pipeline

The augmentation pipeline creates diverse training samples from the same underlying data. This is key to the "different data formats" strategy mentioned in the paper.

### Augmentation Compose

**Location**: `data/data_assembler.py:23-28`

```python
augment_compose = AugCompose([
    SchemaShuffle(tab_rand_p=0.5, col_rand_p=0.3),
    SchemaFilter(tab_rand_p=0.8, col_rand_p=0.7),
    SchemaPermute(),
    SQLTranslate()
])
```

Each augmentation operation is applied sequentially to create format variations.

---

### Augmentation 1: Schema Shuffle

**Purpose**: Train model to be order-invariant.

**Mechanism**:
- Randomly shuffles table order (50% probability)
- Randomly shuffles column order within tables (30% probability)
- Randomly shuffles foreign key order (20% probability)

**Example**:

*Original Schema*:
```
【Schema】
# Table: movies
[
  (movie_id:INTEGER),
  (movie_title:TEXT),
  (release_year:INTEGER)
]
# Table: ratings
[
  (rating_id:INTEGER),
  (movie_id:INTEGER),
  (score:REAL)
]
Foreign Keys: ratings.movie_id -> movies.movie_id
```

*After SchemaShuffle*:
```
【Schema】
# Table: ratings
[
  (score:REAL),
  (rating_id:INTEGER),
  (movie_id:INTEGER)
]
# Table: movies
[
  (release_year:INTEGER),
  (movie_id:INTEGER),
  (movie_title:TEXT)
]
Foreign Keys: ratings.movie_id -> movies.movie_id
```

**Why Important**: Real-world schemas don't have a canonical order. This makes the model robust to different schema presentations.

**Configuration**:
```python
SchemaShuffle(
    tab_rand_p=0.5,   # 50% chance to shuffle tables
    col_rand_p=0.3,   # 30% chance to shuffle columns
    fk_rand_p=0.2     # 20% chance to shuffle foreign keys (hardcoded)
)
```

**Code Location**: `data/data_utils/aug_ops/schema_aug.py:4-48`

---

### Augmentation 2: Schema Filter

**Purpose**: Simulate incomplete/noisy schema information; teach model to focus on relevant elements.

**Mechanism**:
- **Smart filtering**: Tables/columns mentioned in SQL are **always kept**
- Tables NOT in SQL: 80% chance to include (20% filtered out)
- Columns NOT in SQL: 70% chance to include (30% filtered out)
- Foreign keys are filtered to match remaining tables

**Example**:

*Original Schema (5 tables, 20 columns)*:
```
# Table: movies [5 columns]
# Table: ratings [5 columns]
# Table: users [5 columns]
# Table: genres [3 columns]
# Table: directors [2 columns]
```

*SQL Query*:
```sql
SELECT movie_title FROM movies WHERE release_year = 1945
```

*After SchemaFilter*:
```
# Table: movies [required: movie_title, release_year + maybe 2 others]
# Table: ratings [80% chance to include, with 70% of columns]
# Table: users [80% chance, might be filtered out]
# Table: genres [80% chance]
# Table: directors [80% chance]
```

**Why Important**:
- Mimics real scenarios where not all schema info is available
- Forces model to identify relevant tables/columns
- Acts as a form of attention mechanism during training

**Configuration**:
```python
SchemaFilter(
    tab_rand_p=0.8,   # 80% prob to include non-SQL tables
    col_rand_p=0.7    # 70% prob to include non-SQL columns
)
```

**Code Location**: `data/data_utils/aug_ops/schema_aug.py:51-99`

---

### Augmentation 3: Schema Permute

**Purpose**: Create format variations in how schema information is presented.

**Mechanism**: Randomly changes the "Value examples:" prefix format with weighted probabilities:

| Format | Probability | Example |
|--------|-------------|---------|
| `"Value examples:"` | 10% | `(rating:REAL) Value examples: [8.5, 7.2, 9.1]` |
| `"Examples:"` | 83% | `(rating:REAL) Examples: [8.5, 7.2, 9.1]` |
| `"示例值:"` (Chinese) | 3% | `(rating:REAL) 示例值: [8.5, 7.2, 9.1]` |
| `"values:"` | 3% | `(rating:REAL) values: [8.5, 7.2, 9.1]` |
| `"--"` style | 1% | `(rating:REAL) -- Value examples: [8.5, 7.2, 9.1]` |

**Example**:

*Original*:
```
(movie_id:INTEGER) Value examples: [1, 2, 3]
(movie_title:TEXT) Value examples: ['Avatar', 'Titanic']
```

*After Permute (83% case)*:
```
(movie_id:INTEGER) Examples: [1, 2, 3]
(movie_title:TEXT) Examples: ['Avatar', 'Titanic']
```

**Why Important**: Different data sources use different conventions. This creates format-invariant models.

**Configuration**:
```python
SchemaPermute()  # Uses hardcoded weighted probabilities
```

**Additional**: Also randomly adds/removes spaces around schema elements (50% probability).

**Code Location**: `data/data_utils/aug_ops/schema_aug.py:101-142`

---

### Augmentation 4: SQL Translate

**Purpose**: Normalize SQL queries by removing formatting artifacts.

**Mechanism**:
1. Remove SQL comments (lines starting with `--`)
2. Collapse excessive whitespace (3+ newlines → single space)
3. Extract clean SQL from markdown/formatted text

**Example**:

*Before*:
```sql
-- Get all movies from 1945
SELECT
    movie_title,
    release_year
FROM
    movies
WHERE
    release_year = 1945  -- Filter by year
ORDER BY
    movie_popularity DESC
```

*After*:
```sql
SELECT movie_title, release_year FROM movies WHERE release_year = 1945 ORDER BY movie_popularity DESC
```

**Why Important**: Focuses model on semantic SQL structure, not formatting style.

**Code Location**: `data/data_utils/aug_ops/schema_aug.py:187-205`

---

### Augmentation Pipeline Example

**Complete transformation**:

```python
# Input data item
data = {
    "db_name": "movie_db",
    "question": "What are the top 3 movies?",
    "evidence": "top refers to highest rating",
    "db_schema": """[Full schema with 5 tables, specific order]""",
    "sql": "SELECT movie_title FROM movies ORDER BY rating DESC LIMIT 3"
}

# Apply augmentation pipeline
augmented_data = augment_compose(data)

# Result:
# - Schema: shuffled order, filtered to ~3-4 tables, "Examples:" format
# - SQL: normalized, comments removed
# - Same semantic meaning, different surface form
```

---

## Multi-Format Training

Beyond augmentation, the framework supports multiple format variations through:

### 1. Dialect-Specific Prompts

**6 SQL Dialects Supported**:

Each dialect has a specialized prompt template that:
- Identifies the model as a dialect expert
- Uses dialect-specific terminology
- Includes dialect-appropriate syntax hints

**Example Differences**:

*SQLite Template*:
```
你是一名SQLite专家，现在需要...并运用SQLite知识生成sql语句...
```

*PostgreSQL Template*:
```
你是一名PostgreSQL专家，现在需要...并运用PostgreSQL知识生成sql语句...
```

*Cypher Template* (Graph DB):
```
你是一名Neo4j专家，现在需要...并运用Cypher知识生成Cypher Query语句...
【图数据库schema】
...
```cypher
```

**Configuration**:
```json
{
  "dataset1": {"task_name": "nl2sqlite"},
  "dataset2": {"task_name": "nl2postgresql"},
  "dataset3": {"task_name": "nl2mysql"},
  "dataset4": {"task_name": "cypher"}
}
```

---

### 2. Language Variations

**Chinese and English Prompts**:

Both SQLite and PostgreSQL have English variants:
- `NL2SQLITE_TEMPLATE` (Chinese)
- `NL2SQLITE_TEMPLATE_EN` (English)
- `NL2PGSQL_TEMPLATE` (Chinese)
- `NL2PGSQL_TEMPLATE_EN` (English)

**English Example**:
```
You are a SQLite expert. You need to read and understand the following
【Database Schema】description and the possible provided【Evidence】,
and use valid SQLite knowledge to generate SQL for answering the【Question】.
```

**Use Case**: Training multilingual models or models that work across different language contexts.

---

### 3. M-Schema Format

**M-Schema** (Metadata-enhanced Schema) is XiYan's key innovation:

**Standard Schema**:
```sql
CREATE TABLE movies (
  movie_id INTEGER PRIMARY KEY,
  title TEXT,
  year INTEGER
);
```

**M-Schema Format**:
```
【DB_ID】 movie_platform
【Schema】
# Table: movies
[
  (movie_id:INTEGER, Primary Key, Comment: Unique identifier),
  (title:TEXT, Examples: ['The Matrix', 'Inception', 'Interstellar']),
  (year:INTEGER, Examples: [1999, 2010, 2014], Range: [1900-2024]),
  (rating:REAL, Examples: [8.7, 8.8, 8.6], Category: numeric)
]
Foreign Keys: ratings.movie_id -> movies.movie_id
```

**M-Schema Components**:
1. **Field Types**: Data types for each column
2. **Examples**: Actual data samples from the database
3. **Comments**: Column descriptions
4. **Categories**: Field classifications (numeric, date, string)
5. **Constraints**: Primary keys, foreign keys
6. **Ranges**: Value ranges for numeric fields

**Why M-Schema**:
- Provides **semantic context** beyond structure
- Helps model understand **data distribution**
- Enables **better value selection** in SQL
- Supports **schema-aware generation**

**Code Location**: `data/data_utils/m_schema.py`, `data/data_utils/schema_engine.py`

---

## Multi-Generator Strategy

The paper describes training **4 diverse generators (SQLG₁-₄)** to create an ensemble. Here's how to implement this strategy using the provided framework.

### Core Concept

Instead of training a single "best" model, train **4 different models** with:
- Different augmentation intensities
- Different model sizes/architectures
- Different task mixtures
- Different data formats

During inference (not in this repo), combine their outputs through ensemble voting.

---

### Strategy Dimensions

You can create diversity across 5 dimensions:

| Dimension | Generator 1 | Generator 2 | Generator 3 | Generator 4 |
|-----------|-------------|-------------|-------------|-------------|
| **Augmentation** | Heavy (90%) | Moderate (50%) | Light (20%) | None (0%) |
| **Model Size** | 7B | 14B | 7B MOE | 32B |
| **Task Mix** | Pure NL2SQL | NL2SQL+Refine | Multi-dialect | All tasks |
| **Data Format** | All formats | Schema-filtered | Shuffled only | Original |
| **Language** | Chinese | English | Mixed | Chinese |

---

### Dimension 1: Augmentation Intensity

**Create 4 different data assembly configs**:

**SQLG₁ - Heavy Augmentation (Maximum Diversity)**:
```python
# config_heavy_aug.json
augment_compose_g1 = AugCompose([
    SchemaShuffle(tab_rand_p=0.9, col_rand_p=0.7),  # 90%, 70%
    SchemaFilter(tab_rand_p=0.9, col_rand_p=0.9),   # 90%, 90%
    SchemaPermute(),
    SQLTranslate()
])
```

**SQLG₂ - Moderate Augmentation (Balanced)**:
```python
# config_moderate_aug.json (default in repo)
augment_compose_g2 = AugCompose([
    SchemaShuffle(tab_rand_p=0.5, col_rand_p=0.3),  # 50%, 30%
    SchemaFilter(tab_rand_p=0.8, col_rand_p=0.7),   # 80%, 70%
    SchemaPermute(),
    SQLTranslate()
])
```

**SQLG₃ - Light Augmentation (Conservative)**:
```python
# config_light_aug.json
augment_compose_g3 = AugCompose([
    SchemaShuffle(tab_rand_p=0.2, col_rand_p=0.1),  # 20%, 10%
    SchemaFilter(tab_rand_p=0.5, col_rand_p=0.5),   # 50%, 50%
    SchemaPermute(),
    SQLTranslate()
])
```

**SQLG₄ - No Augmentation (Pure Original)**:
```python
# config_no_aug.json
# Set data_aug: false in dataset config
{
  "bird_train": {
    "data_path": "...",
    "task_name": "nl2sqlite",
    "data_aug": false,  # <-- No augmentation
    "sample_num": -1
  }
}
```

**Implementation**:
1. Modify `data/data_assembler.py:23-28` with different probabilities
2. Run data assembly 4 times to create 4 datasets
3. Train 4 separate models on each dataset

---

### Dimension 2: Model Architecture

**SQLG₁ - Qwen2.5-Coder-7B (Standard SFT)**:
```bash
# xiyan_sft_g1.sh
MODEL="model/Qwen/Qwen2.5-Coder-7B-Instruct"
USE_LORA=True
LORA_R=512
EPOCH=5
LR=1e-6
OUTPUT="output/dense/sqlg1_7b_sft/"
```

**SQLG₂ - Qwen2.5-Coder-14B (Larger)**:
```bash
# xiyan_sft_g2.sh
MODEL="model/Qwen/Qwen2.5-Coder-14B-Instruct"
USE_LORA=True
LORA_R=512
EPOCH=5
LR=1e-6
OUTPUT="output/dense/sqlg2_14b_sft/"
```

**SQLG₃ - Qwen2.5-Coder-7B (MOMQ MOE)**:
```bash
# xiyan_momq_sft_g3.sh
MODEL="model/Qwen/Qwen2.5-Coder-7B-Instruct"
USE_MOE_LORA=True
num_experts=24
num_experts_per_tok=2
enable_dialect_router=True
EPOCH=5
LR=1e-5
OUTPUT="output/momq/sqlg3_7b_moe/"
```

**SQLG₄ - Qwen2.5-Coder-32B (Largest, Full FT)**:
```bash
# xiyan_sft_g4.sh
MODEL="model/Qwen/Qwen2.5-Coder-32B-Instruct"
USE_LORA=False  # Full fine-tuning
EPOCH=3
LR=5e-7
OUTPUT="output/dense/sqlg4_32b_full/"
```

---

### Dimension 3: Task Mixture

**SQLG₁ - Pure NL2SQL**:
```json
{
  "bird_train": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2sqlite.json",
    "task_name": "nl2sqlite",
    "data_aug": true,
    "sample_num": -1
  },
  "spider_train": {
    "data_path": "data_warehouse/spider/processed_data/train_nl2sqlite.json",
    "task_name": "nl2sqlite",
    "data_aug": true,
    "sample_num": -1
  }
}
```

**SQLG₂ - NL2SQL + Self-Refine (50/50)**:
```json
{
  "nl2sql_part": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2sqlite.json",
    "task_name": "nl2sqlite",
    "data_aug": true,
    "sample_num": 5000
  },
  "self_refine_part": {
    "data_path": "data_warehouse/bird_train/self_refine/refine_data.json",
    "task_name": "self_refine",
    "data_aug": false,
    "sample_num": 5000
  }
}
```

**SQLG₃ - Multi-Dialect Mix**:
```json
{
  "sqlite_data": {
    "task_name": "nl2sqlite",
    "sample_num": 3000,
    ...
  },
  "postgres_data": {
    "task_name": "nl2postgresql",
    "sample_num": 3000,
    ...
  },
  "mysql_data": {
    "task_name": "nl2mysql",
    "sample_num": 3000,
    ...
  }
}
```

**SQLG₄ - All Tasks (Comprehensive)**:
```json
{
  "nl2sql": {"task_name": "nl2sqlite", "sample_num": 4000},
  "self_refine": {"task_name": "self_refine", "sample_num": 3000},
  "nl2postgres": {"task_name": "nl2postgresql", "sample_num": 2000},
  "nl2mysql": {"task_name": "nl2mysql", "sample_num": 1000}
}
```

---

### Dimension 4: Prompt Language

**SQLG₁ - Chinese Only**:
```python
# In prompt_utils.py, use:
NL2SQLITE_TEMPLATE  # Chinese
NL2PGSQL_TEMPLATE   # Chinese
```

**SQLG₂ - English Only**:
```python
# Create English-only dataset config
# Use templates:
NL2SQLITE_TEMPLATE_EN  # English
NL2PGSQL_TEMPLATE_EN   # English
```

**SQLG₃ - Mixed Language (50/50)**:
```python
# Custom data assembler modification needed
def gen_train_prompt_mixed(idx, data_item, task_type):
    if random.random() < 0.5:
        # Use Chinese template
        template = NL2SQLITE_TEMPLATE
    else:
        # Use English template
        template = NL2SQLITE_TEMPLATE_EN
    ...
```

**SQLG₄ - Dialect-Specific**:
```python
# Use specialized templates for each dialect
# SQLite -> Chinese
# PostgreSQL -> English
# MySQL -> Chinese
```

---

### Recommended 4-Generator Setup

**Production-Ready Configuration**:

| Generator | Size | Aug | Tasks | Lang | Special |
|-----------|------|-----|-------|------|---------|
| **SQLG₁** | 7B | Heavy (90%) | Pure NL2SQL | Chinese | Diversity focus |
| **SQLG₂** | 14B | Moderate (50%) | NL2SQL + Refine | Chinese | Quality focus |
| **SQLG₃** | 7B MOE | Light (20%) | Multi-dialect | Mixed | Dialect specialist |
| **SQLG₄** | 32B | None (0%) | All tasks | English | Power model |

This setup balances:
- **Diversity**: Through different augmentation levels
- **Quality**: Through larger models
- **Specialization**: Through MOE and task-specific training
- **Robustness**: Through multi-task and multi-language training

---

## MOMQ MOE Architecture

**MOMQ** (Multi-Dialect Mixture of Experts) is a specialized architecture for handling multiple SQL dialects efficiently.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MOMQ MOE Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Question + Schema + sql_type (e.g., "postgresql")  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Base Model (Qwen2.5-Coder-7B)                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Dialect Router (first token position)               │  │
│  │  Maps: postgresql→0, mysql→1, sqlite→2, etc.         │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LoRA Expert Selection                                │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐    │  │
│  │  │Expert 1│ │Expert 2│ │Expert 3│ ... │Expert24│    │  │
│  │  └────────┘ └────────┘ └────────┘     └────────┘    │  │
│  │       ↓           ↓                                   │  │
│  │   Select 2 experts per token based on dialect         │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Share Experts (2 experts always active)             │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  Generated SQL (dialect-specific syntax)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

**1. Dialect Router** (`sft4xiyan.py:72-84`):
```python
if training_args.enable_dialect_router:
    sql_type = example.get("sql_type", "sqlite").lower()
    if sql_type == 'postgresql':
        labels[0] = 0  # Dialect ID at first token
    elif sql_type == 'mysql':
        labels[0] = 1
    elif sql_type == 'sqlite':
        labels[0] = 2
    elif sql_type == 'cypher':
        labels[0] = 3
    elif sql_type == 'ngql':
        labels[0] = 4
```

**Purpose**: Uses the first token position to encode dialect information, guiding expert selection.

**2. LoRA Experts**:
- **Total experts**: 24 (configurable)
- **Active per token**: 2 (configurable)
- **Shared experts**: 2 (always active)
- **Target modules**: Typically `down_proj` in MLP layers

**3. Routing Mechanism**:
- **Token-level**: Each token can route to different experts
- **Dialect-aware**: Dialect ID influences routing decisions
- **Auxiliary loss**: Encourages load balancing across experts

### Configuration

**Training Script** (`xiyan_momq_sft.sh`):
```bash
# MOE Configuration
USE_MOE_LORA=True
lora_route_type='token'              # token-level or dialect-level
moe_lora_target_modules=("down_proj") # Which layers to apply MOE
enable_dialect_router=True            # Enable dialect-specific routing
output_router_logits=True             # Output routing decisions
num_experts=24                        # Total number of experts
num_experts_per_tok=2                 # Active experts per token
share_expert_num=2                    # Shared experts (always active)

# Loss Configuration
router_aux_loss_coef=0.001            # Load balancing loss
dialect_router_loss_coef=0.01         # Dialect routing loss
enable_label_smooth=False             # Label smoothing (optional)
smooth_factor=0.01                    # If label smoothing enabled

# Training Configuration
LORA_R=128
LORA_SCALE=2
LR=1e-5                               # Higher than standard SFT
EPOCH=5
MAX_LENGTH=8192
```

### Benefits

1. **Parameter Efficiency**: Only 2 experts active per token (~8% of total)
2. **Dialect Specialization**: Different experts learn different SQL dialects
3. **Better Generalization**: Shared experts capture common patterns
4. **Faster Inference**: Sparse activation reduces computation

### When to Use MOMQ

✅ **Use MOMQ when**:
- Training on multiple SQL dialects (PostgreSQL, MySQL, SQLite, etc.)
- Need parameter-efficient multi-task learning
- Have dialect labels in training data (`sql_type` field)
- Want specialized behavior per dialect

❌ **Use Standard SFT when**:
- Single dialect only
- No dialect labels available
- Simpler deployment requirements
- Smaller model sizes (<7B)

---

## Creating the Four Generators

### Complete Step-by-Step Guide

Here's a production-ready guide to create all 4 generators.

---

### Phase 1: Data Preparation

**Step 1: Process Raw Data** (if starting from raw BIRD/Spider data):
```bash
cd XiYan-SQLTraining/data

# Process BIRD training data
bash data_processing.sh \
  data_warehouse/bird_train/raw_data/train.json \
  data_warehouse/bird_train/db_conn.json \
  data_warehouse/bird_train/processed_data/ \
  data_warehouse/bird_train/mschema/ \
  configs/datasets_all.json

# Process Spider training data (if available)
bash data_processing.sh \
  data_warehouse/spider/raw_data/train.json \
  data_warehouse/spider/db_conn.json \
  data_warehouse/spider/processed_data/ \
  data_warehouse/spider/mschema/ \
  configs/datasets_all.json
```

**Step 2: (Optional) Create Self-Refine Data**:
```python
# Generate self-refine training data
python create_self_refine_data.py \
  --initial_model path/to/initial_model \
  --test_data data_warehouse/bird_train/processed_data/train_nl2sqlite.json \
  --output data_warehouse/bird_train/self_refine/refine_data.json
```

---

### Phase 2: Create 4 Dataset Configurations

**Create 4 JSON config files**:

**configs/datasets_g1_heavy.json** (SQLG₁):
```json
{
  "bird_train": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2sqlite.json",
    "task_name": "nl2sqlite",
    "data_aug": true,
    "sample_num": -1,
    "sum_num": 9000
  }
}
```

**configs/datasets_g2_moderate.json** (SQLG₂):
```json
{
  "bird_nl2sql": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2sqlite.json",
    "task_name": "nl2sqlite",
    "data_aug": true,
    "sample_num": 6000,
    "sum_num": 9000
  },
  "bird_refine": {
    "data_path": "data_warehouse/bird_train/self_refine/refine_data.json",
    "task_name": "self_refine",
    "data_aug": false,
    "sample_num": 3000,
    "sum_num": 3000
  }
}
```

**configs/datasets_g3_multidialect.json** (SQLG₃):
```json
{
  "bird_sqlite": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2sqlite.json",
    "task_name": "nl2sqlite",
    "data_aug": true,
    "sample_num": 3000,
    "sum_num": 9000
  },
  "bird_postgres": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2postgresql.json",
    "task_name": "nl2postgresql",
    "data_aug": true,
    "sample_num": 3000,
    "sum_num": 5000
  },
  "bird_mysql": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2mysql.json",
    "task_name": "nl2mysql",
    "data_aug": true,
    "sample_num": 3000,
    "sum_num": 4000
  }
}
```

**configs/datasets_g4_comprehensive.json** (SQLG₄):
```json
{
  "bird_all": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2sqlite.json",
    "task_name": "nl2sqlite",
    "data_aug": false,
    "sample_num": -1,
    "sum_num": 9000
  }
}
```

---

### Phase 3: Modify Augmentation Parameters

**Edit `data/data_assembler.py` for each generator**:

**For SQLG₁ (Heavy)**:
```python
# Line 23-28 in data_assembler.py
augment_compose = AugCompose([
    SchemaShuffle(tab_rand_p=0.9, col_rand_p=0.7),
    SchemaFilter(tab_rand_p=0.9, col_rand_p=0.9),
    SchemaPermute(),
    SQLTranslate()
])
```

**For SQLG₂ (Moderate)** - use default values (already in code)

**For SQLG₃ (Light)**:
```python
augment_compose = AugCompose([
    SchemaShuffle(tab_rand_p=0.2, col_rand_p=0.1),
    SchemaFilter(tab_rand_p=0.5, col_rand_p=0.5),
    SchemaPermute(),
    SQLTranslate()
])
```

**For SQLG₄** - set `data_aug: false` in config (no code change needed)

---

### Phase 4: Assemble Training Datasets

```bash
cd XiYan-SQLTraining/data

# Generator 1 (Heavy Aug)
# First, modify augmentation params in data_assembler.py
bash data_assembler.sh \
  configs/datasets_g1_heavy.json \
  ../train/datasets/sqlg1_train_heavy.json

# Generator 2 (Moderate Aug)
# Restore default augmentation params
bash data_assembler.sh \
  configs/datasets_g2_moderate.json \
  ../train/datasets/sqlg2_train_moderate.json

# Generator 3 (Light Aug, Multi-dialect)
# Set light augmentation params
bash data_assembler.sh \
  configs/datasets_g3_multidialect.json \
  ../train/datasets/sqlg3_train_multidialect.json

# Generator 4 (No Aug)
bash data_assembler.sh \
  configs/datasets_g4_comprehensive.json \
  ../train/datasets/sqlg4_train_noaug.json
```

---

### Phase 5: Train the 4 Models

**Download Base Model First**:
```bash
cd XiYan-SQLTraining/train/utils
uv run model_download.py
# Select: Qwen2.5-Coder-7B-Instruct, 14B, 32B as needed
```

**Train SQLG₁ (7B, Heavy Aug, Pure NL2SQL)**:
```bash
cd XiYan-SQLTraining/train

# Edit xiyan_sft.sh:
EXPR_ID="sqlg1_7b_heavy_aug"
MODEL="model/Qwen/Qwen2.5-Coder-7B-Instruct"
DATA="datasets/sqlg1_train_heavy.json"
OUTPUT="output/dense/sqlg1_7b_heavy/"
USE_LORA=True
LORA_R=512
EPOCH=5
LR=1e-6
MAX_LENGTH=10240

bash xiyan_sft.sh
```

**Train SQLG₂ (14B, Moderate Aug, NL2SQL+Refine)**:
```bash
EXPR_ID="sqlg2_14b_moderate_refine"
MODEL="model/Qwen/Qwen2.5-Coder-14B-Instruct"
DATA="datasets/sqlg2_train_moderate.json"
OUTPUT="output/dense/sqlg2_14b_moderate/"
USE_LORA=True
LORA_R=512
EPOCH=5
LR=1e-6
MAX_LENGTH=10240

bash xiyan_sft.sh
```

**Train SQLG₃ (7B MOE, Light Aug, Multi-dialect)**:
```bash
# Edit xiyan_momq_sft.sh:
EXPR_ID="sqlg3_7b_moe_multidialect"
MODEL="model/Qwen/Qwen2.5-Coder-7B-Instruct"
DATA="datasets/sqlg3_train_multidialect.json"
OUTPUT="output/momq/sqlg3_7b_moe/"
USE_MOE_LORA=True
enable_dialect_router=True
num_experts=24
num_experts_per_tok=2
EPOCH=5
LR=1e-5
MAX_LENGTH=8192

bash xiyan_momq_sft.sh
```

**Train SQLG₄ (32B, No Aug, Comprehensive)**:
```bash
EXPR_ID="sqlg4_32b_noaug_comprehensive"
MODEL="model/Qwen/Qwen2.5-Coder-32B-Instruct"
DATA="datasets/sqlg4_train_noaug.json"
OUTPUT="output/dense/sqlg4_32b_full/"
USE_LORA=False  # Full fine-tuning
EPOCH=3
LR=5e-7
MAX_LENGTH=10240

bash xiyan_sft.sh
```

---

### Phase 6: Merge LoRA Adapters

**For SQLG₁, SQLG₂** (LoRA models):
```bash
cd XiYan-SQLTraining/train

# Merge SQLG₁
uv run utils/adapter_merge.py \
  --base_model model/Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter output/dense/sqlg1_7b_heavy/checkpoint-XXX \
  --output merged_models/sqlg1_7b_merged

# Merge SQLG₂
uv run utils/adapter_merge.py \
  --base_model model/Qwen/Qwen2.5-Coder-14B-Instruct \
  --adapter output/dense/sqlg2_14b_moderate/checkpoint-XXX \
  --output merged_models/sqlg2_14b_merged

# SQLG₃ (MOE) and SQLG₄ (full FT) don't need merging
```

---

### Phase 7: Evaluate Each Generator

```bash
cd XiYan-SQLTraining/evaluation

# Evaluate SQLG₁
bash sql_infer.sh \
  ../train/merged_models/sqlg1_7b_merged \
  sqlg1_eval \
  bird_evaluation/eval_set/bird_dev.json \
  4

bash sql_eval.sh \
  bird_evaluation/output/sqlg1_eval/sqlg1_eval_results.json \
  bird_evaluation/eval_set/bird_dev.json \
  bird_evaluation/db_conn.json \
  bird_evaluation/output/sqlg1_eval/sqlg1_scores.json

# Repeat for SQLG₂, SQLG₃, SQLG₄
```

---

### Phase 8: Ensemble (Out of Scope - Conceptual)

**Note**: This repository doesn't implement ensemble inference, but here's the concept:

```python
# Pseudo-code for ensemble system (follows XiYan-SQL paper pipeline)
def xiyan_ensemble_inference(question, full_schema, evidence):
    # Stage 1: Inputs (question, evidence, full database schema)
    # Inputs are provided as parameters

    # Stage 2: Schema Filter - Filter to relevant tables/columns FIRST
    filtered_schema = schema_filter(full_schema, question, evidence)
    # This reduces context size and focuses on relevant parts

    # Stage 3: Multiple SQL Generation - Generate from all 4 models
    # All generators use the FILTERED schema, not the full schema
    sql_g1 = sqlg1_model.generate(question, filtered_schema, evidence)
    sql_g2 = sqlg2_model.generate(question, filtered_schema, evidence)
    sql_g3 = sqlg3_model.generate(question, filtered_schema, evidence)
    sql_g4 = sqlg4_model.generate(question, filtered_schema, evidence)

    candidates = [sql_g1, sql_g2, sql_g3, sql_g4]

    # Stage 4: SQL Selection (Ensemble Voting/Selection)
    # Option A: Execution-based voting
    execution_results = [execute_sql(sql, db) for sql in candidates]
    voted_sql = majority_vote(candidates, execution_results)

    # Option B: Use SQL2SELECT model
    selector_prompt = SQL2SELECT_TEMPLATE.format(
        candidates=candidates,
        execution_results=execution_results,
        ...
    )
    selected_sql = selector_model.generate(selector_prompt)

    return voted_sql  # or selected_sql
```

---

## Practical Training Examples

### Example 1: Quick Start (Single Model)

Train a single SQLite model on BIRD data:

```bash
# 1. Setup environment
cd XiYan-SQLTraining
uv sync

# 2. Process data (assuming you have BIRD data)
cd data
bash data_processing.sh \
  data_warehouse/bird_train/raw_data/train.json \
  data_warehouse/bird_train/db_conn.json \
  data_warehouse/bird_train/processed_data/ \
  data_warehouse/bird_train/mschema/ \
  configs/datasets_all.json

# 3. Assemble training data
bash data_assembler.sh \
  configs/datasets_example.json \
  ../train/datasets/my_train.json

# 4. Download model
cd ../train/utils
uv run model_download.py
# Select: Qwen2.5-Coder-7B-Instruct

# 5. Train
cd ..
# Edit xiyan_sft.sh to set DATA="datasets/my_train.json"
bash xiyan_sft.sh

# 6. Merge LoRA (if used)
uv run utils/adapter_merge.py \
  --base_model model/Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter output/dense/namexxx_configxxx_date_xxx/checkpoint-XXX \
  --output merged_models/my_model

# 7. Evaluate
cd ../evaluation
bash sql_infer.sh \
  ../train/merged_models/my_model \
  my_eval \
  bird_evaluation/eval_set/bird_dev.json \
  4
```

**Time Estimate**: ~8-12 hours on 8x A100 GPUs

---

### Example 2: Multi-Dialect Training (MOMQ)

Train a multi-dialect model using MOE:

```bash
# 1. Prepare multi-dialect data
# Assume you have: train_nl2sqlite.json, train_nl2postgresql.json, train_nl2mysql.json

# 2. Create config
cat > data/configs/datasets_multidialect.json << EOF
{
  "sqlite_data": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2sqlite.json",
    "task_name": "nl2sqlite",
    "data_aug": true,
    "sample_num": 3000
  },
  "postgres_data": {
    "data_path": "data_warehouse/postgres_train/processed_data/train_nl2postgresql.json",
    "task_name": "nl2postgresql",
    "data_aug": true,
    "sample_num": 3000
  },
  "mysql_data": {
    "data_path": "data_warehouse/mysql_train/processed_data/train_nl2mysql.json",
    "task_name": "nl2mysql",
    "data_aug": true,
    "sample_num": 3000
  }
}
EOF

# 3. Assemble
cd data
bash data_assembler.sh \
  configs/datasets_multidialect.json \
  ../train/datasets/multidialect_train.json

# 4. Train with MOMQ
cd ../train
# Edit xiyan_momq_sft.sh:
# - Set DATA="datasets/multidialect_train.json"
# - Set dialect_num=3 (sqlite, postgres, mysql)
bash xiyan_momq_sft.sh

# 5. The MOE model is ready to use (no merging needed)
```

**Time Estimate**: ~10-15 hours on 8x A100 GPUs

---

### Example 3: Self-Refine Task Training

Create and train on self-refinement data:

```bash
# 1. First train a base model (see Example 1)

# 2. Generate predictions on training data
cd evaluation
bash sql_infer.sh \
  ../train/merged_models/base_model \
  base_predictions \
  ../data/data_warehouse/bird_train/processed_data/train_nl2sqlite.json \
  4

# 3. Create self-refine dataset (custom script needed)
cat > create_refine_data.py << 'EOF'
import json
from eval_utils.sql_utils import read_json, write_json, execute_sql

def create_self_refine_data(predictions_path, db_config_path, output_path):
    predictions = read_json(predictions_path)
    failed_samples = []

    for pred in predictions:
        # Execute predicted SQL
        result, error = execute_sql(pred['predicted_sql'], pred['db_name'], db_config_path)

        if error:
            # This is a failed prediction - good for self-refine
            failed_samples.append({
                "db_name": pred['db_name'],
                "question": pred['question'],
                "evidence": pred.get('evidence', ''),
                "db_schema": pred['db_schema'],
                "sql": pred['ground_truth_sql'],  # Correct SQL
                "pred_sql_res": [pred['predicted_sql'], str(error)]
            })

    write_json(output_path, failed_samples)
    print(f"Created {len(failed_samples)} self-refine samples")

if __name__ == "__main__":
    create_self_refine_data(
        "bird_evaluation/output/base_predictions/base_predictions_results.json",
        "bird_evaluation/db_conn.json",
        "../data/data_warehouse/bird_train/self_refine/refine_data.json"
    )
EOF

uv run create_refine_data.py

# 4. Create mixed dataset config
cat > ../data/configs/datasets_with_refine.json << EOF
{
  "nl2sql_main": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2sqlite.json",
    "task_name": "nl2sqlite",
    "data_aug": true,
    "sample_num": 6000
  },
  "self_refine": {
    "data_path": "data_warehouse/bird_train/self_refine/refine_data.json",
    "task_name": "self_refine",
    "data_aug": false,
    "sample_num": -1
  }
}
EOF

# 5. Assemble and train
cd ../data
bash data_assembler.sh \
  configs/datasets_with_refine.json \
  ../train/datasets/train_with_refine.json

cd ../train
# Edit xiyan_sft.sh to use this dataset
bash xiyan_sft.sh
```

**Time Estimate**: ~12-16 hours total (including base model training)

---

### Example 4: Custom Augmentation

Create custom augmentation for specific needs:

```python
# data/data_utils/aug_ops/custom_aug.py

class TableNameObfuscation(object):
    """
    Randomly replace table names with generic names like T1, T2, etc.
    Tests if model can understand relationships without semantic table names.
    """
    def __init__(self, obfuscate_prob=0.3):
        self.obfuscate_prob = obfuscate_prob

    def __call__(self, data: dict):
        if random.random() < self.obfuscate_prob:
            schema = data["db_schema"]
            sql = data["sql"]

            # Extract table names
            scm_dict, fk_item_list = scm_text2dict(schema)
            table_names = list(scm_dict.keys())

            # Create mapping: real_name -> generic_name
            table_mapping = {
                name: f"T{i+1}"
                for i, name in enumerate(table_names)
            }

            # Replace in schema and SQL
            for real_name, generic_name in table_mapping.items():
                schema = schema.replace(real_name, generic_name)
                sql = sql.replace(real_name, generic_name)

            data["db_schema"] = schema
            data["sql"] = sql

        return data


class ValueAnonymization(object):
    """
    Replace example values with anonymized versions.
    Tests if model relies too much on specific values.
    """
    def __init__(self, anonymize_prob=0.2):
        self.anonymize_prob = anonymize_prob

    def __call__(self, data: dict):
        if random.random() < self.anonymize_prob:
            schema = data["db_schema"]

            # Replace actual examples with generic placeholders
            # e.g., "Examples: ['Alice', 'Bob']" -> "Examples: ['Name_1', 'Name_2']"
            schema = re.sub(
                r"Examples: \[(.*?)\]",
                lambda m: f"Examples: [<VALUE_{i}> for i in range(3)]",
                schema
            )

            data["db_schema"] = schema

        return data
```

**Use custom augmentation**:
```python
# In data_assembler.py
from data_utils.aug_ops.custom_aug import TableNameObfuscation, ValueAnonymization

augment_compose = AugCompose([
    SchemaShuffle(tab_rand_p=0.5, col_rand_p=0.3),
    SchemaFilter(tab_rand_p=0.8, col_rand_p=0.7),
    SchemaPermute(),
    TableNameObfuscation(obfuscate_prob=0.3),  # New!
    ValueAnonymization(anonymize_prob=0.2),    # New!
    SQLTranslate()
])
```

---

## Implementation Checklist

Use this checklist to track your multi-generator training progress:

### Phase 1: Environment & Data Setup
- [ ] Install uv package manager
- [ ] Run `uv sync` in XiYan-SQLTraining directory
- [ ] Download BIRD training data (train.zip)
- [ ] Extract database files to correct location
- [ ] Verify database connections work
- [ ] Run data_processing.sh successfully
- [ ] Verify M-Schema generation
- [ ] Check processed data format

### Phase 2: Single Model Training (Baseline)
- [ ] Download Qwen2.5-Coder-7B-Instruct
- [ ] Create basic dataset config (pure NL2SQL)
- [ ] Run data_assembler.sh
- [ ] Modify xiyan_sft.sh with correct paths
- [ ] Start training (monitor GPU usage)
- [ ] Verify checkpoint saving works
- [ ] Merge LoRA adapter (if used)
- [ ] Run inference on dev set
- [ ] Run evaluation on dev set
- [ ] Record baseline scores

### Phase 3: Data Augmentation Experiments
- [ ] Test SchemaShuffle with different probabilities
- [ ] Test SchemaFilter with different probabilities
- [ ] Verify SchemaPermute works correctly
- [ ] Check SQLTranslate output
- [ ] Create 4 augmentation configs (heavy, moderate, light, none)
- [ ] Generate 4 training datasets
- [ ] Verify dataset sizes and diversity

### Phase 4: Multi-Task Data Preparation
- [ ] Train initial model for self-refine data creation
- [ ] Run inference on training set
- [ ] Execute SQL and collect errors
- [ ] Create self_refine training data
- [ ] Verify self_refine prompt format
- [ ] (Optional) Create candidate_selection data
- [ ] Create mixed task dataset configs

### Phase 5: Multi-Dialect Setup (If Applicable)
- [ ] Obtain PostgreSQL training data
- [ ] Obtain MySQL training data
- [ ] Process multi-dialect data
- [ ] Verify dialect labels (sql_type field)
- [ ] Create multi-dialect dataset config
- [ ] Test dialect-specific prompts

### Phase 6: Train Generator 1 (SQLG₁)
- [ ] Configure: 7B model, heavy aug, pure NL2SQL
- [ ] Set augmentation: 90% shuffle, 90% filter
- [ ] Prepare dataset
- [ ] Start training
- [ ] Monitor training loss
- [ ] Merge LoRA adapter
- [ ] Evaluate on dev set
- [ ] Save model and scores

### Phase 7: Train Generator 2 (SQLG₂)
- [ ] Configure: 14B model, moderate aug, NL2SQL+Refine
- [ ] Set augmentation: 50% shuffle, 80% filter
- [ ] Prepare mixed task dataset
- [ ] Start training
- [ ] Monitor training loss
- [ ] Merge LoRA adapter
- [ ] Evaluate on dev set
- [ ] Save model and scores

### Phase 8: Train Generator 3 (SQLG₃)
- [ ] Configure: 7B MOE model, light aug, multi-dialect
- [ ] Set augmentation: 20% shuffle, 50% filter
- [ ] Prepare multi-dialect dataset
- [ ] Configure MOMQ parameters
- [ ] Enable dialect router
- [ ] Start training
- [ ] Monitor expert utilization
- [ ] Evaluate on dev set (all dialects)
- [ ] Save model and scores

### Phase 9: Train Generator 4 (SQLG₄)
- [ ] Configure: 32B model, no aug, comprehensive
- [ ] Prepare no-augmentation dataset
- [ ] Set full fine-tuning (no LoRA)
- [ ] Start training (requires more GPU memory)
- [ ] Monitor training loss
- [ ] Evaluate on dev set
- [ ] Save model and scores

### Phase 10: Evaluation & Analysis
- [ ] Run all 4 models on same dev set
- [ ] Compare execution accuracy
- [ ] Analyze SQL diversity
- [ ] Check dialect-specific performance (if applicable)
- [ ] Measure inference speed for each model
- [ ] Document performance characteristics
- [ ] Create comparison table

### Phase 11: Ensemble Preparation (Future Work)
- [ ] Design ensemble voting strategy
- [ ] Implement candidate selection algorithm
- [ ] Create ensemble inference pipeline
- [ ] Test ensemble on dev set
- [ ] Compare ensemble vs individual models
- [ ] Optimize ensemble weights

### Phase 12: Production Deployment
- [ ] Select best individual model or ensemble
- [ ] Optimize inference speed
- [ ] Set up model serving infrastructure
- [ ] Create API endpoints
- [ ] Implement error handling
- [ ] Set up monitoring and logging
- [ ] Create user documentation
- [ ] Deploy to production

---

## Conclusion

This guide covers the complete XiYan-SQL training strategy implementation:

✅ **Multi-Task Training**: NL2SQL, Self-Refine, Candidate Selection
✅ **Data Augmentation**: 4 techniques for format diversity
✅ **Multi-Format**: Dialects, languages, M-Schema
✅ **Multi-Generator**: 4 diverse models with different strategies
✅ **MOMQ MOE**: Efficient multi-dialect architecture
✅ **Practical Examples**: Step-by-step training guides
✅ **Production Checklist**: Complete implementation tracker

### Key Takeaways

1. **This repository trains individual generators**, not the full ensemble system
2. **Diversity is created through**: augmentation, architecture, tasks, formats
3. **Multi-task training** uses different prompt templates (fully implemented)
4. **Evidence is input context**, not a separate generation task
5. **MOMQ MOE** is ideal for multi-dialect scenarios
6. **4 generators** balance diversity, quality, and specialization

### Next Steps

1. **Train baseline model** to understand the framework
2. **Experiment with augmentation** to see impact
3. **Create self-refine data** from baseline predictions
4. **Train all 4 generators** following the guide
5. **Implement ensemble inference** (separate project)

### Resources

- **Code**: `XiYan-SQLTraining/` (this repository)
- **Paper**: XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL
- **Models**: Qwen2.5-Coder series (7B, 14B, 32B)
- **Benchmarks**: BIRD, Spider, BIRD-CRITIC

---

**Version**: 1.0
**Last Updated**: 2026-01-10
**Author**: XiYan-SQL Team / Documentation
**License**: Same as XiYan-SQLTraining repository
