# Phase 3: Mood Context Integration - Developer Guide

## Overview

Phase 3 links recent mood patterns to memory retrieval, enabling the bot to understand emotional context and provide relevant support when users are struggling with specific issues.

**Key Insight**: When a user has a negative mood trend related to "work stress," the bot retrieves and prioritizes work-related memories to show empathy and contextual understanding.

---

## Mood Clustering

Mood topics are automatically classified into 5 semantic clusters:

| Cluster               | Topics                                                   | Example                                  |
| --------------------- | -------------------------------------------------------- | ---------------------------------------- |
| `work_stress`         | Work, Boss, Office, Deadline, Meeting, Career, Promotion | "I had a tough meeting with my boss"     |
| `health_concerns`     | Health, Sick, Tired, Pain, Sleep, Gym, Workout           | "I've been feeling exhausted lately"     |
| `relationship_issues` | Love, Date, Breakup, Partner, Crush, Family, Parents     | "I had a fight with my partner"          |
| `financial_worry`     | Money, Expensive, Broke, Rent, Bill, Cost, Budget        | "I'm worried about rent this month"      |
| `future_anxiety`      | Hope, Dream, Goal, Plan, Worried, Scared, Nervous        | "I'm nervous about my career transition" |
| `general`             | General                                                  | Default fallback                         |

---

## Core Functions

### get_recent_mood_context(user_id, days=7)

Analyzes mood patterns over the past N days.

**Returns:**

```python
{
    'dominant_emotion': 'Heavy',           # Overall emotional state
    'average_score': -0.22,               # Avg sentiment (-1 to 1)
    'trend': 'declining',                 # improving | declining | stable
    'dominant_clusters': ['work_stress'], # Top clusters by frequency
    'is_negative_trend': True,            # Last 3 entries all negative?
    'entry_count': 8,                     # Total mood entries
    'raw_topics': {                       # Detailed breakdown
        'Work': {
            'count': 5,
            'avg_score': -0.35,
            'cluster': 'work_stress'
        }
    }
}
```

**Example:**

```python
context = memory.get_recent_mood_context("user123", days=7)
if context['is_negative_trend']:
    print(f"User is struggling with {context['dominant_clusters'][0]}")
    # Output: "User is struggling with work_stress"
```

---

### get_memories_by_mood_context(user_id, top_k=5, similarity_threshold=0.2)

Retrieves memories grouped by mood cluster relevance.

**Returns:**

```python
{
    'mood_context': {...},  # From get_recent_mood_context()
    'related_memories': [
        {
            'cluster': 'work_stress',
            'memories': [mem1, mem2, ...],
            'relevance_explanation': 'Related to work stress (5 recent mentions)'
        },
        {
            'cluster': 'health_concerns',
            'memories': [mem3, mem4],
            'relevance_explanation': 'Related to health concerns (2 recent mentions)'
        }
    ]
}
```

**Example:**

```python
result = memory.get_memories_by_mood_context("user123", top_k=3)

for cluster_group in result['related_memories']:
    print(f"\n{cluster_group['cluster'].title()}:")
    print(f"  {cluster_group['relevance_explanation']}")
    for mem in cluster_group['memories']:
        print(f"  - {mem['key']}: {mem['value']}")

# Output:
# Work Stress:
#   Related to work stress (5 recent mentions)
#   - Job: Senior Engineer
#   - Deadline: Q4 shipping date
#
# Health Concerns:
#   Related to health concerns (2 recent mentions)
#   - Health: Regular gym routine
```

---

### create_mood_aware_context(user_id, current_input, use_mood_prioritization=True)

Creates a rich memory context for LLM prompts that combines semantic search with mood awareness.

**Returns:** Formatted string for prompt injection

**Example:**

```python
context = memory.create_mood_aware_context(
    "user123",
    "I'm stressed about work",
    use_mood_prioritization=True
)

# Returns (if negative mood trend detected):
# "[Mood: Heavy | Trend: declining]"
# "Job: Senior Engineer; Deadline: Q4; Manager: John (demanding but fair); ..."
#
# Returns (if mood stable):
# "Senior Engineer; Q4 Deadline; Team Lead; ..."
```

**Logic:**

1. Get recent mood context
2. If negative trend detected:
   - Retrieve 3 mood-related memories from dominant clusters
   - Also get 2 semantic memories from current input
   - Combine and deduplicate (priority: mood memories first)
   - Annotate with mood information
3. Otherwise: Standard semantic search

---

## Integration in WarmthBot

In `generate_reply()`:

```python
# Get context with mood awareness
try:
    facts_raw = self.memory.create_mood_aware_context(
        self.user_id,
        user_input,
        use_mood_prioritization=True
    )
except Exception as e:
    logger.error(f"Mood context failed: {e}")
    facts_raw = self.memory.get_memory_context(self.user_id)  # Fallback

# Use in system prompt
system_prompt = f"User Facts: {facts_raw}"
```

**Result**: When user is struggling, bot has access to relevant context:

- User message: "I'm having a really tough week"
- Detected mood: Heavy (trend: declining)
- Retrieved memories: Job, Deadline, Manager (all work-related)
- Bot can provide empathetic, contextual response

---

## Mood Analysis Examples

### Example 1: Positive Trend

```
Recent moods: 0.1, 0.2, 0.3 (trending up)
Trend: improving
Action: Use standard semantic search (no intervention needed)
```

### Example 2: Negative Trend

```
Recent moods: -0.3, -0.4, -0.35 (all negative)
Trend: declining
Clusters: work_stress
Action: Prioritize work-related memories for empathy
Bot response: "I see work has been heavy lately. You're the senior engineer..."
```

### Example 3: Stable Mixed Mood

```
Recent moods: 0.1, -0.1, 0.0 (fluctuating)
Trend: stable
Action: Use standard semantic search
```

---

## Mood Scoring Thresholds

| Score         | Emotion | Interpretation |
| ------------- | ------- | -------------- |
| > 0.3         | Radiant | Very positive  |
| 0.05 to 0.3   | Good    | Positive       |
| -0.05 to 0.05 | Neutral | Balanced       |
| -0.3 to -0.05 | Low     | Negative       |
| < -0.3        | Heavy   | Very negative  |

**Negative Trend Detection**: All last 3 scores ≤ -0.1

---

## Tuning Mood Context

### Adjust trend sensitivity:

In `get_recent_mood_context()`:

```python
threshold = 0.1  # Change to detect smaller trends
if second_half_avg > first_half_avg + threshold:
    trend = 'improving'
```

### Change negative threshold:

```python
is_negative_trend = all(score <= -0.15 for score in recent_scores)  # More strict
```

### Adjust memory prioritization:

In `create_mood_aware_context()`:

```python
combined = mood_memories[:4] + semantic_memories[:1]  # More mood-focused
```

---

## Testing

Run comprehensive mood context tests:

```bash
python test_mood_context.py
```

Tests verify:

- ✓ Mood cluster classification
- ✓ Trend detection (improving/declining/stable)
- ✓ Mood context analysis
- ✓ Mood-aware memory retrieval
- ✓ Context creation with mood prioritization

---

## Performance

| Operation                      | Time  | Notes                       |
| ------------------------------ | ----- | --------------------------- |
| get_recent_mood_context()      | ~5ms  | Simple aggregation          |
| get_memories_by_mood_context() | ~50ms | Semantic search per cluster |
| create_mood_aware_context()    | ~60ms | Combined operations         |

---

## Use Cases

### 1. Supportive Response During Crisis

```
User: "I've been really down about work..."
Context: [Mood: Heavy | Trend: declining]
         "Job: Senior Engineer; Stress: Multiple deadlines; ..."
Response: "I see you've been under a lot of work pressure lately.
           You've handled tough deadlines before though..."
```

### 2. Proactive Check-in

```
Detected: User has declining mood trend + work_stress cluster
Action: Bot proactively asks about work stressors
Response: "I've noticed work's been weighing on you this week.
           Want to talk about what's going on?"
```

### 3. Positive Reinforcement

```
Detected: User trending positive + work_stress diminishing
Action: Acknowledge improvement
Response: "Things seem lighter this week! That work deadline
          passed and you handled it well."
```

---

## Architecture Diagram

```
User Input
    ↓
generate_reply()
    ↓
create_mood_aware_context()
    ├─→ get_recent_mood_context()
    │   └─→ Analyze last 7 days of mood logs
    │       ├─→ Classify topics → clusters
    │       ├─→ Detect trend
    │       └─→ Identify negative pattern
    │
    ├─→ If negative trend:
    │   ├─→ get_memories_by_mood_context()
    │   │   └─→ For each dominant cluster:
    │   │       └─→ get_relevant_memories_semantic()
    │   │           └─→ Return cluster-specific memories
    │   │
    │   └─→ Combine: 60% mood + 40% semantic
    │
    └─→ Else:
        └─→ Standard semantic search (40% importance, 60% similarity)

Result → Format for LLM → Include in system prompt
```

---

## Future Enhancements

### Phase 4: Optimization

- Cache mood context (5min TTL)
- Batch update mood patterns table
- Tune similarity thresholds per cluster

### Beyond Phase 4

- Multi-dimensional mood modeling (tired, stressed, anxious simultaneously)
- Mood recovery recommendations ("You recovered well from X before")
- Mood trigger identification ("Deadlines trigger your stress")
- Mood-based conversation pivoting
