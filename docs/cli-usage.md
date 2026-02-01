# CLI Usage

This document describes how to use the Cogito AI command-line interface.

## Starting the CLI

```bash
python cogito.py
```

On startup, Cogito will:

1. Display the ASCII art logo and license information
2. Start the required Docker containers (Qdrant and PostgreSQL) if available
3. Build the research agent
4. Enter the interactive prompt

## Interactive Session

### Basic Usage

Type your philosophical question at the prompt:

```
▸ You: What is the difference between rationalism and empiricism?
```

Cogito will process your question and display:

1. The AI response with citations
2. Metadata showing:
   - Time taken
   - Research level (0-3)
   - Resources found

### Example Session

```
▸ You: Explain Plato's theory of Forms

[Thinking indicator: "Calibrating my research..."]
[Thinking indicator: "Planning research..."]
[Thinking indicator: "Gathering resources..."]

╭─────────────────────────────────────────────────────────────────────────╮
│ Plato's Theory of Forms (also called the Theory of Ideas) is one of    │
│ the most influential philosophical concepts in Western philosophy...    │
│                                                                         │
│ As stated in "The Republic," Plato argues that Forms are eternal,      │
│ unchanging, and perfect...                                              │
│                                                                         │
│ [Full response with citations]                                          │
╰─────────────────────────────────────────────────────────────────────────╯

Time: 12.34s - Research Level (0-3): ResearchEffort.DEEP
Resources Found:
    - Project Gutenberg: The Republic by Plato
    - Project Gutenberg: Phaedo by Plato
```

### Exiting

To exit the CLI, type:

```
▸ You: exit
```

or

```
▸ You: quit
```

The session will end, and conversation logs will be saved.

## Research Levels

Cogito automatically determines the appropriate research depth:

| Level | Name | Description | Iterations |
|-------|------|-------------|------------|
| 0 | NONE | No research needed (casual conversation) | N/A |
| 1 | SIMPLE | Basic research for straightforward questions | Up to 5 |
| 2 | DEEP | Deep research for complex, multifaceted questions | Up to 8 |

### When Each Level is Used

**NONE (0)**:
- Greetings and casual conversation
- Explicit requests for no research
- Non-philosophical topics

**SIMPLE (1)**:
- Single-concept questions
- Straightforward definitions
- Basic philosophical queries

**DEEP (2)**:
- Multi-philosopher comparisons
- Complex theoretical questions
- Topics requiring synthesis of multiple sources

### Influencing Research Level

You can guide the research level in your prompt:

```
▸ You: Without doing any research, what do you know about Nietzsche?
```

```
▸ You: Please do thorough research on the relationship between Heidegger and Husserl.
```

## Conversation Context

Cogito maintains conversation history during a session. You can:

- Ask follow-up questions
- Reference previous topics
- Build on earlier answers

```
▸ You: What is utilitarianism?
[Response about utilitarianism]

▸ You: How does this compare to Kantian ethics?
[Response comparing utilitarianism to Kantian ethics, using context from previous answer]
```

## Data Sources

Cogito searches two primary sources:

### Stanford Encyclopedia of Philosophy (SEP)

- Authoritative academic encyclopedia
- Peer-reviewed articles
- Comprehensive philosophical coverage

### Project Gutenberg

- 1000+ philosophy texts
- Classical philosophical works
- Full-text semantic search via vector database

## Output Format

### Response Panel

The main response is displayed in a Rich panel with:

- Synthesized answer
- Quoted evidence from sources
- Inline citations

### Metadata Panel

Below the response, metadata is displayed:

```
Time: 12.34s - Research Level (0-3): ResearchEffort.DEEP
Resources Found:
    - Project Gutenberg: The Republic by Plato
    - Project Gutenberg: Nicomachean Ethics by Aristotle
```

## Conversation Logs

When you exit, conversation logs are saved automatically. Logs are stored as JSON files containing:

- All user messages
- All AI responses
- Message timestamps

## Troubleshooting

### Docker Containers Not Starting

If you see errors about database connections:

1. Ensure Docker is running
2. Start containers manually:
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
     -e QDRANT__SERVICE__API_KEY=your-key \
     crazywillbear/cogito-vectors:latest
   
   docker run -d -p 5432:5432 \
     -e POSTGRES_USER=user \
     -e POSTGRES_PASSWORD=pass \
     -e POSTGRES_DB=cogito \
     crazywillbear/cogito-filters-postgres:latest
   ```

### Slow Responses

Deep research queries may take 20-30 seconds. This is normal as the agent:

- Plans research strategy
- Executes multiple queries
- Synthesizes results

For faster responses, rephrase to suggest simpler research:

```
▸ You: Briefly, what is Stoicism?
```

### API Key Errors

Ensure your `.env` file contains valid API keys:

```bash
GROQ_API_KEY=your-groq-key
OPENAI_API_KEY=your-openai-key  # Required for embeddings
```
