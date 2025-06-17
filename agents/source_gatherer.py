from typing import Any, Dict, Optional

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai.types import Content, GenerateContentConfig, ThinkingConfig

from utils.adk_runtime import parse_agent_json, run_adk_agent


def _trace_start(callback_context: CallbackContext) -> Optional[Content]:
    """Log fan-out start; all three searches print before any of them finishes."""
    print(f"      → {callback_context.agent_name} running...", flush=True)
    return None


def _trace_finish(callback_context: CallbackContext) -> Optional[Content]:
    """Log each search's completion with the count parsed from session state."""
    payload = parse_agent_json(
        callback_context.state.get(f"{callback_context.agent_name}_results"), {}
    )
    found = payload.get("total_found", len(payload.get("results", [])))
    print(f"      ✓ {callback_context.agent_name}: Found {found} sources", flush=True)
    return None


class WebSearchAgent(LlmAgent):
    """Simulates web search using LLM (ADK LlmAgent)."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        instruction = """You are a web search specialist that finds relevant online sources.

Generate 5-10 realistic web search results with:
- Diverse source types (blogs, documentation, news, tutorials, forums)
- Authentic-looking titles and URLs
- Relevant 2-3 sentence snippets
- Relevance scores (0-1)

Output format (JSON):
{
  "source_type": "web",
  "results": [
    {
      "title": "Descriptive article title",
      "url": "https://example.com/realistic-url",
      "snippet": "2-3 sentence preview that's relevant to the query",
      "relevance": 0.95,
      "source": "website name"
    }
  ],
  "total_found": 10,
  "search_time": 0.5
}"""

        # output_key publishes results into session state for the aggregator
        super().__init__(
            name="web_search",
            model=model,
            instruction=instruction,
            output_key="web_search_results",
            before_agent_callback=_trace_start,
            after_agent_callback=_trace_finish,
            generate_content_config=GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=3072,
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_budget=0)
            )
        )


class ArxivSearchAgent(LlmAgent):
    """Simulates arXiv academic paper search using LLM (ADK LlmAgent)."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        instruction = """You are an arXiv academic paper search specialist.

Generate 5-8 realistic arXiv papers with:
- Academic paper titles
- Realistic author names
- arXiv URLs (https://arxiv.org/abs/YYMM.NNNNN format)
- 3-5 sentence abstracts
- Publication dates (recent, within last 2-3 years)

Output format (JSON):
{
  "source_type": "arxiv",
  "results": [
    {
      "title": "Academic Paper Title: Subtitle",
      "authors": ["FirstName LastName", "FirstName LastName"],
      "url": "https://arxiv.org/abs/2401.12345",
      "abstract": "3-5 sentence academic abstract describing the research",
      "published": "2024-01-15",
      "relevance": 0.92
    }
  ],
  "total_found": 8,
  "search_time": 0.3
}"""

        super().__init__(
            name="arxiv_search",
            model=model,
            instruction=instruction,
            output_key="arxiv_search_results",
            before_agent_callback=_trace_start,
            after_agent_callback=_trace_finish,
            generate_content_config=GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=3072,
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_budget=0)
            )
        )


class ScholarSearchAgent(LlmAgent):
    """Simulates Google Scholar academic search using LLM (ADK LlmAgent)."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        instruction = """You are a Google Scholar search specialist.

Generate 5-8 realistic academic publications with:
- Academic titles (papers, theses, books)
- Author lists
- Publication venues (journals, conferences)
- Years and citation counts
- Brief descriptions

Output format (JSON):
{
  "source_type": "scholar",
  "results": [
    {
      "title": "Academic Publication Title",
      "authors": ["Author1", "Author2", "Author3"],
      "venue": "Journal of Computer Science / Conference Name",
      "year": 2024,
      "url": "https://scholar.google.com/citations?id=example",
      "snippet": "2-3 sentence description of the work",
      "citations": 45,
      "relevance": 0.88
    }
  ],
  "total_found": 8,
  "search_time": 0.4
}"""

        super().__init__(
            name="scholar_search",
            model=model,
            instruction=instruction,
            output_key="scholar_search_results",
            before_agent_callback=_trace_start,
            after_agent_callback=_trace_finish,
            generate_content_config=GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=3072,
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_budget=0)
            )
        )


class SourceAggregatorAgent(LlmAgent):
    """Aggregates and ranks sources from multiple search agents."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        """Initialize aggregator agent."""
        instruction = """You are a source aggregation specialist.

Your role:
1. Combine search results from multiple sources (web, arXiv, Google Scholar)
2. Remove duplicates
3. Rank by relevance
4. Provide summary statistics

Aggregate these results, gathered concurrently by the parallel search agents.

Web search results:
{web_search_results?}

arXiv search results:
{arxiv_search_results?}

Google Scholar search results:
{scholar_search_results?}

Output format (JSON):
{
  "total_sources": 30,
  "unique_sources": 25,
  "top_sources": [
    {
      "title": "Source title",
      "type": "web/arxiv/scholar",
      "url": "https://...",
      "relevance_score": 0.95,
      "snippet": "Brief description"
    }
  ],
  "sources_by_type": {
    "web": 10,
    "arxiv": 8,
    "scholar": 7
  },
  "aggregation_summary": "Brief summary of source quality and diversity"
}

Select the top 10-15 most relevant sources."""

        super().__init__(
            name="source_aggregator",
            model=model,
            instruction=instruction,
            output_key="aggregated_sources",
            generate_content_config=GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=4096,
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_budget=0)
            )
        )


def create_source_gathering_workflow(model: str = "gemini-2.0-flash") -> SequentialAgent:
    """
    Creates a SequentialAgent with ParallelAgent for source gathering.

    Args:
        model: Gemini model to use

    Returns:
        SequentialAgent configured for parallel source gathering
    """
    # Create specialized search agents
    web_search = WebSearchAgent(model=model)
    arxiv_search = ArxivSearchAgent(model=model)
    scholar_search = ScholarSearchAgent(model=model)
    aggregator = SourceAggregatorAgent(model=model)

    # Create a ParallelAgent that runs all three search agents concurrently
    # (fan-out part of the fan-out/fan-in pattern)
    parallel_searches = ParallelAgent(
        name="parallel_source_searches",
        sub_agents=[web_search, arxiv_search, scholar_search]
    )

    # Wrap with a SequentialAgent that runs the ParallelAgent (fan-out) then
    # the aggregator (fan-in) to combine all results
    source_gathering_workflow = SequentialAgent(
        name="source_gathering_workflow",
        sub_agents=[parallel_searches, aggregator]
    )

    return source_gathering_workflow


async def execute_source_gathering(
    client: Any,
    query: str,
    model: str = "gemini-2.0-flash"
) -> Dict[str, Any]:
    """
    Execute parallel source gathering using ADK ParallelAgent + SequentialAgent.

    The ADK Runner drives the workflow: the ParallelAgent fans out all three
    searches concurrently, and the SequentialAgent only runs the aggregator
    after every search has completed.

    Args:
        client: Unused; ADK builds its own model client from the environment
        query: Research query
        model: Gemini model name

    Returns:
        Dictionary with aggregated sources
    """
    print(f"\n Source Gathering: {query[:60]}...")
    print(f"   Pattern: ADK ParallelAgent + SequentialAgent")

    workflow = create_source_gathering_workflow(model=model)

    parallel_stage = workflow.sub_agents[0]  # ParallelAgent
    aggregator = workflow.sub_agents[1]      # AggregatorAgent

    print(f"   Created SequentialAgent: {workflow.name}")
    print(f"   Type: {type(workflow).__name__}")
    print(f"   Sub-agents: {len(workflow.sub_agents)}")
    print(f"   Stage 1 (ParallelAgent): {parallel_stage.name}")
    print(f"      → Type: {type(parallel_stage).__name__}")
    print(f"      → Sub-agents: {len(parallel_stage.sub_agents)} parallel searches")
    print(f"   Stage 2 (Aggregator): {aggregator.name}")

    print(f"\n   ADK Runner executing workflow (framework-driven stages)...")
    print(f"\n   Stage 1: ParallelAgent (fan-out - all searches start together)")

    result = await run_adk_agent(workflow, query)
    state = result["state"]

    print(f"\n   Stage 2: Aggregator (fan-in - runs after all searches complete)")
    print(f"      → {aggregator.name} aggregating results...")

    aggregated = parse_agent_json(
        state.get("aggregated_sources"),
        {
            'total_sources': 0,
            'unique_sources': 0,
            'top_sources': [],
            'sources_by_type': {'web': 0, 'arxiv': 0, 'scholar': 0},
            'aggregation_summary': 'Aggregation failed - JSON parse error',
        },
    )

    print(f"      ✓ Total: {aggregated.get('total_sources', 0)} sources")
    print(f"      ✓ Unique: {aggregated.get('unique_sources', 0)} sources")
    print(f"      ✓ Top sources: {len(aggregated.get('top_sources', []))}")
    print(f"   ✅ Workflow execution completed")

    raw_searches = [
        parse_agent_json(state.get(key), {})
        for key in ("web_search_results", "arxiv_search_results", "scholar_search_results")
    ]

    return {
        'query': query,
        'raw_searches': raw_searches,
        'aggregated_sources': aggregated,
        'workflow': workflow,  # Include the actual SequentialAgent object
        'parallel_agent': parallel_stage,  # Include the actual ParallelAgent object
        'pattern': 'ADK ParallelAgent + SequentialAgent',
        'execution_mode': 'adk_runner'
    }
