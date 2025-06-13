import json
from typing import Any, Dict
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig, ThinkingConfig

from utils.adk_runtime import parse_agent_json, run_adk_agent

# DomainClassifierAgent lives in agents/router.py, which the orchestrator imports.


class FactCheckAgent(LlmAgent):
    """Validates research claims for accuracy (ADK LlmAgent)."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        """Initialize fact checker.

        Args:
            model: Gemini model name
        """
        instruction = """You are a research fact-checking specialist.

Your role:
1. Verify claims against your knowledge
2. Identify potentially inaccurate or unsupported statements
3. Assess overall credibility
4. Provide verification notes

Output format (JSON):
{
  "verified_claims": ["List of verified statements"],
  "questionable_claims": ["List of claims needing verification"],
  "credibility_score": 0.85,
  "verification_notes": "Brief notes on verification process",
  "sources_checked": ["List of knowledge sources validated against"],
  "recommendations": ["Suggestions for strengthening claims"]
}"""

        # Initialize ADK LlmAgent
        super().__init__(
            name="fact_checker",
            model=model,
            instruction=instruction,
            output_key="fact_check_result",
            generate_content_config=GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_budget=0)
            )
        )

    async def check(self, client: Any, answer: Dict[str, Any],
              sources: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fact-check a research answer by running this agent via the ADK Runner.

        Args:
            client: Unused; ADK builds its own model client from the environment
            answer: Research answer to check
            sources: Source information

        Returns:
            Fact-check results
        """
        prompt = f"""Please fact-check this research answer:

Answer: {answer.get('answer', 'No answer provided')}
Key Points: {json.dumps(answer.get('key_points', []))}
"""
        if sources:
            prompt += f"\nSources mentioned: {json.dumps(sources.get('aggregated_sources', {}).get('top_sources', [])[:5], indent=2)}"

        result = await run_adk_agent(self, prompt)

        fact_check = parse_agent_json(
            result["state"].get("fact_check_result"),
            {
                'verified_claims': [],
                'questionable_claims': [],
                'credibility_score': 0.5,
                'verification_notes': 'Fact-checking failed',
                'sources_checked': [],
                'recommendations': [],
            },
        )
        fact_check['_metadata'] = {'agent': self.name, 'execution': 'adk_runner'}
        return fact_check


class SynthesisAgent(LlmAgent):
    """Synthesizes research findings into coherent narrative (ADK LlmAgent)."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        """Initialize synthesis agent.

        Args:
            model: Gemini model name
        """
        instruction = """You are a research synthesis specialist.

Your role:
1. Combine findings from multiple sources
2. Create coherent narrative structure
3. Identify key themes and insights
4. Maintain academic tone
5. Provide actionable recommendations

Output format (JSON):
{
  "synthesis": "Comprehensive narrative combining all findings (3-5 paragraphs)",
  "key_insights": ["Main insight 1", "Main insight 2", "Main insight 3"],
  "themes": ["Major theme 1", "Major theme 2"],
  "recommendations": ["Actionable recommendation 1", "Actionable recommendation 2"],
  "coherence_score": 0.9,
  "executive_summary": "1-2 sentence summary"
}"""

        # Initialize ADK LlmAgent
        super().__init__(
            name="synthesizer",
            model=model,
            instruction=instruction,
            output_key="synthesis_result",
            generate_content_config=GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048,
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_budget=0)
            )
        )

    async def synthesize(self, client: Any, query: str, answer: Dict[str, Any],
                    fact_check: Dict[str, Any], sources: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize research findings by running this agent via the ADK Runner.

        Args:
            client: Unused; ADK builds its own model client from the environment
            query: Original question
            answer: Research answer
            fact_check: Fact-check results
            sources: Source information

        Returns:
            Synthesized findings
        """
        prompt = f"""Synthesize these research findings:

Question: {query}

Answer: {answer.get('answer', 'No answer')}

Key Points: {json.dumps(answer.get('key_points', []))}

Credibility: {fact_check.get('credibility_score', 0.5)}

Top Sources: {len(sources.get('aggregated_sources', {}).get('top_sources', []))} sources available"""

        result = await run_adk_agent(self, prompt)

        synthesis = parse_agent_json(
            result["state"].get("synthesis_result"),
            {
                'synthesis': answer.get('answer', 'Synthesis failed'),
                'key_insights': answer.get('key_points', []),
                'themes': [],
                'recommendations': [],
                'coherence_score': 0.5,
                'executive_summary': 'Synthesis failed',
            },
        )
        synthesis['_metadata'] = {'agent': self.name, 'execution': 'adk_runner'}
        return synthesis


class CitationAgent(LlmAgent):
    """Generates properly formatted citations (ADK LlmAgent)."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        """Initialize citation agent.

        Args:
            model: Gemini model name
        """
        instruction = """You are an academic citation specialist.

Your role:
1. Format citations in standard academic styles
2. Generate bibliography
3. Ensure proper attribution

Output format (JSON):
{
  "citations": [
    {
      "source_title": "Source title",
      "citation_apa": "APA formatted citation",
      "citation_mla": "MLA formatted citation",
      "citation_number": 1
    }
  ],
  "bibliography": "Complete bibliography in APA format",
  "total_citations": 10,
  "citation_style": "APA"
}"""

        # Initialize ADK LlmAgent
        super().__init__(
            name="citation_formatter",
            model=model,
            instruction=instruction,
            output_key="citation_result",
            generate_content_config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=3072,
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_budget=0)
            )
        )

    async def format_citations(self, client: Any, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Format citations by running this agent via the ADK Runner.

        Args:
            client: Unused; ADK builds its own model client from the environment
            sources: Source information

        Returns:
            Formatted citations
        """
        top_sources = sources.get('aggregated_sources', {}).get('top_sources', [])

        prompt = f"""Generate citations for these sources:

{json.dumps(top_sources[:10], indent=2)}"""

        result = await run_adk_agent(self, prompt)

        citations = parse_agent_json(
            result["state"].get("citation_result"),
            {
                'citations': [],
                'bibliography': '',
                'total_citations': 0,
                'citation_style': 'APA',
            },
        )
        citations['_metadata'] = {'agent': self.name, 'execution': 'adk_runner'}
        return citations
