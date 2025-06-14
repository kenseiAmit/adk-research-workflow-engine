from typing import Any, AsyncGenerator, Dict

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai.types import GenerateContentConfig, ThinkingConfig

from utils.adk_runtime import event_text, parse_agent_json, run_adk_agent
from utils.config import config

# Critic stops the LoopAgent once an answer reaches this score.
QUALITY_THRESHOLD = config.quality_threshold


class ResearcherAgent(LlmAgent):
    """
    Generator agent: Answers research questions or refines based on feedback.

    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        """Initialize the researcher agent.

        Args:
            model: Gemini model name
        """
        instruction = """You are a research assistant that answers questions accurately and thoroughly.

Your role:
1. Answer the research question clearly and comprehensively
2. If critic feedback is present below, rewrite your previous answer and resolve EVERY weakness raised
3. Cite reasoning and provide evidence where possible
4. Include 3-5 key points with supporting details
5. Mention relevant sources or areas of research

Critic feedback from the previous iteration (empty on the first iteration):
{criticism?}

Output format (JSON):
{
  "answer": "Your comprehensive answer here with evidence and reasoning",
  "key_points": ["point1 with evidence", "point2 with evidence", "point3 with evidence"],
  "sources_mentioned": ["source1", "source2", "source3"],
  "confidence": "high/medium/low",
  "iteration_notes": "What you improved this iteration (if applicable)"
}

Focus on accuracy, clarity, and continuous improvement. Each iteration should show measurable progress."""

        # Initialize ADK LlmAgent; output_key publishes the draft into session state
        super().__init__(
            name="researcher",
            model=model,
            instruction=instruction,
            output_key="draft_answer",
            generate_content_config=GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_budget=0)
            )
        )


class ResearchCriticAgent(LlmAgent):
    """
    Validator agent: Evaluates answer quality and provides feedback.

    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        """Initialize the critic agent.

        Args:
            model: Gemini model name
        """
        instruction = f"""You are a demanding research quality critic that evaluates answers.

Your role:
1. Evaluate the answer for accuracy, completeness, and clarity
2. Assign a quality level: excellent, good, needs_improvement, or poor
3. Provide specific, actionable feedback for improvement
4. Decide if answer is good enough to stop, or needs another iteration
5. Calculate a quality score (0-1) based on your assessment

Apply these evidentiary requirements strictly. An answer cannot score at or above
{QUALITY_THRESHOLD:.2f} if ANY of the following apply:
- It cites no named sources, researchers, publications or concrete works
- It states claims generically without specific evidence, mechanisms or examples
- It ignores significant counterarguments, caveats or competing interpretations
- It restates the question's premise instead of critically examining it

Quality criteria:
- Excellent (0.90-1.00): Thorough, accurate, well-structured, comprehensive evidence,
  named sources, and explicit treatment of counterarguments
- Good (0.80-0.89): Accurate and complete, all key points covered with specific evidence
- Needs Improvement (0.50-0.79): Missing key points, generic claims, or unnamed sources
- Poor (0.00-0.49): Incomplete, unclear, or potentially inaccurate

Be rigorous: a first draft that is fluent but unsourced is "needs_improvement", not "good".

Output format (JSON):
{{
  "quality": "excellent/good/needs_improvement/poor",
  "quality_score": 0.85,
  "feedback": "Specific feedback for improvement",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "should_stop": true/false,
  "reasoning": "Why to stop or continue"
}}

Set should_stop=true ONLY if quality_score >= {QUALITY_THRESHOLD:.2f}.
Otherwise set should_stop=false to trigger another iteration."""

        # Initialize ADK LlmAgent; output_key publishes the critique into session state
        super().__init__(
            name="critic",
            model=model,
            instruction=instruction,
            output_key="criticism",
            generate_content_config=GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=768,
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_budget=0)
            )
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Evaluate, then signal the LoopAgent to stop once quality is sufficient.

        Emitting an escalate event is ADK's termination signal, so the LoopAgent -
        not application code - decides when refinement ends.
        """
        async for event in super()._run_async_impl(ctx):
            yield event

        evaluation = parse_agent_json(
            ctx.session.state.get("criticism"), {"quality_score": 0.0}
        )

        try:
            quality_score = float(evaluation.get("quality_score", 0.0))
        except (TypeError, ValueError):
            quality_score = 0.0

        if quality_score >= QUALITY_THRESHOLD:
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True),
            )


def create_research_loop_agent(model: str = "gemini-2.0-flash",
                                max_iterations: int = 3) -> LoopAgent:
    """
    Creates a LoopAgent for iterative research refinement.

    Args:
        model: Gemini model to use
        max_iterations: Maximum loop iterations (safety limit)

    Returns:
        LoopAgent configured for research refinement
    """
    # Create the generator and validator agents
    researcher = ResearcherAgent(model=model)
    critic = ResearchCriticAgent(model=model)

    # Compose agents into a LoopAgent that runs researcher -> critic iteratively
    refinement_loop = LoopAgent(
        name="research_refinement_loop",
        sub_agents=[researcher, critic],
        max_iterations=max_iterations
    )

    return refinement_loop


async def execute_research_loop(
    client: Any,
    query: str,
    max_iterations: int = 3,
    model: str = "gemini-2.0-flash"
) -> Dict[str, Any]:
    """
    Execute iterative research refinement using ADK LoopAgent.

    The LoopAgent owns iteration and termination: each cycle runs
    researcher -> critic, and the critic escalates once quality clears the
    threshold. This function only streams and reports the resulting events.

    Args:
        client: Unused; ADK builds its own model client from the environment
        query: Research question
        max_iterations: Maximum loop iterations
        model: Gemini model name

    Returns:
        Dictionary with final answer and iteration history
    """
    print(f"\n Research Loop: {query[:60]}...")
    print(f"   Max Iterations: {max_iterations}")

    # Create LoopAgent
    loop_agent = create_research_loop_agent(model=model, max_iterations=max_iterations)

    print(f"   Created LoopAgent: {loop_agent.name}")
    print(f"   Type: {type(loop_agent).__name__}")
    print(f"   Sub-agents: {len(loop_agent.sub_agents)} (researcher + critic)")
    print(f"   Quality threshold for early stop: {QUALITY_THRESHOLD:.2f}")

    print(f"\n   🔄 ADK Runner executing LoopAgent (framework-driven iteration)...")

    iteration_history: list = []
    pending_answer: Dict[str, Any] = {}

    def handle_event(event) -> None:
        """Report each Researcher -> Critic cycle as the LoopAgent emits it."""
        nonlocal pending_answer

        text = event_text(event)
        if not text:
            return

        if event.author == "researcher":
            pending_answer = parse_agent_json(
                text,
                {"answer": text, "key_points": [], "confidence": "low"},
            )
            print(f"\n   Iteration {len(iteration_history) + 1}/{max_iterations}")
            print(f"      → researcher generating answer...")
            print(
                f"      ✓ Answer generated "
                f"(confidence: {pending_answer.get('confidence', 'unknown')})"
            )
            notes = pending_answer.get("iteration_notes")
            if notes and iteration_history:
                print(f"      ↳ Addressed feedback: {str(notes)[:70]}...")

        elif event.author == "critic":
            evaluation = parse_agent_json(
                text,
                {"quality": "needs_improvement", "quality_score": 0.5,
                 "feedback": "Unable to parse evaluation", "should_stop": False},
            )
            try:
                quality_score = float(evaluation.get("quality_score", 0.0))
            except (TypeError, ValueError):
                quality_score = 0.0

            stopping = quality_score >= QUALITY_THRESHOLD

            print(f"      → critic evaluating quality...")
            print(
                f"      ✓ Quality: {evaluation.get('quality', 'unknown')} "
                f"(score: {quality_score:.2f})"
            )

            iteration_history.append({
                "iteration": len(iteration_history) + 1,
                "answer": pending_answer,
                "evaluation": evaluation,
                "should_stop": stopping,
            })

            if stopping:
                print(f"      Quality threshold met - LoopAgent escalated to stop")
            else:
                print(f"      Quality below threshold - Continue refining...")
                print(
                    f"      Feedback: "
                    f"{str(evaluation.get('feedback', 'No feedback'))[:80]}..."
                )

    result = await run_adk_agent(loop_agent, query, on_event=handle_event)

    final_answer = parse_agent_json(result["state"].get("draft_answer"), {})
    if not final_answer and iteration_history:
        final_answer = iteration_history[-1]["answer"]

    if iteration_history and not iteration_history[-1]["should_stop"]:
        print(f"\n   Max iterations reached - Returning best attempt")

    print(f"   LoopAgent execution completed ({len(iteration_history)} iterations)")

    return {
        'query': query,
        'final_answer': final_answer,
        'iterations_run': len(iteration_history),
        'iteration_history': iteration_history,
        'loop_agent': loop_agent,  # Include the actual LoopAgent object
        'pattern': 'ADK LoopAgent',
        'execution_mode': 'adk_runner'
    }
