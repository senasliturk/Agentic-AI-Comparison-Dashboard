from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, List
from llm import LLMResult
import time

class LLM(Protocol):
    def generate(self, prompt: str) -> LLMResult: ...

@dataclass
class RunOutput:
    final_text: str
    trace: List[str]          # adım adım neler olduğu (debug için)
    llm_calls: int            # kaç kez model çağrıldı

class ReflectiveAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, task: str) -> RunOutput:
        trace = []
        calls = 0
        draft = self.llm.generate(f"Create an initial solution for: {task}").text
        trace.append("DRAFT:\n" + draft)

        critique = self.llm.generate(
            "Critique the solution. List weaknesses, missing parts, risks.\n\n"
            f"SOLUTION:\n{draft}"
        ).text
        trace.append("CRITIQUE:\n" + critique)

        revised = self.llm.generate(
            "Revise the solution using the critique. Produce a clearer, better final answer.\n\n"
            f"CRITIQUE:\n{critique}\n\nSOLUTION:\n{draft}"
        ).text
        trace.append("REVISED:\n" + revised)

        return RunOutput(final_text=revised, trace=trace, llm_calls=3)

class SimpleAgent:
    """Multi-agent'te rol bazlı ajan."""
    def __init__(self, llm: LLM, role: str):
        self.llm = llm
        self.role = role

    def run(self, content: str) -> str:
        return self.llm.generate(f"ROLE: {self.role}\nTASK:\n{content}").text

class MultiAgentSystem:
    def __init__(self, planner: SimpleAgent, critic: SimpleAgent, optimizer: SimpleAgent):
        self.planner = planner
        self.critic = critic
        self.optimizer = optimizer

    def run(self, task: str) -> RunOutput:
        trace = []
        plan = self.planner.run(f"Make a structured plan for: {task}")
        trace.append("PLAN:\n" + plan)

        critique = self.critic.run(f"Critique this plan:\n{plan}")
        trace.append("CRITIQUE:\n" + critique)

        improved = self.optimizer.run(
            f"Improve the plan using critique.\nPLAN:\n{plan}\n\nCRITIQUE:\n{critique}"
        )
        trace.append("IMPROVED:\n" + improved)

        return RunOutput(final_text=improved, trace=trace, llm_calls=3)

class TreeOfThoughtAgent:
    def __init__(self, llm: LLM, branches: int = 3):
        self.llm = llm
        self.branches = branches

    def run(self, task: str) -> RunOutput:
        trace = []
        ideas = []
        for i in range(1, self.branches + 1):
            idea = self.llm.generate(
                f"Generate strategy #{i} to solve:\n{task}\n"
                f"Make it meaningfully different from others."
            ).text
            ideas.append(idea)
            trace.append(f"BRANCH {i}:\n{idea}")

        judge = self.llm.generate(
            "Evaluate the strategies below and pick the best one. "
            "Explain why briefly, then output the selected strategy as FINAL.\n\n"
            + "\n\n---\n\n".join(ideas)
        ).text
        trace.append("JUDGE:\n" + judge)

        return RunOutput(final_text=judge, trace=trace, llm_calls=self.branches + 1)