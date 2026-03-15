"""
LangGraph onboarding agent.

Builds a personalized developer learning path for a repository and role.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set, TypedDict

from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from retrieval.vector_store import VectorStore
from config.settings import settings

logger = logging.getLogger(__name__)


class OnboardingState(TypedDict, total=False):
	repo: str
	role: str
	search_queries: List[str]
	query_hits: List[dict]
	retrieved_docs: List[Document]
	ranked_files: List[dict]
	learning_plan: str
	total_docs_retrieved: int
	status: str
	error: str


class OnboardingAgent:
	"""LangGraph agent that generates a role-aware codebase onboarding plan."""

	def __init__(self, vector_store: VectorStore, model: str = None):
		self.vector_store = vector_store
		model_name = model or settings.llm_model
		if not model_name.startswith("claude"):
			model_name = "claude-haiku-4-5-20251001"

		self.model_name = model_name
		self.llm = ChatAnthropic(
			model=self.model_name,
			anthropic_api_key=settings.anthropic_api_key,
			temperature=0,
		)
		self.graph = self._build_graph()

	def _build_graph(self):
		workflow = StateGraph(OnboardingState)

		workflow.add_node("analyze_role", self.analyze_role)
		workflow.add_node("query_codebase", self.query_codebase)
		workflow.add_node("rank_files", self.rank_files)
		workflow.add_node("generate_plan", self.generate_plan)

		workflow.add_edge(START, "analyze_role")
		workflow.add_edge("analyze_role", "query_codebase")
		workflow.add_edge("query_codebase", "rank_files")
		workflow.add_edge("rank_files", "generate_plan")
		workflow.add_edge("generate_plan", END)

		return workflow.compile()

	def analyze_role(self, state: OnboardingState) -> OnboardingState:
		"""Pick retrieval queries based on developer role."""
		role = (state.get("role") or "developer").strip().lower()
		repo = state.get("repo", "")

		role_templates: Dict[str, List[str]] = {
			"frontend": [
				"app entrypoint and routing",
				"ui components and pages",
				"state management and data fetching",
				"styling system and global css conventions",
				"frontend testing setup",
			],
			"backend": [
				"api entrypoint and route handlers",
				"service and business logic modules",
				"data models and persistence layer",
				"error handling and logging patterns",
				"backend tests and fixtures",
			],
			"fullstack": [
				"application architecture overview",
				"frontend to backend data flow",
				"api contracts and request lifecycle",
				"configuration and environment setup",
				"integration and end-to-end tests",
			],
			"devops": [
				"deployment configuration and scripts",
				"environment variables and settings",
				"ci cd workflows and automation",
				"infrastructure and observability",
				"runtime and scaling considerations",
			],
			"data": [
				"data ingestion pipelines",
				"vector store or database integrations",
				"chunking, retrieval, and ranking flow",
				"batch jobs and processing logic",
				"data validation and tests",
			],
			"qa": [
				"test suite structure and strategy",
				"unit and integration test helpers",
				"critical user flows",
				"error handling edge cases",
				"release quality checks",
			],
			"security": [
				"authentication and authorization",
				"secrets and environment handling",
				"input validation and sanitization",
				"dependency and supply chain controls",
				"security relevant tests and policies",
			],
		}

		selected = role_templates.get(role, role_templates["fullstack"])
		queries = [f"{repo}: {item}" if repo else item for item in selected]

		return {
			"search_queries": queries,
			"status": "role_analyzed",
		}

	def query_codebase(self, state: OnboardingState) -> OnboardingState:
		"""Run role-specific queries against ChromaDB through VectorStore."""
		repo = state.get("repo", "")
		queries = state.get("search_queries", [])

		all_docs: List[Document] = []
		query_hits: List[dict] = []

		for query in queries:
			docs = self.vector_store.similarity_search(
				query,
				k=settings.retrieval_top_k,
				filter={"repo": repo} if repo else None,
			)

			if not docs and repo:
				docs = self.vector_store.similarity_search(
					query,
					k=settings.retrieval_top_k,
				)

			query_hits.append({"query": query, "docs": docs})
			all_docs.extend(docs)

		# Deduplicate docs while preserving first appearance order.
		seen_keys: Set[str] = set()
		deduped_docs: List[Document] = []
		for doc in all_docs:
			meta = doc.metadata or {}
			key = f"{meta.get('path', '')}:{meta.get('sha', '')}:{meta.get('chunk_index', '')}"
			if key in seen_keys:
				continue
			seen_keys.add(key)
			deduped_docs.append(doc)

		return {
			"query_hits": query_hits,
			"retrieved_docs": deduped_docs,
			"total_docs_retrieved": len(all_docs),
			"status": "codebase_queried",
		}

	def rank_files(self, state: OnboardingState) -> OnboardingState:
		"""Aggregate retrieval hits into a relevance-ranked top file list."""
		query_hits = state.get("query_hits", [])

		scores = defaultdict(float)
		hit_counts = defaultdict(int)
		file_queries: Dict[str, Set[str]] = defaultdict(set)
		file_urls: Dict[str, str] = {}

		for hit in query_hits:
			query = hit.get("query", "")
			docs: List[Document] = hit.get("docs", [])

			for rank, doc in enumerate(docs):
				path = doc.metadata.get("path", "")
				if not path:
					continue

				positional_weight = max(0.2, 1.0 - (rank * 0.1))
				scores[path] += positional_weight
				hit_counts[path] += 1
				file_queries[path].add(query)
				file_urls[path] = doc.metadata.get("url", "")

		ranked = sorted(
			scores.keys(),
			key=lambda path: (scores[path], hit_counts[path]),
			reverse=True,
		)

		top_files = []
		for path in ranked[:8]:
			top_files.append(
				{
					"path": path,
					"url": file_urls.get(path, ""),
					"score": round(scores[path], 3),
					"matches": hit_counts[path],
					"matched_queries": sorted(file_queries[path]),
				}
			)

		return {
			"ranked_files": top_files,
			"status": "files_ranked",
		}

	def generate_plan(self, state: OnboardingState) -> OnboardingState:
		"""Use Claude to write a structured, role-specific onboarding plan."""
		role = state.get("role", "developer")
		repo = state.get("repo", "")
		ranked_files = state.get("ranked_files", [])

		if not ranked_files:
			fallback = (
				"No relevant files were found in the vector store for this role and repository. "
				"Ingest the repository first, then rerun onboarding."
			)
			return {
				"learning_plan": fallback,
				"status": "completed_with_no_files",
			}

		file_brief = "\n".join(
			[
				f"- {file_info['path']} (score={file_info['score']}, matches={file_info['matches']})"
				for file_info in ranked_files
			]
		)

		prompt = ChatPromptTemplate.from_messages(
			[
				(
					"system",
					"You are DevOracle's onboarding architect. "
					"Create concise, practical, and role-specific developer onboarding plans.",
				),
				(
					"human",
					"Repository: {repo}\n"
					"Role: {role}\n"
					"Top relevant files:\n{file_brief}\n\n"
					"Write a structured onboarding plan with the sections below:\n"
					"1) Goal and success criteria\n"
					"2) 3-day learning path (Day 1/2/3)\n"
					"3) File-by-file study guide (why each file matters)\n"
					"4) Hands-on tasks (at least 3)\n"
					"5) Common pitfalls and debugging tips\n"
					"6) Checkpoint questions to validate understanding\n"
					"Keep it practical and specific to the listed files.",
				),
			]
		)

		plan = (prompt | self.llm).invoke(
			{
				"repo": repo,
				"role": role,
				"file_brief": file_brief,
			}
		)

		return {
			"learning_plan": plan.content,
			"status": "completed",
		}

	def run(self, repo: str, role: str) -> dict:
		"""
		Generate a role-specific onboarding learning plan for a repository.

		Returns keys:
		  - learning_plan
		  - files_analyzed
		  - total_docs_retrieved
		  - status
		"""
		initial_state: OnboardingState = {
			"repo": repo,
			"role": role,
			"status": "started",
		}

		try:
			result = self.graph.invoke(initial_state)

			files_analyzed = [item.get("path", "") for item in result.get("ranked_files", [])]

			return {
				"learning_plan": result.get("learning_plan", ""),
				"files_analyzed": [path for path in files_analyzed if path],
				"total_docs_retrieved": result.get("total_docs_retrieved", 0),
				"status": result.get("status", "completed"),
			}
		except Exception as exc:
			logger.error("Onboarding agent failed", exc_info=True)
			return {
				"learning_plan": "",
				"files_analyzed": [],
				"total_docs_retrieved": 0,
				"status": f"failed: {exc}",
			}
