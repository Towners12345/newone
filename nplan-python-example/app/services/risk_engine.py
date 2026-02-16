"""
Risk Engine — Graph-based schedule analysis using NetworkX.

INTERVIEW TALKING POINT:
This uses NetworkX, Python's standard graph library, to analyse schedule
dependencies — the same kind of graph data nPlan works with (10,000+ node
directed acyclic graphs). Their GNN models operate on these same graph
structures, just with deep learning on top.

"At nPlan, I'd be working with schedule graphs that have thousands of nodes
 and complex dependency relationships. I've built this kind of analysis in
 a simpler form using NetworkX, and I understand the graph data structures
 that your GNN models operate on. The step from NetworkX graph analysis to
 understanding GNN inputs is natural."
"""

import networkx as nx
from typing import Optional
from app.models import (
    ScheduleInput,
    ScheduleActivity,
    CriticalPathInfo,
    RiskLevel,
)


class RiskEngine:
    """
    Deterministic schedule risk analysis using graph algorithms.

    This handles the non-LLM analysis — critical path calculation,
    bottleneck detection, dependency analysis. At nPlan, this would
    be augmented by their GNN models for probabilistic forecasting.

    Think of this as the "post-processing pipeline" equivalent of your
    ATP unit-corrections and completeness-audit, but for schedules.
    """

    def __init__(self):
        self.graph: Optional[nx.DiGraph] = None
        self.activities: dict[str, ScheduleActivity] = {}

    def build_graph(self, schedule: ScheduleInput) -> nx.DiGraph:
        """
        Convert schedule activities into a directed graph.

        Each activity = a node. Each dependency = a directed edge.
        This is the fundamental data structure nPlan's entire platform
        operates on. Their 750k historical schedules are all graphs.
        """
        G = nx.DiGraph()

        # Index activities by ID for fast lookup
        self.activities = {a.id: a for a in schedule.activities}

        # Add nodes with attributes
        for activity in schedule.activities:
            G.add_node(
                activity.id,
                name=activity.name,
                duration=activity.duration_days,
                trade=activity.trade or "general",
                status=activity.status.value,
            )

        # Add edges (dependency relationships)
        for activity in schedule.activities:
            for pred_id in activity.predecessors:
                if pred_id in self.activities:
                    G.add_edge(pred_id, activity.id)

        self.graph = G
        return G

    def find_critical_path(self) -> CriticalPathInfo:
        """
        Calculate the critical path using longest-path algorithm.

        The critical path is the sequence of dependent activities that
        determines the minimum project duration. Any delay on the critical
        path delays the entire project.

        nPlan's AI goes further — predicting WHICH activities are likely
        to cause delays based on historical patterns.
        """
        if not self.graph or len(self.graph.nodes) == 0:
            return CriticalPathInfo(
                path_activities=[],
                total_duration_days=0,
                path_risk_level=RiskLevel.LOW,
            )

        # Find source nodes (no predecessors) and sink nodes (no successors)
        sources = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        sinks = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]

        if not sources or not sinks:
            # Handle edge case: schedule has no clear start/end
            all_ids = list(self.graph.nodes)
            return CriticalPathInfo(
                path_activities=all_ids,
                total_duration_days=sum(
                    self.activities[a].duration_days
                    for a in all_ids
                    if a in self.activities
                ),
                path_risk_level=RiskLevel.MEDIUM,
            )

        # Use longest path through the DAG (weighted by duration)
        # This IS the critical path in project scheduling
        longest_path = []
        longest_duration = 0

        for source in sources:
            for sink in sinks:
                try:
                    # Find all paths and pick the longest by total duration
                    for path in nx.all_simple_paths(self.graph, source, sink):
                        duration = sum(
                            self.activities[node].duration_days
                            for node in path
                            if node in self.activities
                        )
                        if duration > longest_duration:
                            longest_duration = duration
                            longest_path = path
                except nx.NetworkXNoPath:
                    continue

        # Identify the bottleneck (longest single activity on critical path)
        bottleneck = None
        max_duration = 0
        for activity_id in longest_path:
            if activity_id in self.activities:
                d = self.activities[activity_id].duration_days
                if d > max_duration:
                    max_duration = d
                    bottleneck = activity_id

        # Risk level based on path characteristics
        risk_level = self._assess_path_risk(longest_path)

        return CriticalPathInfo(
            path_activities=longest_path,
            total_duration_days=longest_duration,
            bottleneck_activity=bottleneck,
            path_risk_level=risk_level,
        )

    def find_high_risk_activities(self) -> list[str]:
        """
        Identify activities that pose the highest risk to the schedule.

        Uses graph metrics — activities with many dependents (high out-degree)
        and activities on the critical path are highest risk.

        At nPlan, this is done with trained ML models. Here we use simpler
        heuristics, but the concept is identical.
        """
        if not self.graph:
            return []

        risk_scores: dict[str, float] = {}

        for node in self.graph.nodes:
            score = 0.0

            # Activities with many successors are higher risk
            # (delays cascade to more downstream activities)
            out_degree = self.graph.out_degree(node)
            score += out_degree * 0.3

            # Activities with long durations have more exposure to delays
            duration = self.activities.get(node)
            if duration:
                score += min(duration.duration_days / 30, 2.0) * 0.4

            # Activities that are bottlenecks (high betweenness centrality)
            # This is a real graph metric — how often a node sits on
            # shortest paths between other nodes
            # (Skipped for very large graphs for performance)
            if len(self.graph.nodes) < 500:
                try:
                    centrality = nx.betweenness_centrality(self.graph)
                    score += centrality.get(node, 0) * 0.3
                except Exception:
                    pass

            risk_scores[node] = score

        # Return top 20% riskiest activities
        sorted_activities = sorted(
            risk_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_count = max(1, len(sorted_activities) // 5)
        return [aid for aid, _ in sorted_activities[:top_count]]

    def validate_schedule_integrity(self) -> list[str]:
        """
        Check schedule for structural issues.

        This is a simplified version of nPlan's Schedule Integrity Checker.
        Same concept as your ATP completeness audit — checking the output
        for problems the AI might have missed.
        """
        issues = []

        if not self.graph:
            return ["No schedule graph built"]

        # Check for cycles (invalid in a schedule — creates impossible ordering)
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            issues.append(
                f"Schedule contains {len(cycles)} circular dependency "
                f"cycle(s) — this makes the schedule impossible to execute"
            )

        # Check for disconnected components (isolated activity groups)
        undirected = self.graph.to_undirected()
        components = list(nx.connected_components(undirected))
        if len(components) > 1:
            issues.append(
                f"Schedule has {len(components)} disconnected groups — "
                f"some activities have no dependency relationship to others"
            )

        # Check for activities with zero duration (potential data issues)
        zero_duration = [
            a.id for a in self.activities.values()
            if a.duration_days == 0
        ]
        if zero_duration:
            issues.append(
                f"{len(zero_duration)} activities have zero duration — "
                f"verify these are milestones, not missing estimates"
            )

        # Check for dangling dependencies
        for node in self.graph.nodes:
            if (self.graph.in_degree(node) == 0 and
                    self.graph.out_degree(node) == 0):
                issues.append(
                    f"Activity '{node}' is completely isolated — "
                    f"no predecessors or successors"
                )

        return issues

    def _assess_path_risk(self, path: list[str]) -> RiskLevel:
        """Heuristic risk assessment for a critical path."""
        if not path:
            return RiskLevel.LOW

        total_duration = sum(
            self.activities[a].duration_days
            for a in path
            if a in self.activities
        )

        # Longer paths = higher risk (more activities that could delay)
        if total_duration > 365:
            return RiskLevel.CRITICAL
        elif total_duration > 180:
            return RiskLevel.HIGH
        elif total_duration > 90:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
