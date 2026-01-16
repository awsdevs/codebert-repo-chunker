"""
Graph analysis utilities for relationship graphs
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import community  # python-louvain
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GraphAnalyzer:
    """Analyze graph structures and relationships"""
    
    def analyze_graph(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Comprehensive graph analysis
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Analysis results
        """
        analysis = {
            'basic_stats': self.get_basic_stats(graph),
            'centrality': self.calculate_centrality(graph),
            'clustering': self.calculate_clustering(graph),
            'connectivity': self.analyze_connectivity(graph),
            'paths': self.analyze_paths(graph)
        }
        
        return analysis
    
    def get_basic_stats(self, graph: nx.Graph) -> Dict[str, Any]:
        """Get basic graph statistics"""
        return {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
            'is_connected': nx.is_connected(graph) if not graph.is_directed() else nx.is_weakly_connected(graph)
        }
    
    def calculate_centrality(self, graph: nx.Graph, top_n: int = 10) -> Dict[str, Any]:
        """Calculate various centrality measures"""
        centrality = {}
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(graph)
        centrality['degree'] = sorted(
            degree_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(graph)
            centrality['betweenness'] = sorted(
                betweenness.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
        except:
            centrality['betweenness'] = []
        
        # Closeness centrality
        try:
            closeness = nx.closeness_centrality(graph)
            centrality['closeness'] = sorted(
                closeness.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
        except:
            centrality['closeness'] = []
        
        # PageRank
        try:
            pagerank = nx.pagerank(graph, max_iter=100)
            centrality['pagerank'] = sorted(
                pagerank.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
        except:
            centrality['pagerank'] = []
        
        return centrality
    
    def calculate_clustering(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate clustering coefficients"""
        clustering = {}
        
        if not graph.is_directed():
            clustering['average_clustering'] = nx.average_clustering(graph)
            clustering['transitivity'] = nx.transitivity(graph)
        else:
            clustering['average_clustering'] = 0
            clustering['transitivity'] = 0
        
        return clustering
    
    def analyze_connectivity(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze graph connectivity"""
        connectivity = {}
        
        if graph.is_directed():
            connectivity['strongly_connected_components'] = nx.number_strongly_connected_components(graph)
            connectivity['weakly_connected_components'] = nx.number_weakly_connected_components(graph)
        else:
            connectivity['connected_components'] = nx.number_connected_components(graph)
            connectivity['node_connectivity'] = nx.node_connectivity(graph) if graph.number_of_nodes() < 1000 else -1
            connectivity['edge_connectivity'] = nx.edge_connectivity(graph) if graph.number_of_nodes() < 1000 else -1
        
        return connectivity
    
    def analyze_paths(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze shortest paths"""
        paths = {}
        
        # Average shortest path length (for connected graphs)
        if nx.is_connected(graph) if not graph.is_directed() else nx.is_strongly_connected(graph):
            paths['avg_shortest_path'] = nx.average_shortest_path_length(graph)
        else:
            paths['avg_shortest_path'] = -1
        
        # Diameter (longest shortest path)
        if graph.number_of_nodes() < 1000:  # Only for small graphs
            try:
                paths['diameter'] = nx.diameter(graph)
            except:
                paths['diameter'] = -1
        
        return paths
    
    def find_communities(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """Find communities in graph using Louvain algorithm"""
        # Convert to undirected if needed
        if graph.is_directed():
            graph = graph.to_undirected()
        
        # Find communities
        partition = community.best_partition(graph)
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        return dict(communities)

class CommunityDetector:
    """Detect communities in graphs"""
    
    def detect_louvain(self, graph: nx.Graph) -> Dict[str, int]:
        """Detect communities using Louvain algorithm"""
        # Convert to undirected if needed
        if graph.is_directed():
            graph = graph.to_undirected()
        
        # Detect communities
        return community.best_partition(graph)
    
    def detect_label_propagation(self, graph: nx.Graph) -> Dict[str, int]:
        """Detect communities using label propagation"""
        communities = nx.community.label_propagation_communities(graph)
        
        # Convert to dict format
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        
        return partition
    
    def detect_girvan_newman(self, graph: nx.Graph, num_communities: int = 5) -> Dict[str, int]:
        """Detect communities using Girvan-Newman algorithm"""
        # This is expensive for large graphs
        if graph.number_of_nodes() > 100:
            logger.warning("Girvan-Newman is expensive for large graphs")
            return {}
        
        # Get communities
        comp = nx.community.girvan_newman(graph)
        
        # Get desired number of communities
        for _ in range(num_communities - 1):
            communities = next(comp)
        
        # Convert to dict format
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        
        return partition
    
    def calculate_modularity(self, graph: nx.Graph, partition: Dict[str, int]) -> float:
        """Calculate modularity of a partition"""
        # Convert partition dict to list of sets
        communities = defaultdict(set)
        for node, comm_id in partition.items():
            communities[comm_id].add(node)
        
        communities_list = list(communities.values())
        
        # Calculate modularity
        return nx.community.modularity(graph, communities_list)
