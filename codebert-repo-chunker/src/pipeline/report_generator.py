import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ReportGenerator:
    """Generates execution reports for the pipeline"""
    
    def __init__(self, output_dir: Path = Path("reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate(self, stats: Dict[str, Any], session_id: str, repo_name: str = "unknown") -> Path:
        """
        Generate JSON and Markdown reports.
        
        Args:
            stats: Pipeline statistics dictionary
            session_id: Unique ID for this run
            repo_name: Name of the repository being analyzed
            
        Returns:
            Path to the main report file
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_base = self.output_dir / f"report_{timestamp}_{session_id}_{repo_name}"
        
        # 1. JSON Report (Machine Readable)
        json_path = report_base.with_suffix(".json")
        try:
            with open(json_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write JSON report: {e}")
            
        # 2. Markdown Report (Human Readable)
        md_path = report_base.with_suffix(".md")
        try:
            markdown = self._format_markdown(stats, session_id, timestamp)
            with open(md_path, 'w') as f:
                f.write(markdown)
            logger.info(f"Report generated at {md_path}")
            return md_path
        except Exception as e:
            logger.error(f"Failed to write Markdown report: {e}")
            return json_path

    def _format_markdown(self, stats: Dict[str, Any], session_id: str, timestamp: str) -> str:
        md = f"# Pipeline Execution Report\n\n"
        md += f"**Session ID:** {session_id}\n"
        md += f"**Timestamp:** {timestamp}\n"
        md += f"**Status:** {stats.get('status', 'UNKNOWN')}\n\n"
        
        md += "## Summary\n"
        md += f"- **Files Scanned:** {stats.get('files_scanned', 0)}\n"
        md += f"- **Chunks Created:** {stats.get('chunks_created', 0)}\n"
        md += f"- **Duration:** {stats.get('duration_seconds', 0)}s\n\n"
        
        if 'errors' in stats and stats['errors']:
            md += "## Errors\n"
            for err in stats['errors']:
                md += f"- {err}\n"
            md += "\n"
            
        return md

    def generate_report(self, stats: Dict[str, Any], dependency_graph: Dict[str, Any], session_id: str = None, module_map: Dict[str, str] = None, repo_name: str = "unknown") -> Path:
        """
        Convenience wrapper for full report generation.
        Generates JSON, Markdown, and HTML reports.
        """
        if session_id is None:
            session_id = datetime.now(timezone.utc).strftime("%H%M%S")
            
        # Generate JSON and Markdown reports
        main_report_path = self.generate(stats, session_id, repo_name)
        
        # Generate HTML report (interactive)
        self.generate_html_report(stats, dependency_graph, session_id, module_map=module_map, repo_name=repo_name)
        
        return main_report_path

    def generate_html_report(self, stats: Dict[str, Any], dependency_graph: Dict[str, Any], session_id: str, module_map: Dict[str, str] = None, repo_name: str = "unknown") -> Path:
        """
        Generate interactive HTML visualization of dependencies.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        html_path = self.output_dir / f"dependency_graph_{timestamp}_{session_id}_{repo_name}.html"
        
        # Prepare nodes and edges for vis.js
        nodes = []
        edges = []
        
        # Graph is expected to be {node: [deps]}
        if dependency_graph:
            processed_nodes = set()
            for source, targets in dependency_graph.items():
                if source not in processed_nodes:
                    # Source is always a file path ID
                    nodes.append({"id": source, "label": Path(source).name, "group": "source"})
                    processed_nodes.add(source)
                    
                for target_dict in targets:
                    target_name = target_dict.get('name')
                    if not target_name: continue
                    
                    # Resolve target to file path if possible
                    target_id = target_name
                    group = "dependency"
                    
                    if module_map and target_name in module_map:
                        target_id = module_map[target_name]
                        group = "source" # It's a known source file
                        
                    if target_id not in processed_nodes:
                        label = target_name if group == "dependency" else Path(target_id).name
                        nodes.append({"id": target_id, "label": label, "group": group})
                        processed_nodes.add(target_id)
                        
                    edges.append({"from": source, "to": target_id, "arrows": "to"})

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dependency Graph - {session_id}</title>
    <!-- Use cdnjs for better reliability -->
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js" onerror="document.getElementById('error-msg').style.display='block';"></script>
    <style type="text/css">
        #mynetwork {{
            width: 100%;
            height: 800px;
            border: 1px solid lightgray;
            background-color: #ffffff;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; color: #333; }}
        .stats {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e9ecef; }}
        .error {{ color: #721c24; background-color: #f8d7da; border-color: #f5c6cb; padding: 15px; border-radius: 5px; margin-bottom: 20px; display: none; }}
        h1 {{ color: #2c3e50; }}
    </style>
</head>
<body>
    <h1>Dependency Graph: {session_id}</h1>
    
    <div id="error-msg" class="error">
        <strong>Error:</strong> Failed to load visualization library (vis-network). Please checks your internet connection.
    </div>
    <div id="js-error" class="error"></div>

    <div class="stats">
        <h3>Statistics</h3>
        <p><strong>Files Scanned:</strong> {stats.get('files_scanned', 0)}</p>
        <p><strong>Total Nodes:</strong> {len(nodes)}</p>
        <p><strong>Total Edges:</strong> {len(edges)}</p>
        <p><strong>Repo:</strong> {repo_name}</p>
    </div>
    
    <div id="mynetwork"></div>
    
    <script type="text/javascript">
        try {{
            // Data
            var nodes = new vis.DataSet({json.dumps(nodes)});
            var edges = new vis.DataSet({json.dumps(edges)});
            
            // Container
            var container = document.getElementById('mynetwork');
            var data = {{ nodes: nodes, edges: edges }};
            
            // Options
            var options = {{
                nodes: {{
                    shape: 'dot',
                    size: 20,
                    font: {{ size: 14, color: '#343a40' }},
                    borderWidth: 2,
                    shadow: true
                }},
                edges: {{
                    width: 1.5,
                    color: {{ color: '#adb5bd', highlight: '#495057' }},
                    arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
                    smooth: {{ type: 'continuous' }}
                }},
                groups: {{
                    source: {{ color: {{ background: '#4dabf7', border: '#339af0' }}, shape: 'dot' }},
                    dependency: {{ color: {{ background: '#ff922b', border: '#fcc419' }}, shape: 'diamond' }}
                }},
                physics: {{
                    enabled: true,
                    stabilization: {{
                        enabled: true,
                        iterations: 1000,
                        updateInterval: 100,
                        onlyDynamicEdges: false,
                        fit: true
                    }},
                    barnesHut: {{
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 95,
                        springConstant: 0.04,
                        damping: 0.09,
                        avoidOverlap: 0.1
                    }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 200,
                    navigationButtons: true,
                    keyboard: true
                }},
                layout: {{
                    improvedLayout: true
                }}
            }};
            
            // Initialize
            var network = new vis.Network(container, data, options);
            
            // Fit to window
            network.once("stabilizationIterationsDone", function() {{
                network.fit({{ 
                    animation: {{
                        duration: 1000,
                        easingFunction: "easeInOutQuad"
                    }}
                }});
            }});
            
        }} catch (err) {{
            document.getElementById('js-error').style.display = 'block';
            document.getElementById('js-error').innerHTML = '<strong>JavaScript Error:</strong> ' + err.message;
            console.error(err);
        }}
    </script>
</body>
</html>
"""
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"HTML report generated at {html_path}")
            return html_path
        except Exception as e:
            logger.error(f"Failed to write HTML report: {e}")
            return Path("")
