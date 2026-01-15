import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates execution reports for the pipeline"""
    
    def __init__(self, output_dir: Path = Path("reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate(self, stats: Dict[str, Any], session_id: str) -> Path:
        """
        Generate JSON and Markdown reports.
        
        Args:
            stats: Pipeline statistics dictionary
            session_id: Unique ID for this run
            
        Returns:
            Path to the main report file
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_base = self.output_dir / f"report_{timestamp}_{session_id}"
        
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

    def generate_report(self, stats: Dict[str, Any], dependency_graph: Dict[str, Any], module_map: Dict[str, str] = None) -> Path:
        """
        Generate all report formats.
        Returns path to the primary report (markdown).
        """
        session_id = f"{stats.get('start_time', datetime.now(timezone.utc)).strftime('%H%M%S')}_{stats.get('scan_id', 'unknown')}"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML first (interactive)
        self.generate_html_report(stats, dependency_graph, session_id, module_map=module_map)
        
        # Graph is expected to be {node: [deps]}
        if dependency_graph:
            processed_nodes = set()
            for source, targets in dependency_graph.items():
                # 1. JSON Report (Machine Readable)
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                report_base = self.output_dir / f"report_{timestamp}_{session_id}"
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
        return Path("") # Should not be reached if dependency_graph is processed or if generate is called

    def generate_html_report(self, stats: Dict[str, Any], dependency_graph: Dict[str, Any], session_id: str, module_map: Dict[str, str] = None) -> Path:
        """
        Generate an interactive HTML report using vis.js
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        html_path = self.output_dir / f"dependency_graph_{timestamp}_{session_id}.html"
        
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
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #mynetwork {{
            width: 100%;
            height: 800px;
            border: 1px solid lightgray;
        }}
        body {{ font-family: sans-serif; margin: 20px; }}
        .stats {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Dependency Graph: {session_id}</h1>
    <div class="stats">
        <h3>Statistics</h3>
        <p><strong>Files Scanned:</strong> {stats.get('files_scanned', 0)}</p>
        <p><strong>Total Nodes:</strong> {len(nodes)}</p>
        <p><strong>Total Edges:</strong> {len(edges)}</p>
    </div>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        var container = document.getElementById('mynetwork');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{
                shape: 'dot',
                size: 16,
                font: {{ size: 14 }}
            }},
            groups: {{
                source: {{ color: '#97C2FC' }},
                dependency: {{ color: '#FB7E81' }}
            }},
            physics: {{
                stabilization: false,
                barnesHut: {{ springLength: 200 }}
            }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""
        try:
            with open(html_path, 'w') as f:
                f.write(html_content)
            logger.info(f"HTML report generated at {html_path}")
            return html_path
        except Exception as e:
            logger.error(f"Failed to write HTML report: {e}")
            return Path("")
