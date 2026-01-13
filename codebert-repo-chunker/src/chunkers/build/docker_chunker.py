"""
Docker-specific chunker for Dockerfiles and Docker Compose files
Handles Docker build contexts, multi-stage builds, and compose services intelligently
"""

import re
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import logging

from src.core.base_chunker import BaseChunker, Chunk
from src.core.file_context import FileContext
from src.utils.yaml_utils import YAMLProcessor
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class DockerStage:
    """Represents a Docker build stage"""
    name: Optional[str]
    base_image: str
    instructions: List[Tuple[str, str]]  # (instruction, arguments)
    start_line: int
    end_line: int
    size_estimate: int  # Estimated layer size
    dependencies: List[str]  # Other stages it depends on
    
@dataclass
class DockerLayer:
    """Represents a Docker layer"""
    instruction: str
    arguments: str
    line_number: int
    creates_layer: bool
    estimated_size: int
    cache_bust_probability: float  # Likelihood of cache invalidation

@dataclass
class ComposeService:
    """Represents a Docker Compose service"""
    name: str
    image: Optional[str]
    build: Optional[Dict[str, Any]]
    ports: List[str]
    volumes: List[str]
    environment: Dict[str, str]
    depends_on: List[str]
    networks: List[str]
    deploy: Optional[Dict[str, Any]]
    healthcheck: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

class DockerfileAnalyzer:
    """Analyzes Dockerfile structure for intelligent chunking"""
    
    # Instructions that create new layers
    LAYER_CREATING_INSTRUCTIONS = {
        'FROM', 'RUN', 'COPY', 'ADD', 'WORKDIR'
    }
    
    # Instructions that don't create layers but modify metadata
    METADATA_INSTRUCTIONS = {
        'ENV', 'LABEL', 'EXPOSE', 'USER', 'VOLUME', 
        'STOPSIGNAL', 'SHELL', 'MAINTAINER'
    }
    
    # Instructions for build process
    BUILD_INSTRUCTIONS = {
        'ARG', 'ONBUILD'
    }
    
    # Entry point instructions
    ENTRYPOINT_INSTRUCTIONS = {
        'CMD', 'ENTRYPOINT', 'HEALTHCHECK'
    }
    
    # Common base images and their typical sizes (MB)
    BASE_IMAGE_SIZES = {
        'alpine': 5,
        'ubuntu': 70,
        'debian': 100,
        'centos': 200,
        'node': 900,
        'python': 900,
        'openjdk': 500,
        'nginx': 140,
        'redis': 100,
        'postgres': 350,
        'mysql': 450,
        'scratch': 0
    }
    
    def __init__(self):
        self.stages: List[DockerStage] = []
        self.global_args: Dict[str, str] = {}
        self.base_images: Set[str] = set()
        
    def analyze_dockerfile(self, content: str) -> Dict[str, Any]:
        """
        Analyze Dockerfile structure
        
        Args:
            content: Dockerfile content
            
        Returns:
            Analysis results dictionary
        """
        lines = content.split('\n')
        
        # Parse instructions
        instructions = self._parse_instructions(lines)
        
        # Identify stages
        stages = self._identify_stages(instructions)
        
        # Analyze layers
        layers = self._analyze_layers(instructions)
        
        # Calculate build complexity
        complexity = self._calculate_complexity(instructions, stages)
        
        # Detect patterns
        patterns = self._detect_patterns(instructions)
        
        # Optimization suggestions
        optimizations = self._suggest_optimizations(instructions, layers)
        
        return {
            'stages': stages,
            'layers': layers,
            'instructions': instructions,
            'complexity': complexity,
            'patterns': patterns,
            'optimizations': optimizations,
            'global_args': self.global_args,
            'base_images': list(self.base_images)
        }
    
    def _parse_instructions(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse Dockerfile instructions"""
        instructions = []
        current_instruction = None
        line_number = 0
        
        for i, line in enumerate(lines):
            line_number = i + 1
            
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Handle line continuations
            if line.endswith('\\'):
                if current_instruction:
                    current_instruction['arguments'] += ' ' + line[:-1].strip()
                continue
            elif current_instruction:
                current_instruction['arguments'] += ' ' + line.strip()
                instructions.append(current_instruction)
                current_instruction = None
                continue
            
            # Parse new instruction
            match = re.match(r'^([A-Z]+)\s*(.*)', line)
            if match:
                instruction = match.group(1)
                arguments = match.group(2).strip()
                
                # Handle line continuation
                if arguments.endswith('\\'):
                    current_instruction = {
                        'instruction': instruction,
                        'arguments': arguments[:-1].strip(),
                        'line_number': line_number,
                        'line_count': 1
                    }
                else:
                    instructions.append({
                        'instruction': instruction,
                        'arguments': arguments,
                        'line_number': line_number,
                        'line_count': 1
                    })
                
                # Track global ARGs
                if instruction == 'ARG' and not self._in_stage(instructions):
                    key_value = arguments.split('=', 1)
                    if len(key_value) == 2:
                        self.global_args[key_value[0]] = key_value[1]
        
        return instructions
    
    def _identify_stages(self, instructions: List[Dict[str, Any]]) -> List[DockerStage]:
        """Identify multi-stage build stages"""
        stages = []
        current_stage = None
        stage_instructions = []
        
        for i, inst in enumerate(instructions):
            if inst['instruction'] == 'FROM':
                # Save previous stage
                if current_stage:
                    current_stage.instructions = stage_instructions
                    current_stage.end_line = instructions[i-1]['line_number']
                    stages.append(current_stage)
                    stage_instructions = []
                
                # Parse FROM instruction
                from_args = inst['arguments']
                stage_name = None
                
                # Check for AS clause
                as_match = re.search(r'\s+AS\s+(\S+)', from_args, re.IGNORECASE)
                if as_match:
                    stage_name = as_match.group(1)
                    base_image = from_args[:as_match.start()].strip()
                else:
                    base_image = from_args.strip()
                
                # Track base images
                self.base_images.add(base_image.split(':')[0])
                
                # Check for stage dependencies
                dependencies = []
                if base_image in [s.name for s in stages if s.name]:
                    dependencies.append(base_image)
                
                current_stage = DockerStage(
                    name=stage_name,
                    base_image=base_image,
                    instructions=[],
                    start_line=inst['line_number'],
                    end_line=inst['line_number'],
                    size_estimate=self._estimate_base_size(base_image),
                    dependencies=dependencies
                )
            else:
                stage_instructions.append((inst['instruction'], inst['arguments']))
        
        # Save last stage
        if current_stage:
            current_stage.instructions = stage_instructions
            if instructions:
                current_stage.end_line = instructions[-1]['line_number']
            stages.append(current_stage)
        
        return stages
    
    def _analyze_layers(self, instructions: List[Dict[str, Any]]) -> List[DockerLayer]:
        """Analyze Docker layers"""
        layers = []
        
        for inst in instructions:
            creates_layer = inst['instruction'] in self.LAYER_CREATING_INSTRUCTIONS
            estimated_size = self._estimate_instruction_size(inst)
            cache_bust = self._calculate_cache_bust_probability(inst)
            
            layers.append(DockerLayer(
                instruction=inst['instruction'],
                arguments=inst['arguments'],
                line_number=inst['line_number'],
                creates_layer=creates_layer,
                estimated_size=estimated_size,
                cache_bust_probability=cache_bust
            ))
        
        return layers
    
    def _calculate_complexity(self, instructions: List[Dict[str, Any]], 
                             stages: List[DockerStage]) -> Dict[str, Any]:
        """Calculate Dockerfile complexity metrics"""
        return {
            'instruction_count': len(instructions),
            'stage_count': len(stages),
            'layer_count': sum(1 for inst in instructions 
                             if inst['instruction'] in self.LAYER_CREATING_INSTRUCTIONS),
            'estimated_size_mb': sum(self._estimate_instruction_size(inst) 
                                   for inst in instructions),
            'multi_stage': len(stages) > 1,
            'uses_build_args': any(inst['instruction'] == 'ARG' for inst in instructions),
            'uses_secrets': any('secret' in inst['arguments'].lower() 
                              for inst in instructions),
            'complexity_score': self._calculate_complexity_score(instructions, stages)
        }
    
    def _calculate_complexity_score(self, instructions: List[Dict[str, Any]], 
                                   stages: List[DockerStage]) -> float:
        """Calculate overall complexity score"""
        score = 0.0
        
        # Base score from instruction count
        score += min(len(instructions) / 50, 1.0) * 0.3
        
        # Multi-stage complexity
        score += min(len(stages) / 3, 1.0) * 0.2
        
        # RUN instruction complexity
        run_instructions = [i for i in instructions if i['instruction'] == 'RUN']
        for run in run_instructions:
            # Complex RUN commands with multiple statements
            if '&&' in run['arguments'] or ';' in run['arguments']:
                score += 0.05
        
        # Build argument usage
        if any(i['instruction'] == 'ARG' for i in instructions):
            score += 0.1
        
        # Health check presence
        if any(i['instruction'] == 'HEALTHCHECK' for i in instructions):
            score += 0.1
        
        return min(score, 1.0)
    
    def _detect_patterns(self, instructions: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Detect common Dockerfile patterns"""
        patterns = {
            'uses_apt_get': False,
            'uses_apk': False,
            'uses_yum': False,
            'uses_pip': False,
            'uses_npm': False,
            'uses_yarn': False,
            'uses_maven': False,
            'uses_gradle': False,
            'layer_caching_optimized': False,
            'multi_stage_build': False,
            'distroless_final': False,
            'non_root_user': False,
            'healthcheck_defined': False
        }
        
        for inst in instructions:
            if inst['instruction'] == 'RUN':
                args = inst['arguments'].lower()
                
                # Package managers
                if 'apt-get' in args or 'apt ' in args:
                    patterns['uses_apt_get'] = True
                if 'apk ' in args:
                    patterns['uses_apk'] = True
                if 'yum ' in args:
                    patterns['uses_yum'] = True
                
                # Language package managers
                if 'pip install' in args:
                    patterns['uses_pip'] = True
                if 'npm install' in args or 'npm ci' in args:
                    patterns['uses_npm'] = True
                if 'yarn install' in args:
                    patterns['uses_yarn'] = True
                if 'mvn ' in args:
                    patterns['uses_maven'] = True
                if 'gradle ' in args:
                    patterns['uses_gradle'] = True
            
            elif inst['instruction'] == 'USER':
                if inst['arguments'] != 'root':
                    patterns['non_root_user'] = True
            
            elif inst['instruction'] == 'HEALTHCHECK':
                patterns['healthcheck_defined'] = True
        
        # Check for multi-stage patterns
        from_instructions = [i for i in instructions if i['instruction'] == 'FROM']
        if len(from_instructions) > 1:
            patterns['multi_stage_build'] = True
            
            # Check for distroless final stage
            last_from = from_instructions[-1]['arguments'].lower()
            if 'distroless' in last_from or 'scratch' in last_from:
                patterns['distroless_final'] = True
        
        # Check for layer caching optimization
        # Dependencies should be copied before application code
        copy_instructions = [i for i in instructions if i['instruction'] == 'COPY']
        if len(copy_instructions) >= 2:
            # Simple heuristic: package files copied before source
            for i, copy_inst in enumerate(copy_instructions[:-1]):
                if any(pkg in copy_inst['arguments'] 
                      for pkg in ['package.json', 'requirements.txt', 'pom.xml', 'build.gradle']):
                    patterns['layer_caching_optimized'] = True
                    break
        
        return patterns
    
    def _suggest_optimizations(self, instructions: List[Dict[str, Any]], 
                              layers: List[DockerLayer]) -> List[str]:
        """Suggest Dockerfile optimizations"""
        suggestions = []
        
        # Check for multiple RUN instructions that could be combined
        run_instructions = [i for i in instructions if i['instruction'] == 'RUN']
        if len(run_instructions) > 3:
            suggestions.append("Consider combining multiple RUN instructions to reduce layers")
        
        # Check for package manager cleanup
        for inst in instructions:
            if inst['instruction'] == 'RUN':
                args = inst['arguments'].lower()
                if 'apt-get install' in args and 'rm -rf /var/lib/apt/lists/*' not in args:
                    suggestions.append("Add 'rm -rf /var/lib/apt/lists/*' to apt-get install commands")
                if 'yum install' in args and 'yum clean all' not in args:
                    suggestions.append("Add 'yum clean all' after yum install commands")
        
        # Check for COPY before dependency installation
        copy_indices = [i for i, inst in enumerate(instructions) if inst['instruction'] == 'COPY']
        run_indices = [i for i, inst in enumerate(instructions) if inst['instruction'] == 'RUN']
        
        if copy_indices and run_indices:
            first_copy = min(copy_indices)
            dependency_runs = [i for i in run_indices 
                             if any(pkg in instructions[i]['arguments'].lower() 
                                   for pkg in ['install', 'npm', 'pip', 'maven', 'gradle'])]
            if dependency_runs and first_copy < min(dependency_runs):
                suggestions.append("Copy dependency files before source code for better caching")
        
        # Check for USER instruction
        if not any(inst['instruction'] == 'USER' for inst in instructions):
            suggestions.append("Consider running as non-root user for security")
        
        # Check for HEALTHCHECK
        if not any(inst['instruction'] == 'HEALTHCHECK' for inst in instructions):
            suggestions.append("Consider adding HEALTHCHECK for container health monitoring")
        
        # Check for .dockerignore usage hint
        add_copy_count = sum(1 for inst in instructions 
                           if inst['instruction'] in ['ADD', 'COPY'])
        if add_copy_count > 5:
            suggestions.append("Ensure .dockerignore is properly configured to exclude unnecessary files")
        
        return suggestions
    
    def _estimate_base_size(self, base_image: str) -> int:
        """Estimate base image size in MB"""
        image_name = base_image.split(':')[0].lower()
        
        for known_image, size in self.BASE_IMAGE_SIZES.items():
            if known_image in image_name:
                return size
        
        return 100  # Default estimate
    
    def _estimate_instruction_size(self, instruction: Dict[str, Any]) -> int:
        """Estimate size impact of instruction in MB"""
        inst_type = instruction['instruction']
        args = instruction['arguments'].lower()
        
        if inst_type == 'RUN':
            # Package installations
            if 'apt-get install' in args or 'yum install' in args:
                return 50  # Rough estimate for package installation
            elif 'npm install' in args:
                return 100  # Node modules can be large
            elif 'pip install' in args:
                return 30
            else:
                return 10
        
        elif inst_type in ['COPY', 'ADD']:
            # Would need actual file sizes, using rough estimate
            if 'node_modules' in args:
                return 200
            elif '.jar' in args or '.war' in args:
                return 50
            else:
                return 10
        
        return 1  # Minimal size for other instructions
    
    def _calculate_cache_bust_probability(self, instruction: Dict[str, Any]) -> float:
        """Calculate probability of cache invalidation"""
        inst_type = instruction['instruction']
        args = instruction['arguments'].lower()
        
        if inst_type in ['COPY', 'ADD']:
            # Source code changes frequently
            if any(src in args for src in ['src/', 'app/', '.py', '.js', '.java']):
                return 0.9
            # Dependencies change less frequently
            elif any(dep in args for dep in ['package.json', 'requirements.txt', 'pom.xml']):
                return 0.3
            else:
                return 0.5
        
        elif inst_type == 'RUN':
            # Updates and upgrades bust cache
            if 'update' in args or 'upgrade' in args:
                return 0.8
            # Installations are more stable
            elif 'install' in args:
                return 0.2
            else:
                return 0.4
        
        return 0.1
    
    def _in_stage(self, instructions: List[Dict[str, Any]]) -> bool:
        """Check if currently in a build stage"""
        return any(inst['instruction'] == 'FROM' for inst in instructions)

class DockerComposeAnalyzer:
    """Analyzes Docker Compose structure"""
    
    def __init__(self):
        self.yaml_processor = YAMLProcessor()
    
    def analyze_compose(self, content: str) -> Dict[str, Any]:
        """
        Analyze Docker Compose file structure
        
        Args:
            content: Docker Compose YAML content
            
        Returns:
            Analysis results
        """
        try:
            data = yaml.safe_load(content)
            
            if not isinstance(data, dict):
                return self._empty_analysis()
            
            # Extract version
            version = data.get('version', '3')
            
            # Extract services
            services = self._extract_services(data.get('services', {}))
            
            # Extract networks
            networks = self._extract_networks(data.get('networks', {}))
            
            # Extract volumes
            volumes = self._extract_volumes(data.get('volumes', {}))
            
            # Extract configs and secrets
            configs = data.get('configs', {})
            secrets = data.get('secrets', {})
            
            # Analyze service dependencies
            dependency_graph = self._build_dependency_graph(services)
            
            # Calculate complexity
            complexity = self._calculate_compose_complexity(
                services, networks, volumes
            )
            
            return {
                'version': version,
                'services': services,
                'networks': networks,
                'volumes': volumes,
                'configs': configs,
                'secrets': secrets,
                'dependency_graph': dependency_graph,
                'complexity': complexity
            }
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse Docker Compose file: {e}")
            return self._empty_analysis()
    
    def _extract_services(self, services_dict: Dict[str, Any]) -> List[ComposeService]:
        """Extract service definitions"""
        services = []
        
        for name, config in services_dict.items():
            if not isinstance(config, dict):
                continue
            
            service = ComposeService(
                name=name,
                image=config.get('image'),
                build=config.get('build'),
                ports=config.get('ports', []),
                volumes=config.get('volumes', []),
                environment=self._extract_environment(config.get('environment')),
                depends_on=self._extract_depends_on(config.get('depends_on')),
                networks=config.get('networks', []),
                deploy=config.get('deploy'),
                healthcheck=config.get('healthcheck'),
                metadata={
                    'restart': config.get('restart'),
                    'command': config.get('command'),
                    'entrypoint': config.get('entrypoint'),
                    'working_dir': config.get('working_dir'),
                    'user': config.get('user'),
                    'privileged': config.get('privileged', False),
                    'cap_add': config.get('cap_add', []),
                    'cap_drop': config.get('cap_drop', [])
                }
            )
            
            services.append(service)
        
        return services
    
    def _extract_environment(self, env_config: Any) -> Dict[str, str]:
        """Extract environment variables"""
        if not env_config:
            return {}
        
        if isinstance(env_config, dict):
            return {k: str(v) for k, v in env_config.items()}
        
        elif isinstance(env_config, list):
            env_dict = {}
            for item in env_config:
                if '=' in item:
                    key, value = item.split('=', 1)
                    env_dict[key] = value
            return env_dict
        
        return {}
    
    def _extract_depends_on(self, depends_config: Any) -> List[str]:
        """Extract service dependencies"""
        if not depends_config:
            return []
        
        if isinstance(depends_config, list):
            return depends_config
        
        elif isinstance(depends_config, dict):
            # New format with conditions
            return list(depends_config.keys())
        
        return []
    
    def _extract_networks(self, networks_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract network definitions"""
        networks = {}
        
        for name, config in networks_dict.items():
            if config is None:
                networks[name] = {'driver': 'bridge'}
            elif isinstance(config, dict):
                networks[name] = config
        
        return networks
    
    def _extract_volumes(self, volumes_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract volume definitions"""
        volumes = {}
        
        for name, config in volumes_dict.items():
            if config is None:
                volumes[name] = {'driver': 'local'}
            elif isinstance(config, dict):
                volumes[name] = config
        
        return volumes
    
    def _build_dependency_graph(self, services: List[ComposeService]) -> Dict[str, List[str]]:
        """Build service dependency graph"""
        graph = {}
        
        for service in services:
            graph[service.name] = service.depends_on
        
        return graph
    
    def _calculate_compose_complexity(self, services: List[ComposeService],
                                     networks: Dict[str, Any],
                                     volumes: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Docker Compose complexity"""
        return {
            'service_count': len(services),
            'network_count': len(networks),
            'volume_count': len(volumes),
            'total_ports': sum(len(s.ports) for s in services),
            'total_volumes': sum(len(s.volumes) for s in services),
            'has_dependencies': any(s.depends_on for s in services),
            'uses_build': any(s.build for s in services),
            'uses_deploy': any(s.deploy for s in services),
            'uses_healthcheck': any(s.healthcheck for s in services),
            'complexity_score': self._calculate_complexity_score(services)
        }
    
    def _calculate_complexity_score(self, services: List[ComposeService]) -> float:
        """Calculate overall complexity score"""
        if not services:
            return 0.0
        
        score = min(len(services) / 10, 1.0) * 0.3
        
        # Dependency complexity
        max_deps = max(len(s.depends_on) for s in services) if services else 0
        score += min(max_deps / 5, 1.0) * 0.2
        
        # Configuration complexity
        for service in services:
            if service.build:
                score += 0.05
            if service.deploy:
                score += 0.05
            if service.healthcheck:
                score += 0.05
            if len(service.environment) > 5:
                score += 0.05
        
        return min(score, 1.0)
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'version': None,
            'services': [],
            'networks': {},
            'volumes': {},
            'configs': {},
            'secrets': {},
            'dependency_graph': {},
            'complexity': {}
        }

class DockerChunker(BaseChunker):
    """Chunker for Docker-related files"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, max_tokens)
        self.dockerfile_analyzer = DockerfileAnalyzer()
        self.compose_analyzer = DockerComposeAnalyzer()
    
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from Docker files
        
        Args:
            content: File content
            file_context: File context
            
        Returns:
            List of chunks
        """
        file_name = file_context.path.name.lower()
        
        if file_name == 'dockerfile' or file_name.endswith('.dockerfile'):
            return self._chunk_dockerfile(content, file_context)
        elif file_name in ['docker-compose.yml', 'docker-compose.yaml', 'compose.yml', 'compose.yaml']:
            return self._chunk_compose(content, file_context)
        elif file_name == '.dockerignore':
            return self._chunk_dockerignore(content, file_context)
        else:
            # Fallback for unknown Docker-related files
            return self._chunk_generic_docker(content, file_context)
    
    def _chunk_dockerfile(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Chunk Dockerfile content"""
        try:
            analysis = self.dockerfile_analyzer.analyze_dockerfile(content)
            chunks = []
            
            # Create chunks for each stage in multi-stage builds
            if analysis['stages']:
                for stage in analysis['stages']:
                    stage_chunk = self._create_stage_chunk(content, stage, file_context)
                    if stage_chunk:
                        chunks.append(stage_chunk)
            
            # If no stages or single stage, chunk by logical sections
            if not chunks or len(analysis['stages']) == 1:
                chunks = self._chunk_dockerfile_sections(content, analysis, file_context)
            
            # Add metadata chunk with analysis results
            metadata_chunk = self._create_dockerfile_metadata_chunk(analysis, file_context)
            if metadata_chunk:
                chunks.insert(0, metadata_chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking Dockerfile {file_context.path}: {e}")
            return self._fallback_dockerfile_chunking(content, file_context)
    
    def _create_stage_chunk(self, content: str, stage: DockerStage,
                          file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for a Docker stage"""
        lines = content.split('\n')
        stage_lines = lines[stage.start_line - 1:stage.end_line]
        stage_content = '\n'.join(stage_lines)
        
        # Check token limit
        if self.count_tokens(stage_content) > self.max_tokens:
            # Split large stage into smaller chunks
            return self._split_large_stage(stage_content, stage, file_context)
        
        return self.create_chunk(
            content=stage_content,
            chunk_type='dockerfile_stage',
            metadata={
                'stage_name': stage.name or 'unnamed',
                'base_image': stage.base_image,
                'instruction_count': len(stage.instructions),
                'size_estimate_mb': stage.size_estimate,
                'dependencies': stage.dependencies,
                'is_multi_stage': True
            },
            file_path=str(file_context.path),
            start_line=stage.start_line,
            end_line=stage.end_line
        )
    
    def _chunk_dockerfile_sections(self, content: str, analysis: Dict[str, Any],
                                  file_context: FileContext) -> List[Chunk]:
        """Chunk Dockerfile by logical sections"""
        chunks = []
        instructions = analysis['instructions']
        
        if not instructions:
            return []
        
        # Group instructions into logical sections
        sections = self._group_dockerfile_instructions(instructions)
        
        for section_name, section_instructions in sections.items():
            if not section_instructions:
                continue
            
            # Build section content
            lines = content.split('\n')
            section_lines = []
            
            for inst in section_instructions:
                line_num = inst['line_number'] - 1
                if line_num < len(lines):
                    section_lines.append(lines[line_num])
                    
                    # Include continuation lines
                    for i in range(1, inst.get('line_count', 1)):
                        if line_num + i < len(lines):
                            section_lines.append(lines[line_num + i])
            
            section_content = '\n'.join(section_lines)
            
            # Check token limit
            if self.count_tokens(section_content) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=section_content,
                    chunk_type='dockerfile_section',
                    metadata={
                        'section': section_name,
                        'instruction_count': len(section_instructions),
                        'instructions': [inst['instruction'] for inst in section_instructions]
                    },
                    file_path=str(file_context.path)
                ))
            else:
                # Split large section
                split_chunks = self._split_large_section(
                    section_content, section_name, file_context
                )
                chunks.extend(split_chunks)
        
        return chunks
    
    def _group_dockerfile_instructions(self, instructions: List[Dict[str, Any]]) -> Dict[str, List]:
        """Group Dockerfile instructions into logical sections"""
        sections = {
            'base': [],
            'arguments': [],
            'environment': [],
            'dependencies': [],
            'application': [],
            'configuration': [],
            'entrypoint': []
        }
        
        for inst in instructions:
            inst_type = inst['instruction']
            
            if inst_type == 'FROM':
                sections['base'].append(inst)
            elif inst_type == 'ARG':
                sections['arguments'].append(inst)
            elif inst_type in ['ENV', 'LABEL']:
                sections['environment'].append(inst)
            elif inst_type == 'RUN':
                # Classify RUN instructions
                args = inst['arguments'].lower()
                if any(pkg in args for pkg in ['apt', 'yum', 'apk', 'pip', 'npm', 'maven', 'gradle']):
                    sections['dependencies'].append(inst)
                else:
                    sections['application'].append(inst)
            elif inst_type in ['COPY', 'ADD', 'WORKDIR']:
                sections['application'].append(inst)
            elif inst_type in ['EXPOSE', 'VOLUME', 'USER', 'STOPSIGNAL']:
                sections['configuration'].append(inst)
            elif inst_type in ['CMD', 'ENTRYPOINT', 'HEALTHCHECK']:
                sections['entrypoint'].append(inst)
            else:
                # Default to configuration
                sections['configuration'].append(inst)
        
        # Remove empty sections
        return {k: v for k, v in sections.items() if v}
    
    def _create_dockerfile_metadata_chunk(self, analysis: Dict[str, Any],
                                        file_context: FileContext) -> Optional[Chunk]:
        """Create metadata chunk with Dockerfile analysis"""
        metadata_content = []
        
        metadata_content.append("# Dockerfile Analysis")
        metadata_content.append(f"# Stages: {analysis['complexity']['stage_count']}")
        metadata_content.append(f"# Layers: {analysis['complexity']['layer_count']}")
        metadata_content.append(f"# Estimated Size: {analysis['complexity']['estimated_size_mb']}MB")
        metadata_content.append(f"# Complexity Score: {analysis['complexity']['complexity_score']:.2f}")
        
        if analysis['patterns']:
            metadata_content.append("\n# Detected Patterns:")
            for pattern, value in analysis['patterns'].items():
                if value:
                    metadata_content.append(f"#   - {pattern}")
        
        if analysis['optimizations']:
            metadata_content.append("\n# Optimization Suggestions:")
            for suggestion in analysis['optimizations']:
                metadata_content.append(f"#   - {suggestion}")
        
        metadata_str = '\n'.join(metadata_content)
        
        return self.create_chunk(
            content=metadata_str,
            chunk_type='dockerfile_metadata',
            metadata={
                'is_metadata': True,
                'complexity': analysis['complexity'],
                'patterns': analysis['patterns'],
                'optimization_count': len(analysis['optimizations'])
            },
            file_path=str(file_context.path)
        )
    
    def _chunk_compose(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Chunk Docker Compose file"""
        try:
            analysis = self.compose_analyzer.analyze_compose(content)
            chunks = []
            
            # Create chunk for each service
            for service in analysis['services']:
                service_chunk = self._create_service_chunk(service, file_context)
                if service_chunk:
                    chunks.append(service_chunk)
            
            # Create chunk for networks if present
            if analysis['networks']:
                networks_chunk = self._create_networks_chunk(
                    analysis['networks'], file_context
                )
                if networks_chunk:
                    chunks.append(networks_chunk)
            
            # Create chunk for volumes if present
            if analysis['volumes']:
                volumes_chunk = self._create_volumes_chunk(
                    analysis['volumes'], file_context
                )
                if volumes_chunk:
                    chunks.append(volumes_chunk)
            
            # Create metadata chunk
            metadata_chunk = self._create_compose_metadata_chunk(analysis, file_context)
            if metadata_chunk:
                chunks.insert(0, metadata_chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking Docker Compose {file_context.path}: {e}")
            return self._fallback_yaml_chunking(content, file_context)
    
    def _create_service_chunk(self, service: ComposeService,
                             file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for a Docker Compose service"""
        service_dict = {
            service.name: {}
        }
        
        # Build service configuration
        if service.image:
            service_dict[service.name]['image'] = service.image
        if service.build:
            service_dict[service.name]['build'] = service.build
        if service.ports:
            service_dict[service.name]['ports'] = service.ports
        if service.volumes:
            service_dict[service.name]['volumes'] = service.volumes
        if service.environment:
            service_dict[service.name]['environment'] = service.environment
        if service.depends_on:
            service_dict[service.name]['depends_on'] = service.depends_on
        if service.networks:
            service_dict[service.name]['networks'] = service.networks
        if service.deploy:
            service_dict[service.name]['deploy'] = service.deploy
        if service.healthcheck:
            service_dict[service.name]['healthcheck'] = service.healthcheck
        
        # Add metadata fields
        for key, value in service.metadata.items():
            if value is not None:
                service_dict[service.name][key] = value
        
        # Convert to YAML
        service_yaml = yaml.dump({'services': service_dict}, default_flow_style=False)
        
        # Check token limit
        if self.count_tokens(service_yaml) <= self.max_tokens:
            return self.create_chunk(
                content=service_yaml,
                chunk_type='compose_service',
                metadata={
                    'service_name': service.name,
                    'has_build': service.build is not None,
                    'has_image': service.image is not None,
                    'port_count': len(service.ports),
                    'volume_count': len(service.volumes),
                    'depends_on': service.depends_on,
                    'networks': service.networks
                },
                file_path=str(file_context.path)
            )
        
        return None
    
    def _create_networks_chunk(self, networks: Dict[str, Any],
                              file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for Docker Compose networks"""
        networks_yaml = yaml.dump({'networks': networks}, default_flow_style=False)
        
        if self.count_tokens(networks_yaml) <= self.max_tokens:
            return self.create_chunk(
                content=networks_yaml,
                chunk_type='compose_networks',
                metadata={
                    'network_count': len(networks),
                    'network_names': list(networks.keys())
                },
                file_path=str(file_context.path)
            )
        
        return None
    
    def _create_volumes_chunk(self, volumes: Dict[str, Any],
                             file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for Docker Compose volumes"""
        volumes_yaml = yaml.dump({'volumes': volumes}, default_flow_style=False)
        
        if self.count_tokens(volumes_yaml) <= self.max_tokens:
            return self.create_chunk(
                content=volumes_yaml,
                chunk_type='compose_volumes',
                metadata={
                    'volume_count': len(volumes),
                    'volume_names': list(volumes.keys())
                },
                file_path=str(file_context.path)
            )
        
        return None
    
    def _create_compose_metadata_chunk(self, analysis: Dict[str, Any],
                                      file_context: FileContext) -> Optional[Chunk]:
        """Create metadata chunk for Docker Compose"""
        metadata_lines = []
        
        metadata_lines.append(f"# Docker Compose Version: {analysis['version']}")
        metadata_lines.append(f"# Services: {analysis['complexity']['service_count']}")
        metadata_lines.append(f"# Networks: {analysis['complexity']['network_count']}")
        metadata_lines.append(f"# Volumes: {analysis['complexity']['volume_count']}")
        metadata_lines.append(f"# Complexity Score: {analysis['complexity']['complexity_score']:.2f}")
        
        if analysis['dependency_graph']:
            metadata_lines.append("\n# Service Dependencies:")
            for service, deps in analysis['dependency_graph'].items():
                if deps:
                    metadata_lines.append(f"#   {service} -> {', '.join(deps)}")
        
        metadata_content = '\n'.join(metadata_lines)
        
        return self.create_chunk(
            content=metadata_content,
            chunk_type='compose_metadata',
            metadata={
                'is_metadata': True,
                'version': analysis['version'],
                'complexity': analysis['complexity'],
                'has_dependencies': analysis['complexity']['has_dependencies']
            },
            file_path=str(file_context.path)
        )
    
    def _chunk_dockerignore(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Chunk .dockerignore file"""
        chunks = []
        lines = content.split('\n')
        
        # Group patterns by category
        categories = {
            'version_control': [],
            'dependencies': [],
            'build_artifacts': [],
            'documentation': [],
            'development': [],
            'other': []
        }
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Categorize patterns
            if any(vc in stripped for vc in ['.git', '.svn', '.hg']):
                categories['version_control'].append(line)
            elif any(dep in stripped for dep in ['node_modules', 'vendor', '.venv', '__pycache__']):
                categories['dependencies'].append(line)
            elif any(build in stripped for build in ['dist/', 'build/', 'target/', '*.class', '*.jar']):
                categories['build_artifacts'].append(line)
            elif any(doc in stripped for doc in ['README', 'docs/', '*.md', 'LICENSE']):
                categories['documentation'].append(line)
            elif any(dev in stripped for dev in ['.env', '.vscode', '.idea', '*.log', 'test/']):
                categories['development'].append(line)
            else:
                categories['other'].append(line)
        
        # Create chunks for non-empty categories
        for category, patterns in categories.items():
            if patterns:
                chunk_content = '\n'.join([
                    f"# {category.replace('_', ' ').title()} patterns",
                    *patterns
                ])
                
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='dockerignore',
                    metadata={
                        'category': category,
                        'pattern_count': len(patterns)
                    },
                    file_path=str(file_context.path)
                ))
        
        return chunks if chunks else [self.create_chunk(
            content=content,
            chunk_type='dockerignore',
            metadata={'is_empty': not bool(content.strip())},
            file_path=str(file_context.path)
        )]
    
    def _split_large_stage(self, content: str, stage: DockerStage,
                          file_context: FileContext) -> Optional[Chunk]:
        """Split large Docker stage into smaller chunk"""
        # For now, return truncated version
        # In production, would implement more sophisticated splitting
        lines = content.split('\n')
        truncated_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            if current_tokens + line_tokens > self.max_tokens * 0.9:
                break
            truncated_lines.append(line)
            current_tokens += line_tokens
        
        return self.create_chunk(
            content='\n'.join(truncated_lines),
            chunk_type='dockerfile_stage_partial',
            metadata={
                'stage_name': stage.name or 'unnamed',
                'base_image': stage.base_image,
                'is_partial': True
            },
            file_path=str(file_context.path)
        )
    
    def _split_large_section(self, content: str, section_name: str,
                            file_context: FileContext) -> List[Chunk]:
        """Split large Dockerfile section"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_tokens = 0
        part_index = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens * 0.9 and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='dockerfile_section_part',
                    metadata={
                        'section': section_name,
                        'part_index': part_index,
                        'is_partial': True
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = []
                current_tokens = 0
                part_index += 1
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='dockerfile_section_part',
                metadata={
                    'section': section_name,
                    'part_index': part_index,
                    'is_partial': True
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _chunk_generic_docker(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Generic chunking for Docker-related files"""
        # Simple line-based chunking with Docker context
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='docker_generic',
                    metadata={
                        'file_name': file_context.path.name
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='docker_generic',
                metadata={
                    'file_name': file_context.path.name
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _fallback_dockerfile_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback chunking for Dockerfiles"""
        logger.warning(f"Using fallback chunking for Dockerfile {file_context.path}")
        return self._chunk_generic_docker(content, file_context)
    
    def _fallback_yaml_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback chunking for YAML files"""
        logger.warning(f"Using fallback YAML chunking for {file_context.path}")
        return self._chunk_generic_docker(content, file_context)