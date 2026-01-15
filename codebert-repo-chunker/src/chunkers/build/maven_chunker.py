"""
Maven-specific chunker for POM files and Maven configurations
Handles Maven project structures, dependencies, and build configurations intelligently
"""

import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from collections import defaultdict

from src.core.base_chunker import BaseChunker, Chunk, ChunkerConfig
from src.core.file_context import FileContext
from src.utils.xml_utils import XMLProcessor, XMLNamespaceHandler
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class MavenElement:
    """Represents a Maven POM element"""
    tag: str
    content: str
    element_type: str  # 'dependency', 'plugin', 'profile', etc.
    metadata: Dict[str, Any]
    start_line: int
    end_line: int
    parent_path: List[str]  # Hierarchy path
    
@dataclass
class MavenProject:
    """Represents a Maven project structure"""
    group_id: Optional[str]
    artifact_id: Optional[str]
    version: Optional[str]
    packaging: Optional[str]
    name: Optional[str]
    description: Optional[str]
    parent: Optional[Dict[str, str]]
    modules: List[str]
    properties: Dict[str, str]
    dependencies: List[Dict[str, Any]]
    dependency_management: List[Dict[str, Any]]
    plugins: List[Dict[str, Any]]
    profiles: List[Dict[str, Any]]
    repositories: List[Dict[str, Any]]
    
class MavenStructureAnalyzer:
    """Analyzes Maven POM structure for intelligent chunking"""
    
    # Maven namespace
    MAVEN_NS = {'maven': 'http://maven.apache.org/POM/4.0.0'}
    
    # Important Maven sections that should be chunked separately
    MAJOR_SECTIONS = [
        'properties',
        'dependencies',
        'dependencyManagement',
        'build',
        'profiles',
        'repositories',
        'pluginRepositories',
        'distributionManagement',
        'reporting',
        'modules'
    ]
    
    # Sections that should be kept together if small
    ATOMIC_SECTIONS = [
        'parent',
        'organization',
        'developers',
        'contributors',
        'licenses',
        'scm',
        'issueManagement',
        'ciManagement'
    ]
    
    def __init__(self):
        self.xml_processor = XMLProcessor()
        self.namespace_handler = XMLNamespaceHandler()
    
    def analyze_pom(self, content: str) -> MavenProject:
        """
        Analyze POM file structure
        
        Args:
            content: POM XML content
            
        Returns:
            MavenProject object with parsed structure
        """
        try:
            root = ET.fromstring(content)
            
            # Handle namespace
            ns = self._get_namespace(root)
            
            project = MavenProject(
                group_id=self._get_text(root, 'groupId', ns),
                artifact_id=self._get_text(root, 'artifactId', ns),
                version=self._get_text(root, 'version', ns),
                packaging=self._get_text(root, 'packaging', ns) or 'jar',
                name=self._get_text(root, 'name', ns),
                description=self._get_text(root, 'description', ns),
                parent=self._extract_parent(root, ns),
                modules=self._extract_modules(root, ns),
                properties=self._extract_properties(root, ns),
                dependencies=self._extract_dependencies(root, ns),
                dependency_management=self._extract_dependency_management(root, ns),
                plugins=self._extract_plugins(root, ns),
                profiles=self._extract_profiles(root, ns),
                repositories=self._extract_repositories(root, ns)
            )
            
            return project
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse POM: {e}")
            return self._empty_project()
    
    def _get_namespace(self, root: ET.Element) -> Dict[str, str]:
        """Extract namespace from root element"""
        if root.tag.startswith('{'):
            namespace = root.tag[1:root.tag.index('}')]
            return {'': namespace}
        return {}
    
    def _get_text(self, element: ET.Element, tag: str, 
                  ns: Dict[str, str]) -> Optional[str]:
        """Get text content of a child element"""
        if ns:
            tag_with_ns = f"{{{ns['']}}}{tag}"
            child = element.find(tag_with_ns)
        else:
            child = element.find(tag)
        
        return child.text if child is not None else None
    
    def _extract_parent(self, root: ET.Element, ns: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Extract parent POM information"""
        parent = root.find('parent' if not ns else f"{{{ns['']}}}parent")
        
        if parent is not None:
            return {
                'groupId': self._get_text(parent, 'groupId', ns),
                'artifactId': self._get_text(parent, 'artifactId', ns),
                'version': self._get_text(parent, 'version', ns),
                'relativePath': self._get_text(parent, 'relativePath', ns)
            }
        
        return None
    
    def _extract_modules(self, root: ET.Element, ns: Dict[str, str]) -> List[str]:
        """Extract module list"""
        modules = []
        modules_elem = root.find('modules' if not ns else f"{{{ns['']}}}modules")
        
        if modules_elem is not None:
            for module in modules_elem.findall('module' if not ns else f"{{{ns['']}}}module"):
                if module.text:
                    modules.append(module.text)
        
        return modules
    
    def _extract_properties(self, root: ET.Element, ns: Dict[str, str]) -> Dict[str, str]:
        """Extract properties"""
        properties = {}
        props_elem = root.find('properties' if not ns else f"{{{ns['']}}}properties")
        
        if props_elem is not None:
            for prop in props_elem:
                tag = prop.tag.split('}')[-1] if '}' in prop.tag else prop.tag
                if prop.text:
                    properties[tag] = prop.text
        
        return properties
    
    def _extract_dependencies(self, root: ET.Element, ns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract dependencies"""
        dependencies = []
        deps_elem = root.find('dependencies' if not ns else f"{{{ns['']}}}dependencies")
        
        if deps_elem is not None:
            for dep in deps_elem.findall('dependency' if not ns else f"{{{ns['']}}}dependency"):
                dependency = {
                    'groupId': self._get_text(dep, 'groupId', ns),
                    'artifactId': self._get_text(dep, 'artifactId', ns),
                    'version': self._get_text(dep, 'version', ns),
                    'scope': self._get_text(dep, 'scope', ns),
                    'type': self._get_text(dep, 'type', ns),
                    'classifier': self._get_text(dep, 'classifier', ns),
                    'optional': self._get_text(dep, 'optional', ns),
                }
                
                # Extract exclusions
                exclusions = []
                exclusions_elem = dep.find('exclusions' if not ns else f"{{{ns['']}}}exclusions")
                if exclusions_elem is not None:
                    for excl in exclusions_elem.findall('exclusion' if not ns else f"{{{ns['']}}}exclusion"):
                        exclusions.append({
                            'groupId': self._get_text(excl, 'groupId', ns),
                            'artifactId': self._get_text(excl, 'artifactId', ns)
                        })
                
                if exclusions:
                    dependency['exclusions'] = exclusions
                
                dependencies.append(dependency)
        
        return dependencies
    
    def _extract_dependency_management(self, root: ET.Element, 
                                      ns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract dependency management section"""
        dependencies = []
        mgmt_elem = root.find('dependencyManagement' if not ns else f"{{{ns['']}}}dependencyManagement")
        
        if mgmt_elem is not None:
            deps_elem = mgmt_elem.find('dependencies' if not ns else f"{{{ns['']}}}dependencies")
            if deps_elem is not None:
                for dep in deps_elem.findall('dependency' if not ns else f"{{{ns['']}}}dependency"):
                    dependencies.append({
                        'groupId': self._get_text(dep, 'groupId', ns),
                        'artifactId': self._get_text(dep, 'artifactId', ns),
                        'version': self._get_text(dep, 'version', ns),
                        'scope': self._get_text(dep, 'scope', ns),
                        'type': self._get_text(dep, 'type', ns)
                    })
        
        return dependencies
    
    def _extract_plugins(self, root: ET.Element, ns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract build plugins"""
        plugins = []
        build_elem = root.find('build' if not ns else f"{{{ns['']}}}build")
        
        if build_elem is not None:
            plugins_elem = build_elem.find('plugins' if not ns else f"{{{ns['']}}}plugins")
            if plugins_elem is not None:
                for plugin in plugins_elem.findall('plugin' if not ns else f"{{{ns['']}}}plugin"):
                    plugin_info = {
                        'groupId': self._get_text(plugin, 'groupId', ns) or 'org.apache.maven.plugins',
                        'artifactId': self._get_text(plugin, 'artifactId', ns),
                        'version': self._get_text(plugin, 'version', ns),
                    }
                    
                    # Extract configuration
                    config_elem = plugin.find('configuration' if not ns else f"{{{ns['']}}}configuration")
                    if config_elem is not None:
                        plugin_info['configuration'] = self._element_to_dict(config_elem)
                    
                    # Extract executions
                    executions = []
                    exec_elem = plugin.find('executions' if not ns else f"{{{ns['']}}}executions")
                    if exec_elem is not None:
                        for execution in exec_elem.findall('execution' if not ns else f"{{{ns['']}}}execution"):
                            executions.append({
                                'id': self._get_text(execution, 'id', ns),
                                'phase': self._get_text(execution, 'phase', ns),
                                'goals': [goal.text for goal in execution.findall('.//goal')]
                            })
                    
                    if executions:
                        plugin_info['executions'] = executions
                    
                    plugins.append(plugin_info)
        
        return plugins
    
    def _extract_profiles(self, root: ET.Element, ns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract profiles"""
        profiles = []
        profiles_elem = root.find('profiles' if not ns else f"{{{ns['']}}}profiles")
        
        if profiles_elem is not None:
            for profile in profiles_elem.findall('profile' if not ns else f"{{{ns['']}}}profile"):
                profile_info = {
                    'id': self._get_text(profile, 'id', ns),
                    'activation': {},
                    'properties': {},
                    'dependencies': [],
                    'build': {}
                }
                
                # Extract activation
                activation = profile.find('activation' if not ns else f"{{{ns['']}}}activation")
                if activation is not None:
                    profile_info['activation'] = self._element_to_dict(activation)
                
                # Extract profile properties
                props = profile.find('properties' if not ns else f"{{{ns['']}}}properties")
                if props is not None:
                    for prop in props:
                        tag = prop.tag.split('}')[-1] if '}' in prop.tag else prop.tag
                        if prop.text:
                            profile_info['properties'][tag] = prop.text
                
                profiles.append(profile_info)
        
        return profiles
    
    def _extract_repositories(self, root: ET.Element, ns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract repository configurations"""
        repositories = []
        repos_elem = root.find('repositories' if not ns else f"{{{ns['']}}}repositories")
        
        if repos_elem is not None:
            for repo in repos_elem.findall('repository' if not ns else f"{{{ns['']}}}repository"):
                repo_info = {
                    'id': self._get_text(repo, 'id', ns),
                    'name': self._get_text(repo, 'name', ns),
                    'url': self._get_text(repo, 'url', ns),
                    'layout': self._get_text(repo, 'layout', ns) or 'default',
                }
                
                # Extract snapshots/releases policies
                for policy_type in ['snapshots', 'releases']:
                    policy = repo.find(policy_type if not ns else f"{{{ns['']}}}{policy_type}")
                    if policy is not None:
                        repo_info[policy_type] = {
                            'enabled': self._get_text(policy, 'enabled', ns) == 'true',
                            'updatePolicy': self._get_text(policy, 'updatePolicy', ns),
                            'checksumPolicy': self._get_text(policy, 'checksumPolicy', ns)
                        }
                
                repositories.append(repo_info)
        
        return repositories
    
    def _element_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        result = {}
        
        for child in element:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            
            if len(child) > 0:
                result[tag] = self._element_to_dict(child)
            else:
                result[tag] = child.text
        
        return result
    
    def _empty_project(self) -> MavenProject:
        """Return empty Maven project"""
        return MavenProject(
            group_id=None,
            artifact_id=None,
            version=None,
            packaging=None,
            name=None,
            description=None,
            parent=None,
            modules=[],
            properties={},
            dependencies=[],
            dependency_management=[],
            plugins=[],
            profiles=[],
            repositories=[]
        )

class MavenChunker(BaseChunker):
    """Chunker specialized for Maven POM files"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, ChunkerConfig(max_tokens=max_tokens))
        self.analyzer = MavenStructureAnalyzer()
        
        # Section priorities for chunking
        self.section_priorities = {
            'project_info': 1,  # GAV coordinates, parent, packaging
            'properties': 2,
            'dependency_management': 3,
            'dependencies': 4,
            'build': 5,
            'profiles': 6,
            'repositories': 7,
            'other': 8
        }
    
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from Maven POM file
        
        Args:
            content: POM XML content
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Parse POM structure
            project = self.analyzer.analyze_pom(content)
            
            # Create chunks based on structure
            chunks = []
            
            # Add project info chunk
            project_info_chunk = self._create_project_info_chunk(content, project, file_context)
            if project_info_chunk:
                chunks.append(project_info_chunk)
            
            # Add properties chunk
            properties_chunk = self._create_properties_chunk(content, project, file_context)
            if properties_chunk:
                chunks.append(properties_chunk)
            
            # Add dependency chunks
            dependency_chunks = self._create_dependency_chunks(content, project, file_context)
            chunks.extend(dependency_chunks)
            
            # Add build configuration chunks
            build_chunks = self._create_build_chunks(content, project, file_context)
            chunks.extend(build_chunks)
            
            # Add profile chunks
            profile_chunks = self._create_profile_chunks(content, project, file_context)
            chunks.extend(profile_chunks)
            
            # Add repository chunks
            repository_chunks = self._create_repository_chunks(content, project, file_context)
            chunks.extend(repository_chunks)
            
            # If no structured chunks created, fall back to XML chunking
            if not chunks:
                chunks = self._fallback_xml_chunking(content, file_context)
            
            logger.info(f"Created {len(chunks)} chunks for Maven POM {file_context.path}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking Maven POM {file_context.path}: {e}")
            return self._fallback_xml_chunking(content, file_context)
    
    def _create_project_info_chunk(self, content: str, project: MavenProject,
                                   file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for project basic information"""
        try:
            root = ET.fromstring(content)
            project_info_parts = []
            
            # XML declaration
            project_info_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
            
            # Project opening tag with namespace
            if root.attrib:
                attrs = ' '.join(f'{k}="{v}"' for k, v in root.attrib.items())
                project_info_parts.append(f'<project {attrs}>')
            else:
                project_info_parts.append('<project>')
            
            # Model version
            project_info_parts.append('    <modelVersion>4.0.0</modelVersion>')
            
            # Parent if exists
            if project.parent:
                project_info_parts.append('    <parent>')
                project_info_parts.append(f'        <groupId>{project.parent["groupId"]}</groupId>')
                project_info_parts.append(f'        <artifactId>{project.parent["artifactId"]}</artifactId>')
                project_info_parts.append(f'        <version>{project.parent["version"]}</version>')
                if project.parent.get('relativePath'):
                    project_info_parts.append(f'        <relativePath>{project.parent["relativePath"]}</relativePath>')
                project_info_parts.append('    </parent>')
            
            # GAV coordinates
            if project.group_id:
                project_info_parts.append(f'    <groupId>{project.group_id}</groupId>')
            if project.artifact_id:
                project_info_parts.append(f'    <artifactId>{project.artifact_id}</artifactId>')
            if project.version:
                project_info_parts.append(f'    <version>{project.version}</version>')
            if project.packaging and project.packaging != 'jar':
                project_info_parts.append(f'    <packaging>{project.packaging}</packaging>')
            
            # Name and description
            if project.name:
                project_info_parts.append(f'    <name>{project.name}</name>')
            if project.description:
                desc_lines = project.description.split('\n')
                if len(desc_lines) == 1:
                    project_info_parts.append(f'    <description>{project.description}</description>')
                else:
                    project_info_parts.append('    <description>')
                    for line in desc_lines:
                        project_info_parts.append(f'        {line}')
                    project_info_parts.append('    </description>')
            
            # Modules if multi-module project
            if project.modules:
                project_info_parts.append('    <modules>')
                for module in project.modules:
                    project_info_parts.append(f'        <module>{module}</module>')
                project_info_parts.append('    </modules>')
            
            project_info_parts.append('</project>')
            
            chunk_content = '\n'.join(project_info_parts)
            
            # Check token limit
            if self.count_tokens(chunk_content) <= self.max_tokens:
                return self.create_chunk(
                    content=chunk_content,
                    chunk_type='maven_project_info',
                    metadata={
                        'section': 'project_info',
                        'groupId': project.group_id,
                        'artifactId': project.artifact_id,
                        'version': project.version,
                        'packaging': project.packaging,
                        'has_parent': project.parent is not None,
                        'has_modules': len(project.modules) > 0,
                        'module_count': len(project.modules)
                    },
                    file_path=str(file_context.path)
                )
            
        except Exception as e:
            logger.warning(f"Failed to create project info chunk: {e}")
        
        return None
    
    def _create_properties_chunk(self, content: str, project: MavenProject,
                                file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for properties section"""
        if not project.properties:
            return None
        
        try:
            chunk_parts = ['<properties>']
            
            # Group properties by category
            property_groups = self._group_properties(project.properties)
            
            for group_name, properties in property_groups.items():
                if group_name != 'other':
                    chunk_parts.append(f'    <!-- {group_name} properties -->')
                
                for key, value in sorted(properties.items()):
                    # Escape XML special characters
                    value = self._escape_xml(value)
                    chunk_parts.append(f'    <{key}>{value}</{key}>')
            
            chunk_parts.append('</properties>')
            
            chunk_content = '\n'.join(chunk_parts)
            
            # Check token limit
            if self.count_tokens(chunk_content) <= self.max_tokens:
                return self.create_chunk(
                    content=chunk_content,
                    chunk_type='maven_properties',
                    metadata={
                        'section': 'properties',
                        'property_count': len(project.properties),
                        'property_groups': list(property_groups.keys())
                    },
                    file_path=str(file_context.path)
                )
            else:
                # Split properties into multiple chunks
                return self._split_properties_chunk(project.properties, file_context)
            
        except Exception as e:
            logger.warning(f"Failed to create properties chunk: {e}")
        
        return None
    
    def _create_dependency_chunks(self, content: str, project: MavenProject,
                                 file_context: FileContext) -> List[Chunk]:
        """Create chunks for dependencies"""
        chunks = []
        
        # Dependency management chunk
        if project.dependency_management:
            mgmt_chunk = self._create_dependency_management_chunk(
                project.dependency_management, file_context
            )
            if mgmt_chunk:
                chunks.append(mgmt_chunk)
        
        # Regular dependencies
        if project.dependencies:
            # Group dependencies by scope
            scoped_deps = self._group_dependencies_by_scope(project.dependencies)
            
            for scope, deps in scoped_deps.items():
                # Try to fit all dependencies of same scope in one chunk
                scope_chunk = self._create_scoped_dependencies_chunk(
                    deps, scope, file_context
                )
                
                if scope_chunk:
                    chunks.append(scope_chunk)
                else:
                    # Split into smaller chunks if too large
                    split_chunks = self._split_dependencies_chunk(
                        deps, scope, file_context
                    )
                    chunks.extend(split_chunks)
        
        return chunks
    
    def _create_dependency_management_chunk(self, dependencies: List[Dict[str, Any]],
                                           file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for dependency management section"""
        chunk_parts = ['<dependencyManagement>', '    <dependencies>']
        
        for dep in dependencies:
            chunk_parts.append('        <dependency>')
            if dep.get('groupId'):
                chunk_parts.append(f'            <groupId>{dep["groupId"]}</groupId>')
            if dep.get('artifactId'):
                chunk_parts.append(f'            <artifactId>{dep["artifactId"]}</artifactId>')
            if dep.get('version'):
                chunk_parts.append(f'            <version>{dep["version"]}</version>')
            if dep.get('scope'):
                chunk_parts.append(f'            <scope>{dep["scope"]}</scope>')
            if dep.get('type') and dep['type'] != 'jar':
                chunk_parts.append(f'            <type>{dep["type"]}</type>')
            chunk_parts.append('        </dependency>')
        
        chunk_parts.extend(['    </dependencies>', '</dependencyManagement>'])
        
        chunk_content = '\n'.join(chunk_parts)
        
        if self.count_tokens(chunk_content) <= self.max_tokens:
            return self.create_chunk(
                content=chunk_content,
                chunk_type='maven_dependency_management',
                metadata={
                    'section': 'dependency_management',
                    'dependency_count': len(dependencies)
                },
                file_path=str(file_context.path)
            )
        
        return None
    
    def _create_scoped_dependencies_chunk(self, dependencies: List[Dict[str, Any]],
                                         scope: str, file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for dependencies of specific scope"""
        chunk_parts = [f'<!-- Dependencies with scope: {scope} -->', '<dependencies>']
        
        for dep in dependencies:
            chunk_parts.append('    <dependency>')
            chunk_parts.append(f'        <groupId>{dep["groupId"]}</groupId>')
            chunk_parts.append(f'        <artifactId>{dep["artifactId"]}</artifactId>')
            
            if dep.get('version'):
                chunk_parts.append(f'        <version>{dep["version"]}</version>')
            if dep.get('scope') and dep['scope'] != 'compile':
                chunk_parts.append(f'        <scope>{dep["scope"]}</scope>')
            if dep.get('type') and dep['type'] != 'jar':
                chunk_parts.append(f'        <type>{dep["type"]}</type>')
            if dep.get('classifier'):
                chunk_parts.append(f'        <classifier>{dep["classifier"]}</classifier>')
            if dep.get('optional'):
                chunk_parts.append(f'        <optional>{dep["optional"]}</optional>')
            
            # Add exclusions if present
            if dep.get('exclusions'):
                chunk_parts.append('        <exclusions>')
                for exclusion in dep['exclusions']:
                    chunk_parts.append('            <exclusion>')
                    chunk_parts.append(f'                <groupId>{exclusion["groupId"]}</groupId>')
                    chunk_parts.append(f'                <artifactId>{exclusion["artifactId"]}</artifactId>')
                    chunk_parts.append('            </exclusion>')
                chunk_parts.append('        </exclusions>')
            
            chunk_parts.append('    </dependency>')
        
        chunk_parts.append('</dependencies>')
        
        chunk_content = '\n'.join(chunk_parts)
        
        if self.count_tokens(chunk_content) <= self.max_tokens:
            return self.create_chunk(
                content=chunk_content,
                chunk_type='maven_dependencies',
                metadata={
                    'section': 'dependencies',
                    'scope': scope,
                    'dependency_count': len(dependencies)
                },
                file_path=str(file_context.path)
            )
        
        return None
    
    def _create_build_chunks(self, content: str, project: MavenProject,
                           file_context: FileContext) -> List[Chunk]:
        """Create chunks for build configuration"""
        chunks = []
        
        if not project.plugins:
            return chunks
        
        # Group plugins by type/purpose
        plugin_groups = self._group_plugins(project.plugins)
        
        for group_name, plugins in plugin_groups.items():
            chunk_parts = ['<build>', '    <plugins>']
            chunk_parts.append(f'        <!-- {group_name} plugins -->')
            
            for plugin in plugins:
                chunk_parts.append('        <plugin>')
                
                if plugin.get('groupId'):
                    chunk_parts.append(f'            <groupId>{plugin["groupId"]}</groupId>')
                chunk_parts.append(f'            <artifactId>{plugin["artifactId"]}</artifactId>')
                if plugin.get('version'):
                    chunk_parts.append(f'            <version>{plugin["version"]}</version>')
                
                # Add configuration if present and not too complex
                if plugin.get('configuration'):
                    config_xml = self._dict_to_xml(plugin['configuration'], indent=12)
                    if config_xml and len(config_xml) < 50:  # Limit configuration lines
                        chunk_parts.append('            <configuration>')
                        chunk_parts.extend(config_xml)
                        chunk_parts.append('            </configuration>')
                
                # Add executions
                if plugin.get('executions'):
                    chunk_parts.append('            <executions>')
                    for execution in plugin['executions']:
                        chunk_parts.append('                <execution>')
                        if execution.get('id'):
                            chunk_parts.append(f'                    <id>{execution["id"]}</id>')
                        if execution.get('phase'):
                            chunk_parts.append(f'                    <phase>{execution["phase"]}</phase>')
                        if execution.get('goals'):
                            chunk_parts.append('                    <goals>')
                            for goal in execution['goals']:
                                chunk_parts.append(f'                        <goal>{goal}</goal>')
                            chunk_parts.append('                    </goals>')
                        chunk_parts.append('                </execution>')
                    chunk_parts.append('            </executions>')
                
                chunk_parts.append('        </plugin>')
            
            chunk_parts.extend(['    </plugins>', '</build>'])
            
            chunk_content = '\n'.join(chunk_parts)
            
            if self.count_tokens(chunk_content) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='maven_build',
                    metadata={
                        'section': 'build',
                        'plugin_group': group_name,
                        'plugin_count': len(plugins),
                        'plugins': [p['artifactId'] for p in plugins]
                    },
                    file_path=str(file_context.path)
                ))
            else:
                # Split plugins into individual chunks
                for plugin in plugins:
                    plugin_chunk = self._create_single_plugin_chunk(plugin, file_context)
                    if plugin_chunk:
                        chunks.append(plugin_chunk)
        
        return chunks
    
    def _create_single_plugin_chunk(self, plugin: Dict[str, Any],
                                   file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for a single plugin"""
        chunk_parts = ['<plugin>']
        
        if plugin.get('groupId'):
            chunk_parts.append(f'    <groupId>{plugin["groupId"]}</groupId>')
        chunk_parts.append(f'    <artifactId>{plugin["artifactId"]}</artifactId>')
        if plugin.get('version'):
            chunk_parts.append(f'    <version>{plugin["version"]}</version>')
        
        if plugin.get('configuration'):
            chunk_parts.append('    <configuration>')
            config_xml = self._dict_to_xml(plugin['configuration'], indent=8)
            chunk_parts.extend(config_xml)
            chunk_parts.append('    </configuration>')
        
        if plugin.get('executions'):
            chunk_parts.append('    <executions>')
            for execution in plugin['executions']:
                chunk_parts.append('        <execution>')
                if execution.get('id'):
                    chunk_parts.append(f'            <id>{execution["id"]}</id>')
                if execution.get('phase'):
                    chunk_parts.append(f'            <phase>{execution["phase"]}</phase>')
                if execution.get('goals'):
                    chunk_parts.append('            <goals>')
                    for goal in execution['goals']:
                        chunk_parts.append(f'                <goal>{goal}</goal>')
                    chunk_parts.append('            </goals>')
                chunk_parts.append('        </execution>')
            chunk_parts.append('    </executions>')
        
        chunk_parts.append('</plugin>')
        
        chunk_content = '\n'.join(chunk_parts)
        
        if self.count_tokens(chunk_content) <= self.max_tokens:
            return self.create_chunk(
                content=chunk_content,
                chunk_type='maven_plugin',
                metadata={
                    'section': 'build_plugin',
                    'plugin_id': f"{plugin.get('groupId', 'org.apache.maven.plugins')}:{plugin['artifactId']}",
                    'has_configuration': plugin.get('configuration') is not None,
                    'execution_count': len(plugin.get('executions', []))
                },
                file_path=str(file_context.path)
            )
        
        return None
    
    def _create_profile_chunks(self, content: str, project: MavenProject,
                              file_context: FileContext) -> List[Chunk]:
        """Create chunks for profiles"""
        chunks = []
        
        for profile in project.profiles:
            chunk_parts = ['<profile>']
            
            if profile.get('id'):
                chunk_parts.append(f'    <id>{profile["id"]}</id>')
            
            # Add activation
            if profile.get('activation'):
                chunk_parts.append('    <activation>')
                activation_xml = self._dict_to_xml(profile['activation'], indent=8)
                chunk_parts.extend(activation_xml)
                chunk_parts.append('    </activation>')
            
            # Add properties
            if profile.get('properties'):
                chunk_parts.append('    <properties>')
                for key, value in profile['properties'].items():
                    chunk_parts.append(f'        <{key}>{self._escape_xml(value)}</{key}>')
                chunk_parts.append('    </properties>')
            
            chunk_parts.append('</profile>')
            
            chunk_content = '\n'.join(chunk_parts)
            
            if self.count_tokens(chunk_content) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=chunk_content,
                    chunk_type='maven_profile',
                    metadata={
                        'section': 'profile',
                        'profile_id': profile.get('id', 'default'),
                        'has_activation': profile.get('activation') is not None,
                        'property_count': len(profile.get('properties', {}))
                    },
                    file_path=str(file_context.path)
                ))
        
        return chunks
    
    def _create_repository_chunks(self, content: str, project: MavenProject,
                                 file_context: FileContext) -> List[Chunk]:
        """Create chunks for repository configurations"""
        if not project.repositories:
            return []
        
        chunk_parts = ['<repositories>']
        
        for repo in project.repositories:
            chunk_parts.append('    <repository>')
            chunk_parts.append(f'        <id>{repo["id"]}</id>')
            if repo.get('name'):
                chunk_parts.append(f'        <name>{repo["name"]}</name>')
            chunk_parts.append(f'        <url>{repo["url"]}</url>')
            
            if repo.get('layout') and repo['layout'] != 'default':
                chunk_parts.append(f'        <layout>{repo["layout"]}</layout>')
            
            # Add snapshot/release policies
            for policy_type in ['snapshots', 'releases']:
                if repo.get(policy_type):
                    policy = repo[policy_type]
                    chunk_parts.append(f'        <{policy_type}>')
                    chunk_parts.append(f'            <enabled>{str(policy.get("enabled", True)).lower()}</enabled>')
                    if policy.get('updatePolicy'):
                        chunk_parts.append(f'            <updatePolicy>{policy["updatePolicy"]}</updatePolicy>')
                    if policy.get('checksumPolicy'):
                        chunk_parts.append(f'            <checksumPolicy>{policy["checksumPolicy"]}</checksumPolicy>')
                    chunk_parts.append(f'        </{policy_type}>')
            
            chunk_parts.append('    </repository>')
        
        chunk_parts.append('</repositories>')
        
        chunk_content = '\n'.join(chunk_parts)
        
        if self.count_tokens(chunk_content) <= self.max_tokens:
            return [self.create_chunk(
                content=chunk_content,
                chunk_type='maven_repositories',
                metadata={
                    'section': 'repositories',
                    'repository_count': len(project.repositories),
                    'repository_ids': [r['id'] for r in project.repositories]
                },
                file_path=str(file_context.path)
            )]
        
        return []
    
    def _group_properties(self, properties: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Group properties by category"""
        groups = defaultdict(dict)
        
        for key, value in properties.items():
            # Categorize based on property name patterns
            if 'version' in key.lower():
                groups['versions'][key] = value
            elif 'encoding' in key.lower() or 'charset' in key.lower():
                groups['encoding'][key] = value
            elif key.startswith('maven.') or key.startswith('project.'):
                groups['maven'][key] = value
            elif key.startswith('spring.'):
                groups['spring'][key] = value
            elif key.startswith('java.'):
                groups['java'][key] = value
            else:
                groups['other'][key] = value
        
        return dict(groups)
    
    def _group_dependencies_by_scope(self, dependencies: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group dependencies by scope"""
        groups = defaultdict(list)
        
        for dep in dependencies:
            scope = dep.get('scope', 'compile')
            groups[scope].append(dep)
        
        return dict(groups)
    
    def _group_plugins(self, plugins: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group plugins by type/purpose"""
        groups = defaultdict(list)
        
        for plugin in plugins:
            artifact_id = plugin.get('artifactId', '')
            
            # Categorize based on plugin artifact ID
            if 'compiler' in artifact_id or 'javac' in artifact_id:
                groups['compilation'].append(plugin)
            elif 'test' in artifact_id or 'surefire' in artifact_id or 'failsafe' in artifact_id:
                groups['testing'].append(plugin)
            elif 'package' in artifact_id or 'jar' in artifact_id or 'war' in artifact_id:
                groups['packaging'].append(plugin)
            elif 'deploy' in artifact_id or 'release' in artifact_id:
                groups['deployment'].append(plugin)
            elif 'clean' in artifact_id:
                groups['lifecycle'].append(plugin)
            elif 'spring' in artifact_id:
                groups['spring'].append(plugin)
            else:
                groups['other'].append(plugin)
        
        return dict(groups)
    
    def _split_properties_chunk(self, properties: Dict[str, str],
                               file_context: FileContext) -> Optional[Chunk]:
        """Split large properties section into smaller chunk"""
        # Simple approach: take first N properties that fit
        chunk_parts = ['<properties>']
        included_props = {}
        
        for key, value in sorted(properties.items()):
            test_line = f'    <{key}>{self._escape_xml(value)}</{key}>'
            test_content = '\n'.join(chunk_parts + [test_line, '</properties>'])
            
            if self.count_tokens(test_content) <= self.max_tokens:
                chunk_parts.append(test_line)
                included_props[key] = value
            else:
                break
        
        chunk_parts.append('</properties>')
        
        if included_props:
            return self.create_chunk(
                content='\n'.join(chunk_parts),
                chunk_type='maven_properties_partial',
                metadata={
                    'section': 'properties',
                    'property_count': len(included_props),
                    'is_partial': True
                },
                file_path=str(file_context.path)
            )
        
        return None
    
    def _split_dependencies_chunk(self, dependencies: List[Dict[str, Any]],
                                 scope: str, file_context: FileContext) -> List[Chunk]:
        """Split dependencies into multiple chunks"""
        chunks = []
        current_deps = []
        current_chunk_parts = [f'<!-- Dependencies with scope: {scope} (part) -->', '<dependencies>']
        
        for dep in dependencies:
            dep_lines = self._dependency_to_xml_lines(dep)
            test_content = '\n'.join(current_chunk_parts + dep_lines + ['</dependencies>'])
            
            if self.count_tokens(test_content) <= self.max_tokens:
                current_chunk_parts.extend(dep_lines)
                current_deps.append(dep)
            else:
                # Save current chunk
                if current_deps:
                    current_chunk_parts.append('</dependencies>')
                    chunks.append(self.create_chunk(
                        content='\n'.join(current_chunk_parts),
                        chunk_type='maven_dependencies_partial',
                        metadata={
                            'section': 'dependencies',
                            'scope': scope,
                            'dependency_count': len(current_deps),
                            'is_partial': True,
                            'part_index': len(chunks)
                        },
                        file_path=str(file_context.path)
                    ))
                
                # Start new chunk
                current_deps = [dep]
                current_chunk_parts = [f'<!-- Dependencies with scope: {scope} (continued) -->', '<dependencies>']
                current_chunk_parts.extend(dep_lines)
        
        # Add remaining dependencies
        if current_deps:
            current_chunk_parts.append('</dependencies>')
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk_parts),
                chunk_type='maven_dependencies_partial',
                metadata={
                    'section': 'dependencies',
                    'scope': scope,
                    'dependency_count': len(current_deps),
                    'is_partial': True,
                    'part_index': len(chunks)
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _dependency_to_xml_lines(self, dep: Dict[str, Any]) -> List[str]:
        """Convert dependency dict to XML lines"""
        lines = ['    <dependency>']
        lines.append(f'        <groupId>{dep["groupId"]}</groupId>')
        lines.append(f'        <artifactId>{dep["artifactId"]}</artifactId>')
        
        if dep.get('version'):
            lines.append(f'        <version>{dep["version"]}</version>')
        if dep.get('scope') and dep['scope'] != 'compile':
            lines.append(f'        <scope>{dep["scope"]}</scope>')
        if dep.get('type') and dep['type'] != 'jar':
            lines.append(f'        <type>{dep["type"]}</type>')
        if dep.get('classifier'):
            lines.append(f'        <classifier>{dep["classifier"]}</classifier>')
        if dep.get('optional'):
            lines.append(f'        <optional>{dep["optional"]}</optional>')
        
        if dep.get('exclusions'):
            lines.append('        <exclusions>')
            for exclusion in dep['exclusions']:
                lines.append('            <exclusion>')
                lines.append(f'                <groupId>{exclusion["groupId"]}</groupId>')
                lines.append(f'                <artifactId>{exclusion["artifactId"]}</artifactId>')
                lines.append('            </exclusion>')
            lines.append('        </exclusions>')
        
        lines.append('    </dependency>')
        
        return lines
    
    def _dict_to_xml(self, data: Dict[str, Any], indent: int = 0) -> List[str]:
        """Convert dictionary to XML lines"""
        lines = []
        indent_str = ' ' * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f'{indent_str}<{key}>')
                lines.extend(self._dict_to_xml(value, indent + 4))
                lines.append(f'{indent_str}</{key}>')
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f'{indent_str}<{key}>')
                        lines.extend(self._dict_to_xml(item, indent + 4))
                        lines.append(f'{indent_str}</{key}>')
                    else:
                        lines.append(f'{indent_str}<{key}>{self._escape_xml(str(item))}</{key}>')
            else:
                lines.append(f'{indent_str}<{key}>{self._escape_xml(str(value))}</{key}>')
        
        return lines
    
    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters"""
        if not text:
            return text
        
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&apos;'
        }
        
        for char, escape in replacements.items():
            text = text.replace(char, escape)
        
        return text
    
    def _fallback_xml_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback to generic XML chunking"""
        logger.warning(f"Using fallback XML chunking for {file_context.path}")
        
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_start = 0
        
        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='maven_xml_fallback',
                    metadata={
                        'section': 'unknown',
                        'is_fallback': True
                    },
                    file_path=str(file_context.path),
                    start_line=chunk_start,
                    end_line=i
                ))
                current_chunk = []
                current_tokens = 0
                chunk_start = i
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='maven_xml_fallback',
                metadata={
                    'section': 'unknown',
                    'is_fallback': True
                },
                file_path=str(file_context.path),
                start_line=chunk_start,
                end_line=len(lines)
            ))
        
        return chunks