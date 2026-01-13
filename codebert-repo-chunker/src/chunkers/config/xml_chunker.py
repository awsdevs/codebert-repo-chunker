"""
XML-specific chunker for intelligent semantic chunking
Handles complex XML documents, schemas, SOAP, RSS, SVG, and various XML formats
"""

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from lxml import etree
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from enum import Enum
import html

from src.core.base_chunker import BaseChunker, Chunk
from src.core.file_context import FileContext
from config.settings import settings

logger = logging.getLogger(__name__)

class XMLFormat(Enum):
    """Types of XML formats"""
    GENERIC = "generic"
    XHTML = "xhtml"
    SVG = "svg"
    RSS = "rss"
    ATOM = "atom"
    SOAP = "soap"
    WSDL = "wsdl"
    XSD = "xsd"
    XSLT = "xslt"
    MAVEN_POM = "maven_pom"
    ANT_BUILD = "ant_build"
    SPRING_CONFIG = "spring_config"
    WEB_XML = "web_xml"
    ANDROID_MANIFEST = "android_manifest"
    ANDROID_LAYOUT = "android_layout"
    DOCBOOK = "docbook"
    OFFICE_XML = "office_xml"
    XAML = "xaml"
    GRAPHML = "graphml"
    MATHML = "mathml"
    MUSICXML = "musicxml"

@dataclass
class XMLNode:
    """Represents an XML node"""
    tag: str
    attrib: Dict[str, str]
    text: Optional[str]
    tail: Optional[str]
    namespace: Optional[str]
    prefix: Optional[str]
    path: str
    xpath: str
    depth: int
    position: int
    parent: Optional['XMLNode']
    children: List['XMLNode']
    size: int  # Size in characters
    token_count: int
    line_start: int
    line_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class XMLNamespace:
    """Represents an XML namespace"""
    prefix: str
    uri: str
    is_default: bool
    usage_count: int
    elements: Set[str]

@dataclass
class XMLStructure:
    """Represents overall XML structure"""
    format: XMLFormat
    root_tag: str
    namespaces: Dict[str, XMLNamespace]
    total_elements: int
    total_attributes: int
    max_depth: int
    has_mixed_content: bool
    has_cdata: bool
    has_comments: bool
    has_processing_instructions: bool
    encoding: str
    version: str
    doctype: Optional[str]
    schema_location: Optional[str]
    validation_errors: List[str]
    statistics: Dict[str, Any]

class XMLAnalyzer:
    """Analyzes XML structure for intelligent chunking"""
    
    # Namespace URIs for common formats
    NAMESPACE_FORMATS = {
        'http://www.w3.org/1999/xhtml': XMLFormat.XHTML,
        'http://www.w3.org/2000/svg': XMLFormat.SVG,
        'http://www.w3.org/2005/Atom': XMLFormat.ATOM,
        'http://schemas.xmlsoap.org/soap/envelope/': XMLFormat.SOAP,
        'http://schemas.xmlsoap.org/wsdl/': XMLFormat.WSDL,
        'http://www.w3.org/2001/XMLSchema': XMLFormat.XSD,
        'http://www.w3.org/1999/XSL/Transform': XMLFormat.XSLT,
        'http://maven.apache.org/POM/4.0.0': XMLFormat.MAVEN_POM,
        'http://www.springframework.org/schema/beans': XMLFormat.SPRING_CONFIG,
        'http://schemas.android.com/apk/res/android': XMLFormat.ANDROID_LAYOUT,
        'http://docbook.org/ns/docbook': XMLFormat.DOCBOOK,
        'http://schemas.microsoft.com/winfx/2006/xaml': XMLFormat.XAML,
        'http://graphml.graphdrawing.org/xmlns': XMLFormat.GRAPHML,
        'http://www.w3.org/1998/Math/MathML': XMLFormat.MATHML,
    }
    
    # Root element format detection
    ROOT_ELEMENT_FORMATS = {
        'project': XMLFormat.MAVEN_POM,
        'build': XMLFormat.ANT_BUILD,
        'beans': XMLFormat.SPRING_CONFIG,
        'web-app': XMLFormat.WEB_XML,
        'manifest': XMLFormat.ANDROID_MANIFEST,
        'svg': XMLFormat.SVG,
        'rss': XMLFormat.RSS,
        'feed': XMLFormat.ATOM,
        'envelope': XMLFormat.SOAP,
        'definitions': XMLFormat.WSDL,
        'schema': XMLFormat.XSD,
        'stylesheet': XMLFormat.XSLT,
        'transform': XMLFormat.XSLT,
        'html': XMLFormat.XHTML,
        'article': XMLFormat.DOCBOOK,
        'book': XMLFormat.DOCBOOK,
    }
    
    # Special elements that should be kept together
    ATOMIC_ELEMENTS = {
        # Maven POM
        'dependency', 'plugin', 'repository', 'profile',
        # Spring
        'bean', 'property', 'constructor-arg',
        # Android
        'activity', 'service', 'permission', 'intent-filter',
        # SOAP
        'operation', 'message', 'portType',
        # General
        'entry', 'item', 'record', 'row'
    }
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.namespaces = {}
        self.line_map = {}
        self.nodes = []
        
    def analyze_xml(self, content: str, file_path: Optional[Path] = None) -> XMLStructure:
        """
        Analyze XML structure
        
        Args:
            content: XML content as string
            file_path: Optional file path for format detection
            
        Returns:
            XMLStructure analysis
        """
        try:
            # Parse XML with lxml for better namespace handling
            parser = etree.XMLParser(
                remove_blank_text=False,
                resolve_entities=False,
                strip_cdata=False,
                remove_comments=False
            )
            
            # Parse document
            doc = etree.fromstring(content.encode('utf-8'), parser)
            
            # Build line map for position tracking
            self._build_line_map(content)
            
            # Extract basic info
            encoding = self._detect_encoding(content)
            version = self._detect_version(content)
            doctype = self._extract_doctype(content)
            
            # Detect format
            xml_format = self._detect_format(doc, file_path)
            
            # Extract namespaces
            namespaces = self._extract_namespaces(doc)
            
            # Build node tree
            root_node = self._build_node_tree(doc, None, "", "/", 0, 0)
            
            # Analyze structure
            structure = XMLStructure(
                format=xml_format,
                root_tag=doc.tag,
                namespaces=namespaces,
                total_elements=self._count_elements(root_node),
                total_attributes=self._count_attributes(root_node),
                max_depth=self._calculate_max_depth(root_node),
                has_mixed_content=self._has_mixed_content(doc),
                has_cdata=self._has_cdata(content),
                has_comments=self._has_comments(content),
                has_processing_instructions=self._has_processing_instructions(content),
                encoding=encoding,
                version=version,
                doctype=doctype,
                schema_location=self._extract_schema_location(doc),
                validation_errors=[],
                statistics=self._calculate_statistics(root_node)
            )
            
            # Validate if schema is present
            if structure.schema_location:
                structure.validation_errors = self._validate_against_schema(doc, structure.schema_location)
            
            # Store for chunking
            self.root_node = root_node
            self.structure = structure
            
            return structure
            
        except etree.XMLSyntaxError as e:
            logger.error(f"XML syntax error: {e}")
            return self._create_error_structure(str(e))
        except Exception as e:
            logger.error(f"Error analyzing XML: {e}")
            return self._create_error_structure(str(e))
    
    def _build_line_map(self, content: str):
        """Build map of line positions"""
        self.line_map = {}
        lines = content.split('\n')
        position = 0
        
        for i, line in enumerate(lines, 1):
            self.line_map[i] = position
            position += len(line) + 1
    
    def _detect_encoding(self, content: str) -> str:
        """Detect XML encoding"""
        match = re.search(r'<\?xml[^>]+encoding=["\']([^"\']+)["\']', content)
        return match.group(1) if match else 'utf-8'
    
    def _detect_version(self, content: str) -> str:
        """Detect XML version"""
        match = re.search(r'<\?xml[^>]+version=["\']([^"\']+)["\']', content)
        return match.group(1) if match else '1.0'
    
    def _extract_doctype(self, content: str) -> Optional[str]:
        """Extract DOCTYPE declaration"""
        match = re.search(r'<!DOCTYPE[^>]+>', content, re.DOTALL)
        return match.group(0) if match else None
    
    def _detect_format(self, doc: etree.Element, file_path: Optional[Path]) -> XMLFormat:
        """Detect XML format"""
        # Check file name hints
        if file_path:
            name = file_path.name.lower()
            if name == 'pom.xml':
                return XMLFormat.MAVEN_POM
            elif name == 'build.xml':
                return XMLFormat.ANT_BUILD
            elif name == 'web.xml':
                return XMLFormat.WEB_XML
            elif name == 'androidmanifest.xml':
                return XMLFormat.ANDROID_MANIFEST
            elif name.endswith('.xsd'):
                return XMLFormat.XSD
            elif name.endswith('.xslt') or name.endswith('.xsl'):
                return XMLFormat.XSLT
            elif name.endswith('.wsdl'):
                return XMLFormat.WSDL
            elif name.endswith('.svg'):
                return XMLFormat.SVG
        
        # Check namespaces
        nsmap = doc.nsmap if hasattr(doc, 'nsmap') else {}
        for uri in nsmap.values():
            if uri in self.NAMESPACE_FORMATS:
                return self.NAMESPACE_FORMATS[uri]
        
        # Check root element
        root_tag = etree.QName(doc.tag).localname
        if root_tag in self.ROOT_ELEMENT_FORMATS:
            return self.ROOT_ELEMENT_FORMATS[root_tag]
        
        return XMLFormat.GENERIC
    
    def _extract_namespaces(self, doc: etree.Element) -> Dict[str, XMLNamespace]:
        """Extract all namespaces"""
        namespaces = {}
        
        # Get namespace map
        nsmap = doc.nsmap if hasattr(doc, 'nsmap') else {}
        
        for prefix, uri in nsmap.items():
            ns = XMLNamespace(
                prefix=prefix or 'default',
                uri=uri,
                is_default=prefix is None,
                usage_count=0,
                elements=set()
            )
            namespaces[uri] = ns
        
        # Count namespace usage
        for elem in doc.iter():
            if elem.tag.startswith('{'):
                uri = elem.tag[1:elem.tag.index('}')]
                if uri in namespaces:
                    namespaces[uri].usage_count += 1
                    namespaces[uri].elements.add(etree.QName(elem.tag).localname)
        
        return namespaces
    
    def _build_node_tree(self, elem: etree.Element, parent: Optional[XMLNode],
                        path: str, xpath: str, depth: int, position: int) -> XMLNode:
        """Build tree of XML nodes"""
        # Get tag info
        qname = etree.QName(elem.tag)
        namespace = qname.namespace
        localname = qname.localname
        prefix = elem.prefix
        
        # Calculate size and tokens
        elem_str = etree.tostring(elem, encoding='unicode')
        size = len(elem_str)
        token_count = self._count_tokens(elem_str) if self.tokenizer else size // 4
        
        # Get position info
        line_start = elem.sourceline if hasattr(elem, 'sourceline') else 0
        
        # Create node
        node = XMLNode(
            tag=localname,
            attrib=dict(elem.attrib),
            text=elem.text,
            tail=elem.tail,
            namespace=namespace,
            prefix=prefix,
            path=f"{path}/{localname}" if path else localname,
            xpath=f"{xpath}/{localname}[{position}]",
            depth=depth,
            position=position,
            parent=parent,
            children=[],
            size=size,
            token_count=token_count,
            line_start=line_start,
            line_end=0,  # Will be updated
            metadata={
                'element_count': len(list(elem)),
                'has_text': bool(elem.text and elem.text.strip()),
                'has_attributes': bool(elem.attrib),
                'is_empty': len(list(elem)) == 0 and not (elem.text and elem.text.strip())
            }
        )
        
        # Process children
        child_positions = defaultdict(int)
        for child_elem in elem:
            if isinstance(child_elem.tag, str):  # Skip comments and PIs
                child_tag = etree.QName(child_elem.tag).localname
                child_positions[child_tag] += 1
                
                child_node = self._build_node_tree(
                    child_elem, 
                    node,
                    node.path,
                    node.xpath,
                    depth + 1,
                    child_positions[child_tag]
                )
                node.children.append(child_node)
        
        # Update end line
        if node.children:
            node.line_end = max(child.line_end for child in node.children)
        else:
            node.line_end = node.line_start
        
        self.nodes.append(node)
        return node
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        # Fallback estimation
        return len(text.split()) + text.count('<') + text.count('>')
    
    def _count_elements(self, node: XMLNode) -> int:
        """Count total elements"""
        return 1 + sum(self._count_elements(child) for child in node.children)
    
    def _count_attributes(self, node: XMLNode) -> int:
        """Count total attributes"""
        count = len(node.attrib)
        return count + sum(self._count_attributes(child) for child in node.children)
    
    def _calculate_max_depth(self, node: XMLNode) -> int:
        """Calculate maximum depth"""
        if not node.children:
            return node.depth
        return max(self._calculate_max_depth(child) for child in node.children)
    
    def _has_mixed_content(self, elem: etree.Element) -> bool:
        """Check if document has mixed content (text and elements)"""
        for element in elem.iter():
            if element.text and element.text.strip() and len(list(element)) > 0:
                return True
        return False
    
    def _has_cdata(self, content: str) -> bool:
        """Check if document has CDATA sections"""
        return '<![CDATA[' in content
    
    def _has_comments(self, content: str) -> bool:
        """Check if document has comments"""
        return '<!--' in content
    
    def _has_processing_instructions(self, content: str) -> bool:
        """Check if document has processing instructions"""
        return '<?xml-' in content or re.search(r'<\?(?!xml\s)', content) is not None
    
    def _extract_schema_location(self, doc: etree.Element) -> Optional[str]:
        """Extract schema location"""
        # Check xsi:schemaLocation
        xsi_ns = 'http://www.w3.org/2001/XMLSchema-instance'
        schema_loc = doc.get(f'{{{xsi_ns}}}schemaLocation')
        if schema_loc:
            return schema_loc
        
        # Check xsi:noNamespaceSchemaLocation
        no_ns_schema = doc.get(f'{{{xsi_ns}}}noNamespaceSchemaLocation')
        return no_ns_schema
    
    def _validate_against_schema(self, doc: etree.Element, schema_location: str) -> List[str]:
        """Validate XML against schema"""
        errors = []
        # Schema validation would be implemented here
        # This is a placeholder
        return errors
    
    def _calculate_statistics(self, node: XMLNode) -> Dict[str, Any]:
        """Calculate statistics about XML structure"""
        stats = {
            'unique_tags': set(),
            'tag_frequency': defaultdict(int),
            'max_children': 0,
            'avg_children': 0,
            'leaf_nodes': 0,
            'text_nodes': 0,
            'empty_elements': 0,
            'max_attributes': 0,
            'common_attributes': defaultdict(int)
        }
        
        def analyze(n: XMLNode):
            stats['unique_tags'].add(n.tag)
            stats['tag_frequency'][n.tag] += 1
            
            if not n.children:
                stats['leaf_nodes'] += 1
            else:
                stats['max_children'] = max(stats['max_children'], len(n.children))
            
            if n.metadata.get('has_text'):
                stats['text_nodes'] += 1
            
            if n.metadata.get('is_empty'):
                stats['empty_elements'] += 1
            
            stats['max_attributes'] = max(stats['max_attributes'], len(n.attrib))
            
            for attr in n.attrib:
                stats['common_attributes'][attr] += 1
            
            for child in n.children:
                analyze(child)
        
        analyze(node)
        
        # Convert sets to lists for serialization
        stats['unique_tags'] = list(stats['unique_tags'])
        stats['tag_frequency'] = dict(stats['tag_frequency'])
        stats['common_attributes'] = dict(stats['common_attributes'])
        
        # Calculate average children
        total_parents = sum(1 for n in self.nodes if n.children)
        total_children = sum(len(n.children) for n in self.nodes if n.children)
        stats['avg_children'] = total_children / total_parents if total_parents else 0
        
        return stats
    
    def _create_error_structure(self, error_msg: str) -> XMLStructure:
        """Create error structure for invalid XML"""
        return XMLStructure(
            format=XMLFormat.GENERIC,
            root_tag='error',
            namespaces={},
            total_elements=0,
            total_attributes=0,
            max_depth=0,
            has_mixed_content=False,
            has_cdata=False,
            has_comments=False,
            has_processing_instructions=False,
            encoding='utf-8',
            version='1.0',
            doctype=None,
            schema_location=None,
            validation_errors=[error_msg],
            statistics={}
        )

class XMLChunkingStrategy:
    """Base class for XML chunking strategies"""
    
    def chunk(self, node: XMLNode, max_tokens: int, format_type: XMLFormat) -> List[Dict[str, Any]]:
        """Create chunks from XML node"""
        raise NotImplementedError

class DepthBasedStrategy(XMLChunkingStrategy):
    """Chunk XML by depth levels"""
    
    def chunk(self, node: XMLNode, max_tokens: int, format_type: XMLFormat) -> List[Dict[str, Any]]:
        """Chunk by depth"""
        chunks = []
        
        # If node fits in one chunk
        if node.token_count <= max_tokens:
            return [self._node_to_chunk(node)]
        
        # Group children by depth level
        depth_groups = defaultdict(list)
        self._group_by_depth(node, depth_groups)
        
        # Create chunks for each depth level
        for depth, nodes in sorted(depth_groups.items()):
            chunk_data = {
                'depth': depth,
                'nodes': [self._serialize_node(n) for n in nodes],
                'path': f"depth_{depth}"
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _group_by_depth(self, node: XMLNode, groups: Dict[int, List[XMLNode]]):
        """Group nodes by depth"""
        groups[node.depth].append(node)
        for child in node.children:
            self._group_by_depth(child, groups)
    
    def _node_to_chunk(self, node: XMLNode) -> Dict[str, Any]:
        """Convert node to chunk data"""
        return {
            'path': node.path,
            'xpath': node.xpath,
            'content': self._serialize_node(node),
            'metadata': node.metadata
        }
    
    def _serialize_node(self, node: XMLNode) -> str:
        """Serialize node to XML string"""
        # Simplified serialization
        parts = [f"<{node.tag}"]
        
        for key, value in node.attrib.items():
            parts.append(f' {key}="{html.escape(value)}"')
        
        if not node.children and not node.text:
            parts.append("/>")
        else:
            parts.append(">")
            if node.text:
                parts.append(html.escape(node.text))
            parts.append(f"</{node.tag}>")
        
        return ''.join(parts)

class PathBasedStrategy(XMLChunkingStrategy):
    """Chunk XML by element paths"""
    
    def chunk(self, node: XMLNode, max_tokens: int, format_type: XMLFormat) -> List[Dict[str, Any]]:
        """Chunk by paths"""
        chunks = []
        
        # Special handling for specific formats
        if format_type == XMLFormat.MAVEN_POM:
            return self._chunk_maven_pom(node, max_tokens)
        elif format_type == XMLFormat.SPRING_CONFIG:
            return self._chunk_spring_config(node, max_tokens)
        
        # Generic path-based chunking
        return self._chunk_by_path(node, max_tokens)
    
    def _chunk_maven_pom(self, node: XMLNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Special chunking for Maven POM"""
        chunks = []
        
        # Key sections to chunk separately
        sections = ['dependencies', 'build', 'properties', 'profiles', 
                   'repositories', 'pluginRepositories', 'reporting']
        
        for section in sections:
            section_node = self._find_child_by_tag(node, section)
            if section_node:
                if section_node.token_count <= max_tokens:
                    chunks.append({
                        'path': section_node.path,
                        'type': f'maven_{section}',
                        'content': self._serialize_subtree(section_node)
                    })
                else:
                    # Further chunk large sections
                    if section == 'dependencies':
                        dep_chunks = self._chunk_dependencies(section_node, max_tokens)
                        chunks.extend(dep_chunks)
                    elif section == 'build':
                        build_chunks = self._chunk_build_section(section_node, max_tokens)
                        chunks.extend(build_chunks)
                    else:
                        # Generic chunking for other sections
                        sub_chunks = self._chunk_by_children(section_node, max_tokens)
                        chunks.extend(sub_chunks)
        
        return chunks
    
    def _chunk_spring_config(self, node: XMLNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Special chunking for Spring configuration"""
        chunks = []
        
        # Chunk by beans
        beans = [child for child in node.children if child.tag == 'bean']
        
        for bean in beans:
            if bean.token_count <= max_tokens:
                chunks.append({
                    'path': bean.path,
                    'type': 'spring_bean',
                    'bean_id': bean.attrib.get('id', 'anonymous'),
                    'content': self._serialize_subtree(bean)
                })
            else:
                # Split large bean definition
                chunks.extend(self._split_large_element(bean, max_tokens))
        
        # Add other configurations
        non_beans = [child for child in node.children if child.tag != 'bean']
        for elem in non_beans:
            chunks.append({
                'path': elem.path,
                'type': f'spring_{elem.tag}',
                'content': self._serialize_subtree(elem)
            })
        
        return chunks
    
    def _chunk_by_path(self, node: XMLNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Generic path-based chunking"""
        chunks = []
        
        # Chunk by major child elements
        for child in node.children:
            if child.token_count <= max_tokens:
                chunks.append({
                    'path': child.path,
                    'content': self._serialize_subtree(child)
                })
            else:
                # Recursively chunk large elements
                sub_chunks = self._chunk_by_path(child, max_tokens)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _chunk_dependencies(self, deps_node: XMLNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk Maven dependencies"""
        chunks = []
        current_deps = []
        current_tokens = 0
        
        for dep in deps_node.children:
            if dep.tag == 'dependency':
                if current_tokens + dep.token_count > max_tokens and current_deps:
                    chunks.append({
                        'path': f"{deps_node.path}/batch_{len(chunks)}",
                        'type': 'maven_dependencies_batch',
                        'content': self._wrap_dependencies(current_deps)
                    })
                    current_deps = []
                    current_tokens = 0
                
                current_deps.append(self._serialize_subtree(dep))
                current_tokens += dep.token_count
        
        if current_deps:
            chunks.append({
                'path': f"{deps_node.path}/batch_{len(chunks)}",
                'type': 'maven_dependencies_batch',
                'content': self._wrap_dependencies(current_deps)
            })
        
        return chunks
    
    def _chunk_build_section(self, build_node: XMLNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk Maven build section"""
        chunks = []
        
        # Separate plugins, resources, etc.
        for child in build_node.children:
            if child.token_count <= max_tokens:
                chunks.append({
                    'path': child.path,
                    'type': f'maven_build_{child.tag}',
                    'content': self._serialize_subtree(child)
                })
            else:
                # Further chunk if needed
                sub_chunks = self._chunk_by_children(child, max_tokens)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _chunk_by_children(self, node: XMLNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Chunk node by its children"""
        chunks = []
        current_children = []
        current_tokens = 0
        
        for child in node.children:
            if current_tokens + child.token_count > max_tokens and current_children:
                chunks.append({
                    'path': f"{node.path}/part_{len(chunks)}",
                    'content': self._wrap_elements(node.tag, current_children),
                    'partial': True
                })
                current_children = []
                current_tokens = 0
            
            current_children.append(self._serialize_subtree(child))
            current_tokens += child.token_count
        
        if current_children:
            chunks.append({
                'path': f"{node.path}/part_{len(chunks)}",
                'content': self._wrap_elements(node.tag, current_children),
                'partial': len(chunks) > 0
            })
        
        return chunks
    
    def _split_large_element(self, node: XMLNode, max_tokens: int) -> List[Dict[str, Any]]:
        """Split large element into chunks"""
        chunks = []
        
        # Create header chunk with attributes
        header = f"<{node.tag}"
        for key, value in node.attrib.items():
            header += f' {key}="{html.escape(value)}"'
        header += ">"
        
        chunks.append({
            'path': f"{node.path}/header",
            'type': 'element_header',
            'content': header
        })
        
        # Chunk children
        if node.children:
            child_chunks = self._chunk_by_children(node, max_tokens)
            chunks.extend(child_chunks)
        
        # Add text content if present
        if node.text and node.text.strip():
            chunks.append({
                'path': f"{node.path}/text",
                'type': 'text_content',
                'content': node.text
            })
        
        return chunks
    
    def _find_child_by_tag(self, node: XMLNode, tag: str) -> Optional[XMLNode]:
        """Find child node by tag name"""
        for child in node.children:
            if child.tag == tag:
                return child
        return None
    
    def _serialize_subtree(self, node: XMLNode) -> str:
        """Serialize node and its subtree to XML"""
        # This would use proper XML serialization
        # Simplified for illustration
        parts = [f"<{node.tag}"]
        
        for key, value in node.attrib.items():
            parts.append(f' {key}="{html.escape(value)}"')
        
        if not node.children and not node.text:
            parts.append("/>")
        else:
            parts.append(">")
            
            if node.text:
                parts.append(html.escape(node.text))
            
            for child in node.children:
                parts.append(self._serialize_subtree(child))
            
            parts.append(f"</{node.tag}>")
        
        return ''.join(parts)
    
    def _wrap_dependencies(self, deps: List[str]) -> str:
        """Wrap dependencies in dependencies element"""
        return f"<dependencies>\n{''.join(deps)}\n</dependencies>"
    
    def _wrap_elements(self, tag: str, elements: List[str]) -> str:
        """Wrap elements in parent tag"""
        return f"<{tag}>\n{''.join(elements)}\n</{tag}>"

class XMLChunker(BaseChunker):
    """Chunker specialized for XML files"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, max_tokens)
        self.analyzer = XMLAnalyzer(tokenizer)
        self.depth_strategy = DepthBasedStrategy()
        self.path_strategy = PathBasedStrategy()
        
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from XML file
        
        Args:
            content: XML content as string
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Analyze XML structure
            structure = self.analyzer.analyze_xml(content, file_context.path)
            
            # Check for validation errors
            if structure.validation_errors:
                logger.warning(f"XML validation errors: {structure.validation_errors}")
            
            # Choose chunking strategy based on format
            if structure.format in [XMLFormat.MAVEN_POM, XMLFormat.SPRING_CONFIG, 
                                   XMLFormat.ANT_BUILD]:
                return self._chunk_structured_xml(content, structure, file_context)
            elif structure.format in [XMLFormat.SVG, XMLFormat.MATHML]:
                return self._chunk_visual_xml(content, structure, file_context)
            elif structure.format in [XMLFormat.RSS, XMLFormat.ATOM]:
                return self._chunk_feed_xml(content, structure, file_context)
            elif structure.format in [XMLFormat.SOAP, XMLFormat.WSDL]:
                return self._chunk_service_xml(content, structure, file_context)
            else:
                return self._chunk_generic_xml(content, structure, file_context)
                
        except Exception as e:
            logger.error(f"Error chunking XML file {file_context.path}: {e}")
            return self._fallback_chunking(content, file_context)
    
    def _chunk_structured_xml(self, content: str, structure: XMLStructure,
                             file_context: FileContext) -> List[Chunk]:
        """Chunk structured XML like POM, Spring config"""
        chunks = []
        
        # Add XML declaration and root element start
        header = self._create_xml_header(structure)
        chunks.append(self.create_chunk(
            content=header,
            chunk_type='xml_header',
            metadata={
                'format': structure.format.value,
                'root_tag': structure.root_tag,
                'namespaces': len(structure.namespaces),
                'encoding': structure.encoding
            },
            file_path=str(file_context.path)
        ))
        
        # Use path-based strategy for structured XML
        chunk_data = self.path_strategy.chunk(
            self.analyzer.root_node,
            self.max_tokens,
            structure.format
        )
        
        for i, data in enumerate(chunk_data):
            chunks.append(self.create_chunk(
                content=data['content'],
                chunk_type=data.get('type', 'xml_element'),
                metadata={
                    'format': structure.format.value,
                    'path': data['path'],
                    'chunk_index': i,
                    'is_partial': data.get('partial', False)
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _chunk_visual_xml(self, content: str, structure: XMLStructure,
                         file_context: FileContext) -> List[Chunk]:
        """Chunk visual XML like SVG, MathML"""
        chunks = []
        
        # For visual XML, try to keep related elements together
        if structure.format == XMLFormat.SVG:
            chunks.extend(self._chunk_svg(content, structure, file_context))
        else:
            # Generic visual XML chunking
            chunks.extend(self._chunk_by_visual_groups(content, structure, file_context))
        
        return chunks
    
    def _chunk_feed_xml(self, content: str, structure: XMLStructure,
                       file_context: FileContext) -> List[Chunk]:
        """Chunk feed XML like RSS, Atom"""
        chunks = []
        root = self.analyzer.root_node
        
        # Find feed metadata
        metadata_elements = ['title', 'description', 'link', 'language', 
                           'copyright', 'pubDate', 'lastBuildDate']
        
        metadata_content = []
        for elem_name in metadata_elements:
            elem = self._find_element_by_tag(root, elem_name)
            if elem:
                metadata_content.append(self._element_to_string(elem))
        
        if metadata_content:
            chunks.append(self.create_chunk(
                content='\n'.join(metadata_content),
                chunk_type='feed_metadata',
                metadata={
                    'format': structure.format.value,
                    'feed_type': 'rss' if structure.format == XMLFormat.RSS else 'atom'
                },
                file_path=str(file_context.path)
            ))
        
        # Chunk items/entries
        item_tag = 'item' if structure.format == XMLFormat.RSS else 'entry'
        items = [child for child in root.children if child.tag == item_tag]
        
        # Batch items
        item_batches = self._batch_elements(items, self.max_tokens)
        
        for i, batch in enumerate(item_batches):
            batch_content = '\n'.join(self._element_to_string(item) for item in batch)
            
            chunks.append(self.create_chunk(
                content=batch_content,
                chunk_type='feed_items',
                metadata={
                    'format': structure.format.value,
                    'batch_index': i,
                    'item_count': len(batch)
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _chunk_service_xml(self, content: str, structure: XMLStructure,
                          file_context: FileContext) -> List[Chunk]:
        """Chunk service XML like SOAP, WSDL"""
        chunks = []
        
        if structure.format == XMLFormat.WSDL:
            # Chunk WSDL by service definitions
            chunks.extend(self._chunk_wsdl(content, structure, file_context))
        elif structure.format == XMLFormat.SOAP:
            # Chunk SOAP by envelope parts
            chunks.extend(self._chunk_soap(content, structure, file_context))
        
        return chunks
    
    def _chunk_generic_xml(self, content: str, structure: XMLStructure,
                          file_context: FileContext) -> List[Chunk]:
        """Generic XML chunking"""
        chunks = []
        
        # If small enough, single chunk
        if self.count_tokens(content) <= self.max_tokens:
            return [self.create_chunk(
                content=content,
                chunk_type='xml_complete',
                metadata={
                    'format': structure.format.value,
                    'element_count': structure.total_elements,
                    'max_depth': structure.max_depth
                },
                file_path=str(file_context.path)
            )]
        
        # Use depth-based strategy for generic XML
        chunk_data = self.depth_strategy.chunk(
            self.analyzer.root_node,
            self.max_tokens,
            structure.format
        )
        
        for i, data in enumerate(chunk_data):
            chunks.append(self.create_chunk(
                content=str(data),
                chunk_type='xml_depth_level',
                metadata={
                    'depth': data.get('depth', 0),
                    'chunk_index': i
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _chunk_svg(self, content: str, structure: XMLStructure,
                  file_context: FileContext) -> List[Chunk]:
        """Chunk SVG files"""
        chunks = []
        root = self.analyzer.root_node
        
        # SVG metadata and defs
        defs = self._find_element_by_tag(root, 'defs')
        if defs:
            chunks.append(self.create_chunk(
                content=self._element_to_string(defs),
                chunk_type='svg_definitions',
                metadata={'has_gradients': 'linearGradient' in str(defs)},
                file_path=str(file_context.path)
            ))
        
        # Group elements by type
        groups = [child for child in root.children if child.tag == 'g']
        paths = [child for child in root.children if child.tag == 'path']
        shapes = [child for child in root.children 
                 if child.tag in ['rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon']]
        
        # Chunk groups
        if groups:
            group_batches = self._batch_elements(groups, self.max_tokens)
            for i, batch in enumerate(group_batches):
                chunks.append(self.create_chunk(
                    content=self._elements_to_string(batch),
                    chunk_type='svg_groups',
                    metadata={'group_count': len(batch), 'batch': i},
                    file_path=str(file_context.path)
                ))
        
        # Chunk paths
        if paths:
            path_batches = self._batch_elements(paths, self.max_tokens)
            for i, batch in enumerate(path_batches):
                chunks.append(self.create_chunk(
                    content=self._elements_to_string(batch),
                    chunk_type='svg_paths',
                    metadata={'path_count': len(batch), 'batch': i},
                    file_path=str(file_context.path)
                ))
        
        return chunks
    
    def _chunk_wsdl(self, content: str, structure: XMLStructure,
                   file_context: FileContext) -> List[Chunk]:
        """Chunk WSDL files"""
        chunks = []
        root = self.analyzer.root_node
        
        # WSDL sections
        sections = ['types', 'message', 'portType', 'binding', 'service']
        
        for section in sections:
            elements = [child for child in root.children if child.tag == section]
            
            for elem in elements:
                if elem.token_count <= self.max_tokens:
                    chunks.append(self.create_chunk(
                        content=self._element_to_string(elem),
                        chunk_type=f'wsdl_{section}',
                        metadata={
                            'section': section,
                            'name': elem.attrib.get('name', 'unnamed')
                        },
                        file_path=str(file_context.path)
                    ))
                else:
                    # Split large section
                    sub_chunks = self._split_large_wsdl_section(elem, section, file_context)
                    chunks.extend(sub_chunks)
        
        return chunks
    
    def _chunk_soap(self, content: str, structure: XMLStructure,
                   file_context: FileContext) -> List[Chunk]:
        """Chunk SOAP envelope"""
        chunks = []
        root = self.analyzer.root_node
        
        # SOAP parts
        header = self._find_element_by_tag(root, 'Header')
        body = self._find_element_by_tag(root, 'Body')
        fault = self._find_element_by_tag(root, 'Fault')
        
        if header:
            chunks.append(self.create_chunk(
                content=self._element_to_string(header),
                chunk_type='soap_header',
                metadata={'has_security': 'Security' in str(header)},
                file_path=str(file_context.path)
            ))
        
        if body:
            if body.token_count <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=self._element_to_string(body),
                    chunk_type='soap_body',
                    metadata={'operation_count': len(body.children)},
                    file_path=str(file_context.path)
                ))
            else:
                # Split large body
                for child in body.children:
                    chunks.append(self.create_chunk(
                        content=self._element_to_string(child),
                        chunk_type='soap_operation',
                        metadata={'operation': child.tag},
                        file_path=str(file_context.path)
                    ))
        
        if fault:
            chunks.append(self.create_chunk(
                content=self._element_to_string(fault),
                chunk_type='soap_fault',
                metadata={},
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _chunk_by_visual_groups(self, content: str, structure: XMLStructure,
                               file_context: FileContext) -> List[Chunk]:
        """Chunk visual XML by logical groups"""
        chunks = []
        root = self.analyzer.root_node
        
        # Group by element depth and type
        depth_groups = defaultdict(list)
        
        for node in self.analyzer.nodes:
            if node.depth <= 3:  # Only chunk top-level elements
                depth_groups[node.depth].append(node)
        
        for depth, nodes in sorted(depth_groups.items()):
            node_batches = self._batch_elements(nodes, self.max_tokens)
            
            for i, batch in enumerate(node_batches):
                chunks.append(self.create_chunk(
                    content=self._elements_to_string(batch),
                    chunk_type='xml_visual_group',
                    metadata={
                        'depth': depth,
                        'element_count': len(batch),
                        'batch': i
                    },
                    file_path=str(file_context.path)
                ))
        
        return chunks
    
    def _create_xml_header(self, structure: XMLStructure) -> str:
        """Create XML header with declaration and root element"""
        parts = []
        
        # XML declaration
        parts.append(f'<?xml version="{structure.version}" encoding="{structure.encoding}"?>')
        
        # DOCTYPE if present
        if structure.doctype:
            parts.append(structure.doctype)
        
        # Root element with namespaces
        root_parts = [f"<{structure.root_tag}"]
        
        for ns in structure.namespaces.values():
            if ns.is_default:
                root_parts.append(f' xmlns="{ns.uri}"')
            else:
                root_parts.append(f' xmlns:{ns.prefix}="{ns.uri}"')
        
        # Schema location if present
        if structure.schema_location:
            root_parts.append(f' xsi:schemaLocation="{structure.schema_location}"')
        
        root_parts.append('>')
        parts.append(''.join(root_parts))
        
        return '\n'.join(parts)
    
    def _find_element_by_tag(self, node: XMLNode, tag: str) -> Optional[XMLNode]:
        """Find element by tag name"""
        if node.tag == tag:
            return node
        
        for child in node.children:
            result = self._find_element_by_tag(child, tag)
            if result:
                return result
        
        return None
    
    def _element_to_string(self, node: XMLNode) -> str:
        """Convert element to XML string"""
        # This would use proper XML serialization
        # Simplified version
        return self.path_strategy._serialize_subtree(node)
    
    def _elements_to_string(self, nodes: List[XMLNode]) -> str:
        """Convert multiple elements to XML string"""
        return '\n'.join(self._element_to_string(node) for node in nodes)
    
    def _batch_elements(self, elements: List[XMLNode], max_tokens: int) -> List[List[XMLNode]]:
        """Batch elements by token count"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for elem in elements:
            if current_tokens + elem.token_count > max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [elem]
                current_tokens = elem.token_count
            else:
                current_batch.append(elem)
                current_tokens += elem.token_count
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _split_large_wsdl_section(self, elem: XMLNode, section: str,
                                 file_context: FileContext) -> List[Chunk]:
        """Split large WSDL section"""
        chunks = []
        
        # Split by child elements
        for child in elem.children:
            chunks.append(self.create_chunk(
                content=self._element_to_string(child),
                chunk_type=f'wsdl_{section}_part',
                metadata={
                    'section': section,
                    'part': child.attrib.get('name', 'unnamed'),
                    'parent': elem.attrib.get('name', 'unnamed')
                },
                file_path=str(file_context.path)
            ))
        
        return chunks
    
    def _fallback_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback chunking for invalid XML"""
        logger.warning(f"Using fallback chunking for XML file {file_context.path}")
        
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        in_element = False
        element_stack = []
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            # Track element boundaries
            if '<' in line and '>' in line:
                # Simple element tracking
                if '</' in line:
                    if element_stack:
                        element_stack.pop()
                elif not line.strip().startswith('<!'):
                    match = re.search(r'<(\w+)', line)
                    if match:
                        element_stack.append(match.group(1))
            
            # Check if we should start new chunk
            should_split = (
                current_tokens + line_tokens > self.max_tokens and 
                current_chunk and
                len(element_stack) == 0  # Try to keep elements together
            )
            
            if should_split:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='xml_fallback',
                    metadata={
                        'is_fallback': True,
                        'line_count': len(current_chunk)
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='xml_fallback',
                metadata={
                    'is_fallback': True,
                    'line_count': len(current_chunk)
                },
                file_path=str(file_context.path)
            ))
        
        return chunks