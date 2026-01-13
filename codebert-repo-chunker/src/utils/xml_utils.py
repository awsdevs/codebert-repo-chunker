"""
XML processing utilities for handling various XML formats
Provides robust XML parsing, validation, transformation, and namespace handling
"""

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from lxml import etree, html
from typing import Dict, List, Optional, Any, Union, Tuple, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import logging
from io import StringIO, BytesIO
import xmlschema
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class XMLNode:
    """Represents an XML node with metadata"""
    tag: str
    text: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    children: List['XMLNode'] = field(default_factory=list)
    namespace: Optional[str] = None
    prefix: Optional[str] = None
    line_number: Optional[int] = None
    parent: Optional['XMLNode'] = None
    
    def get_path(self) -> str:
        """Get XPath-like path to this node"""
        if self.parent is None:
            return f"/{self.tag}"
        return f"{self.parent.get_path()}/{self.tag}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary"""
        result = {
            'tag': self.tag,
            'attributes': self.attributes,
        }
        
        if self.text and self.text.strip():
            result['text'] = self.text.strip()
        
        if self.children:
            result['children'] = [child.to_dict() for child in self.children]
        
        if self.namespace:
            result['namespace'] = self.namespace
            
        return result

class XMLNamespaceHandler:
    """Handles XML namespaces for parsing and processing"""
    
    # Common XML namespaces
    COMMON_NAMESPACES = {
        'xml': 'http://www.w3.org/XML/1998/namespace',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsd': 'http://www.w3.org/2001/XMLSchema',
        'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
        'wsdl': 'http://schemas.xmlsoap.org/wsdl/',
        'html': 'http://www.w3.org/1999/xhtml',
        'svg': 'http://www.w3.org/2000/svg',
        'mathml': 'http://www.w3.org/1998/Math/MathML',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'atom': 'http://www.w3.org/2005/Atom',
        'rss': 'http://purl.org/rss/1.0/',
    }
    
    # Technology-specific namespaces
    TECH_NAMESPACES = {
        # Maven
        'maven': 'http://maven.apache.org/POM/4.0.0',
        'maven-settings': 'http://maven.apache.org/SETTINGS/1.0.0',
        
        # Spring
        'spring': 'http://www.springframework.org/schema/beans',
        'spring-context': 'http://www.springframework.org/schema/context',
        'spring-mvc': 'http://www.springframework.org/schema/mvc',
        'spring-security': 'http://www.springframework.org/schema/security',
        'spring-boot': 'http://www.springframework.org/schema/boot',
        
        # Android
        'android': 'http://schemas.android.com/apk/res/android',
        'app': 'http://schemas.android.com/apk/res-auto',
        'tools': 'http://schemas.android.com/tools',
        
        # .NET
        'msbuild': 'http://schemas.microsoft.com/developer/msbuild/2003',
        'nuget': 'http://schemas.microsoft.com/packaging/2010/07/nuspec.xsd',
        
        # SOAP/WSDL
        'soapenv': 'http://schemas.xmlsoap.org/soap/envelope/',
        'wsdl': 'http://schemas.xmlsoap.org/wsdl/',
        
        # Other
        'ant': 'http://ant.apache.org',
        'ivy': 'http://ant.apache.org/ivy',
    }
    
    def __init__(self):
        """Initialize namespace handler"""
        self.namespaces = {}
        self.prefixes = {}
        self._register_common_namespaces()
    
    def _register_common_namespaces(self):
        """Register common namespaces"""
        for prefix, uri in self.COMMON_NAMESPACES.items():
            self.register_namespace(prefix, uri)
        
        for prefix, uri in self.TECH_NAMESPACES.items():
            self.register_namespace(prefix, uri)
    
    def register_namespace(self, prefix: str, uri: str):
        """Register a namespace prefix and URI"""
        self.namespaces[prefix] = uri
        self.prefixes[uri] = prefix
        
        # Also register with ElementTree
        ET.register_namespace(prefix, uri)
        
        # Register with lxml
        etree.register_namespace(prefix, uri)
    
    def extract_namespaces(self, xml_content: str) -> Dict[str, str]:
        """Extract all namespaces from XML content"""
        namespaces = {}
        
        # Parse XML to find namespace declarations
        try:
            root = etree.fromstring(xml_content.encode('utf-8'))
            
            # Get all namespaces from the document
            for prefix, uri in root.nsmap.items():
                if prefix is None:
                    prefix = 'default'
                namespaces[prefix] = uri
                self.register_namespace(prefix, uri)
            
            # Also check for namespaces in the document
            for elem in root.iter():
                for prefix, uri in elem.nsmap.items():
                    if prefix is None:
                        prefix = 'default'
                    if prefix not in namespaces:
                        namespaces[prefix] = uri
                        self.register_namespace(prefix, uri)
        
        except Exception as e:
            logger.warning(f"Failed to extract namespaces: {e}")
        
        return namespaces
    
    def get_namespace_for_element(self, element: ET.Element) -> Optional[str]:
        """Get namespace for an element"""
        if '}' in element.tag:
            return element.tag.split('}')[0][1:]
        return None
    
    def get_local_name(self, tag: str) -> str:
        """Get local name without namespace"""
        if '}' in tag:
            return tag.split('}')[1]
        return tag
    
    def format_tag_with_prefix(self, tag: str) -> str:
        """Format tag with namespace prefix"""
        if '}' in tag:
            namespace = tag.split('}')[0][1:]
            local_name = tag.split('}')[1]
            
            prefix = self.prefixes.get(namespace)
            if prefix:
                return f"{prefix}:{local_name}"
        
        return tag

class XMLProcessor:
    """Main XML processing utility"""
    
    def __init__(self, namespace_aware: bool = True):
        """
        Initialize XML processor
        
        Args:
            namespace_aware: Enable namespace handling
        """
        self.namespace_aware = namespace_aware
        self.namespace_handler = XMLNamespaceHandler() if namespace_aware else None
        self.schemas = {}  # Cache for loaded schemas
    
    def parse(self, 
             xml_input: Union[str, Path, bytes],
             validate: bool = False,
             schema_path: Optional[Path] = None) -> XMLNode:
        """
        Parse XML from various input sources
        
        Args:
            xml_input: XML content, file path, or bytes
            validate: Validate against schema
            schema_path: Path to XSD schema
            
        Returns:
            Parsed XMLNode tree
        """
        # Get XML content
        if isinstance(xml_input, Path):
            with open(xml_input, 'r', encoding='utf-8') as f:
                content = f.read()
        elif isinstance(xml_input, bytes):
            content = xml_input.decode('utf-8')
        else:
            content = xml_input
        
        # Extract namespaces if namespace aware
        if self.namespace_aware:
            self.namespace_handler.extract_namespaces(content)
        
        # Validate if requested
        if validate and schema_path:
            if not self.validate_against_schema(content, schema_path):
                raise ValueError("XML validation failed")
        
        # Parse with lxml for better error handling and line numbers
        try:
            parser = etree.XMLParser(recover=True, remove_blank_text=True)
            root = etree.fromstring(content.encode('utf-8'), parser)
            
            # Convert to XMLNode structure
            return self._etree_to_node(root)
            
        except Exception as e:
            logger.error(f"Failed to parse XML: {e}")
            # Fallback to ElementTree
            try:
                root = ET.fromstring(content)
                return self._element_to_node(root)
            except Exception as e2:
                logger.error(f"Fallback parsing also failed: {e2}")
                raise
    
    def _etree_to_node(self, element: etree._Element, parent: Optional[XMLNode] = None) -> XMLNode:
        """Convert lxml element to XMLNode"""
        # Get tag info
        tag = self.namespace_handler.get_local_name(element.tag) if self.namespace_aware else element.tag
        namespace = self.namespace_handler.get_namespace_for_element(element) if self.namespace_aware else None
        
        # Create node
        node = XMLNode(
            tag=tag,
            text=element.text if element.text else None,
            attributes=dict(element.attrib),
            namespace=namespace,
            line_number=element.sourceline if hasattr(element, 'sourceline') else None,
            parent=parent
        )
        
        # Process children
        for child in element:
            child_node = self._etree_to_node(child, node)
            node.children.append(child_node)
        
        return node
    
    def _element_to_node(self, element: ET.Element, parent: Optional[XMLNode] = None) -> XMLNode:
        """Convert ElementTree element to XMLNode"""
        # Get tag info
        tag = self.namespace_handler.get_local_name(element.tag) if self.namespace_aware else element.tag
        namespace = self.namespace_handler.get_namespace_for_element(element) if self.namespace_aware else None
        
        # Create node
        node = XMLNode(
            tag=tag,
            text=element.text if element.text else None,
            attributes=dict(element.attrib),
            namespace=namespace,
            parent=parent
        )
        
        # Process children
        for child in element:
            child_node = self._element_to_node(child, node)
            node.children.append(child_node)
        
        return node
    
    def parse_file(self, file_path: Path) -> XMLNode:
        """Parse XML from file"""
        return self.parse(file_path)
    
    def find_all(self, 
                root: XMLNode,
                tag: str,
                namespace: Optional[str] = None) -> List[XMLNode]:
        """
        Find all nodes with specified tag
        
        Args:
            root: Root node to search from
            tag: Tag name to search for
            namespace: Optional namespace filter
            
        Returns:
            List of matching nodes
        """
        results = []
        
        def _search(node: XMLNode):
            if node.tag == tag:
                if namespace is None or node.namespace == namespace:
                    results.append(node)
            
            for child in node.children:
                _search(child)
        
        _search(root)
        return results
    
    def find_first(self,
                  root: XMLNode,
                  tag: str,
                  namespace: Optional[str] = None) -> Optional[XMLNode]:
        """Find first node with specified tag"""
        results = self.find_all(root, tag, namespace)
        return results[0] if results else None
    
    def xpath(self, 
             root: Union[XMLNode, ET.Element, etree._Element],
             expression: str,
             namespaces: Optional[Dict[str, str]] = None) -> List[Any]:
        """
        Execute XPath expression
        
        Args:
            root: Root node or element
            expression: XPath expression
            namespaces: Namespace mappings for XPath
            
        Returns:
            List of matching nodes/values
        """
        # Convert XMLNode to etree if needed
        if isinstance(root, XMLNode):
            xml_str = self.node_to_string(root)
            root = etree.fromstring(xml_str.encode('utf-8'))
        
        # Use namespace handler's namespaces if not provided
        if namespaces is None and self.namespace_aware:
            namespaces = self.namespace_handler.namespaces
        
        try:
            return root.xpath(expression, namespaces=namespaces)
        except Exception as e:
            logger.error(f"XPath error: {e}")
            return []
    
    def get_text(self, node: XMLNode, path: str = None) -> Optional[str]:
        """
        Get text content from node or path
        
        Args:
            node: Node to get text from
            path: Optional path to child node (e.g., "child/grandchild")
            
        Returns:
            Text content or None
        """
        if path:
            parts = path.split('/')
            current = node
            
            for part in parts:
                found = False
                for child in current.children:
                    if child.tag == part:
                        current = child
                        found = True
                        break
                
                if not found:
                    return None
            
            node = current
        
        # Collect all text
        text_parts = []
        
        def _collect_text(n: XMLNode):
            if n.text:
                text_parts.append(n.text.strip())
            for child in n.children:
                _collect_text(child)
        
        _collect_text(node)
        
        return ' '.join(text_parts) if text_parts else None
    
    def get_attribute(self,
                     node: XMLNode,
                     attribute: str,
                     default: Optional[str] = None) -> Optional[str]:
        """Get attribute value from node"""
        return node.attributes.get(attribute, default)
    
    def node_to_string(self,
                      node: XMLNode,
                      pretty: bool = True,
                      include_declaration: bool = True) -> str:
        """
        Convert XMLNode back to XML string
        
        Args:
            node: Node to convert
            pretty: Pretty print the output
            include_declaration: Include XML declaration
            
        Returns:
            XML string
        """
        # Build ElementTree
        element = self._node_to_element(node)
        
        # Convert to string
        xml_str = ET.tostring(element, encoding='unicode', method='xml')
        
        # Pretty print if requested
        if pretty:
            try:
                dom = minidom.parseString(xml_str)
                pretty_xml = dom.toprettyxml(indent='  ')
                
                # Remove extra blank lines
                lines = [line for line in pretty_xml.split('\n') if line.strip()]
                
                # Remove declaration if not wanted
                if not include_declaration and lines and lines[0].startswith('<?xml'):
                    lines = lines[1:]
                
                return '\n'.join(lines)
            except:
                pass
        
        # Add declaration if requested
        if include_declaration and not xml_str.startswith('<?xml'):
            xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str
        
        return xml_str
    
    def _node_to_element(self, node: XMLNode) -> ET.Element:
        """Convert XMLNode to ElementTree Element"""
        # Create element with namespace if present
        if node.namespace and self.namespace_aware:
            tag = f"{{{node.namespace}}}{node.tag}"
        else:
            tag = node.tag
        
        element = ET.Element(tag, attrib=node.attributes)
        
        if node.text:
            element.text = node.text
        
        # Add children
        for child in node.children:
            child_element = self._node_to_element(child)
            element.append(child_element)
        
        return element
    
    def validate_against_schema(self,
                               xml_content: Union[str, Path],
                               schema_path: Path) -> bool:
        """
        Validate XML against XSD schema
        
        Args:
            xml_content: XML content or file path
            schema_path: Path to XSD schema
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Load schema if not cached
            schema_key = str(schema_path)
            if schema_key not in self.schemas:
                self.schemas[schema_key] = xmlschema.XMLSchema(str(schema_path))
            
            schema = self.schemas[schema_key]
            
            # Get XML content
            if isinstance(xml_content, Path):
                with open(xml_content, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
            
            # Validate
            schema.validate(xml_content)
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    def transform_xslt(self,
                      xml_input: Union[str, Path, XMLNode],
                      xslt_path: Path) -> str:
        """
        Transform XML using XSLT
        
        Args:
            xml_input: XML input
            xslt_path: Path to XSLT stylesheet
            
        Returns:
            Transformed XML string
        """
        try:
            # Get XML content
            if isinstance(xml_input, XMLNode):
                xml_str = self.node_to_string(xml_input)
            elif isinstance(xml_input, Path):
                with open(xml_input, 'r', encoding='utf-8') as f:
                    xml_str = f.read()
            else:
                xml_str = xml_input
            
            # Parse XML and XSLT
            xml_doc = etree.fromstring(xml_str.encode('utf-8'))
            
            with open(xslt_path, 'r', encoding='utf-8') as f:
                xslt_doc = etree.fromstring(f.read().encode('utf-8'))
            
            # Create transformer
            transform = etree.XSLT(xslt_doc)
            
            # Transform
            result = transform(xml_doc)
            
            return str(result)
            
        except Exception as e:
            logger.error(f"XSLT transformation failed: {e}")
            raise
    
    def xml_to_dict(self, node: XMLNode) -> Dict[str, Any]:
        """Convert XMLNode tree to dictionary"""
        return node.to_dict()
    
    def xml_to_json(self, 
                   node: XMLNode,
                   indent: int = 2) -> str:
        """Convert XMLNode tree to JSON string"""
        return json.dumps(node.to_dict(), indent=indent, default=str)
    
    def extract_maven_info(self, pom_path: Path) -> Dict[str, Any]:
        """Extract information from Maven POM file"""
        try:
            root = self.parse_file(pom_path)
            
            info = {
                'groupId': self.get_text(root, 'groupId'),
                'artifactId': self.get_text(root, 'artifactId'),
                'version': self.get_text(root, 'version'),
                'packaging': self.get_text(root, 'packaging') or 'jar',
                'name': self.get_text(root, 'name'),
                'description': self.get_text(root, 'description'),
                'dependencies': [],
                'properties': {},
                'modules': []
            }
            
            # Extract parent info
            parent_node = self.find_first(root, 'parent')
            if parent_node:
                info['parent'] = {
                    'groupId': self.get_text(parent_node, 'groupId'),
                    'artifactId': self.get_text(parent_node, 'artifactId'),
                    'version': self.get_text(parent_node, 'version')
                }
            
            # Extract dependencies
            deps_node = self.find_first(root, 'dependencies')
            if deps_node:
                for dep in self.find_all(deps_node, 'dependency'):
                    dep_info = {
                        'groupId': self.get_text(dep, 'groupId'),
                        'artifactId': self.get_text(dep, 'artifactId'),
                        'version': self.get_text(dep, 'version'),
                        'scope': self.get_text(dep, 'scope') or 'compile'
                    }
                    info['dependencies'].append(dep_info)
            
            # Extract properties
            props_node = self.find_first(root, 'properties')
            if props_node:
                for child in props_node.children:
                    if child.text:
                        info['properties'][child.tag] = child.text.strip()
            
            # Extract modules (for multi-module projects)
            modules_node = self.find_first(root, 'modules')
            if modules_node:
                for module in self.find_all(modules_node, 'module'):
                    if module.text:
                        info['modules'].append(module.text.strip())
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to extract Maven info: {e}")
            return {}
    
    def extract_spring_beans(self, xml_path: Path) -> List[Dict[str, Any]]:
        """Extract Spring bean definitions from XML"""
        try:
            root = self.parse_file(xml_path)
            beans = []
            
            for bean_node in self.find_all(root, 'bean'):
                bean_info = {
                    'id': self.get_attribute(bean_node, 'id'),
                    'name': self.get_attribute(bean_node, 'name'),
                    'class': self.get_attribute(bean_node, 'class'),
                    'scope': self.get_attribute(bean_node, 'scope', 'singleton'),
                    'properties': [],
                    'constructor_args': []
                }
                
                # Extract properties
                for prop in self.find_all(bean_node, 'property'):
                    prop_info = {
                        'name': self.get_attribute(prop, 'name'),
                        'value': self.get_attribute(prop, 'value'),
                        'ref': self.get_attribute(prop, 'ref')
                    }
                    bean_info['properties'].append(prop_info)
                
                # Extract constructor arguments
                for arg in self.find_all(bean_node, 'constructor-arg'):
                    arg_info = {
                        'value': self.get_attribute(arg, 'value'),
                        'ref': self.get_attribute(arg, 'ref'),
                        'type': self.get_attribute(arg, 'type')
                    }
                    bean_info['constructor_args'].append(arg_info)
                
                beans.append(bean_info)
            
            return beans
            
        except Exception as e:
            logger.error(f"Failed to extract Spring beans: {e}")
            return []
    
    def prettify(self, xml_content: str) -> str:
        """Prettify XML content"""
        try:
            dom = minidom.parseString(xml_content)
            pretty = dom.toprettyxml(indent='  ')
            
            # Remove extra blank lines
            lines = [line for line in pretty.split('\n') if line.strip()]
            return '\n'.join(lines)
            
        except Exception as e:
            logger.error(f"Failed to prettify XML: {e}")
            return xml_content
    
    def minify(self, xml_content: str) -> str:
        """Minify XML content (remove whitespace)"""
        try:
            # Parse and reconstruct without pretty printing
            root = ET.fromstring(xml_content)
            return ET.tostring(root, encoding='unicode', method='xml')
            
        except Exception as e:
            logger.error(f"Failed to minify XML: {e}")
            return xml_content

# Convenience functions
def parse_xml(xml_input: Union[str, Path, bytes], namespace_aware: bool = True) -> XMLNode:
    """Parse XML content"""
    processor = XMLProcessor(namespace_aware=namespace_aware)
    return processor.parse(xml_input)

def xml_to_dict(xml_input: Union[str, Path, bytes]) -> Dict[str, Any]:
    """Convert XML to dictionary"""
    processor = XMLProcessor()
    node = processor.parse(xml_input)
    return node.to_dict()

def xml_to_json(xml_input: Union[str, Path, bytes], indent: int = 2) -> str:
    """Convert XML to JSON string"""
    processor = XMLProcessor()
    node = processor.parse(xml_input)
    return processor.xml_to_json(node, indent)

def validate_xml(xml_content: Union[str, Path], schema_path: Path) -> bool:
    """Validate XML against schema"""
    processor = XMLProcessor()
    return processor.validate_against_schema(xml_content, schema_path)

def extract_maven_dependencies(pom_path: Path) -> List[Dict[str, str]]:
    """Extract dependencies from Maven POM"""
    processor = XMLProcessor()
    info = processor.extract_maven_info(pom_path)
    return info.get('dependencies', [])