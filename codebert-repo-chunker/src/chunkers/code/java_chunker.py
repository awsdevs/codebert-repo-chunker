"""
Java-specific chunker for intelligent semantic chunking of Java source files
Handles classes, interfaces, enums, records, and modern Java features
"""

import re
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import javalang
from enum import Enum

from src.core.base_chunker import BaseChunker, Chunk, ChunkerConfig
from src.core.file_context import FileContext
from config.settings import settings

logger = logging.getLogger(__name__)

class JavaElementType(Enum):
    """Types of Java elements"""
    PACKAGE = "package"
    IMPORT = "import"
    CLASS = "class"
    INTERFACE = "interface"
    ENUM = "enum"
    RECORD = "record"
    ANNOTATION = "annotation"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    FIELD = "field"
    STATIC_BLOCK = "static_block"
    INNER_CLASS = "inner_class"
    ANONYMOUS_CLASS = "anonymous_class"
    LAMBDA = "lambda"

@dataclass
class JavaElement:
    """Represents a Java code element"""
    element_type: JavaElementType
    name: str
    content: str
    start_line: int
    end_line: int
    modifiers: List[str]
    annotations: List[str]
    parent: Optional['JavaElement'] = None
    children: List['JavaElement'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class JavaClass:
    """Represents a Java class structure"""
    package: Optional[str]
    imports: List[str]
    class_name: str
    class_type: JavaElementType
    extends: Optional[str]
    implements: List[str]
    modifiers: List[str]
    annotations: List[str]
    fields: List[JavaElement]
    methods: List[JavaElement]
    constructors: List[JavaElement]
    inner_classes: List['JavaClass']
    static_blocks: List[JavaElement]
    content: str
    start_line: int
    end_line: int
    javadoc: Optional[str]
    type_parameters: List[str]  # Generics

class JavaSyntaxAnalyzer:
    """Analyzes Java syntax and structure"""
    
    # Common Java annotations to recognize
    COMMON_ANNOTATIONS = {
        '@Override', '@Deprecated', '@SuppressWarnings', '@SafeVarargs',
        '@FunctionalInterface', '@Native', '@Target', '@Retention',
        # Spring
        '@Component', '@Service', '@Repository', '@Controller', '@RestController',
        '@Autowired', '@Bean', '@Configuration', '@RequestMapping', '@GetMapping',
        '@PostMapping', '@PutMapping', '@DeleteMapping', '@PathVariable',
        '@RequestParam', '@RequestBody', '@ResponseBody', '@Transactional',
        # JPA/Hibernate
        '@Entity', '@Table', '@Id', '@GeneratedValue', '@Column', '@OneToMany',
        '@ManyToOne', '@ManyToMany', '@OneToOne', '@JoinColumn',
        # Lombok
        '@Data', '@Getter', '@Setter', '@Builder', '@NoArgsConstructor',
        '@AllArgsConstructor', '@RequiredArgsConstructor', '@ToString',
        '@EqualsAndHashCode', '@Slf4j',
        # JUnit
        '@Test', '@Before', '@After', '@BeforeClass', '@AfterClass',
        '@BeforeEach', '@AfterEach', '@BeforeAll', '@AfterAll',
        '@DisplayName', '@Nested', '@ParameterizedTest',
        # Jackson
        '@JsonProperty', '@JsonIgnore', '@JsonFormat', '@JsonInclude',
        # Validation
        '@NotNull', '@NotEmpty', '@NotBlank', '@Size', '@Min', '@Max',
        '@Email', '@Pattern', '@Valid'
    }
    
    # Java keywords for context
    JAVA_KEYWORDS = {
        'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch',
        'char', 'class', 'const', 'continue', 'default', 'do', 'double',
        'else', 'enum', 'extends', 'final', 'finally', 'float', 'for',
        'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface',
        'long', 'native', 'new', 'package', 'private', 'protected', 'public',
        'return', 'short', 'static', 'strictfp', 'super', 'switch',
        'synchronized', 'this', 'throw', 'throws', 'transient', 'try',
        'void', 'volatile', 'while', 'var', 'record', 'sealed', 'permits',
        'yield', 'module', 'requires', 'exports', 'opens', 'uses', 'provides'
    }
    
    def __init__(self):
        self.current_package = None
        self.imports = []
        self.classes = []
        
    def analyze_java_file(self, content: str) -> Dict[str, Any]:
        """
        Analyze Java file structure
        
        Args:
            content: Java source code
            
        Returns:
            Analysis results
        """
        try:
            # Parse using javalang
            tree = javalang.parse.parse(content)
            
            # Extract structure
            analysis = {
                'package': self._extract_package(tree),
                'imports': self._extract_imports(tree),
                'classes': self._extract_classes(tree, content),
                'interfaces': self._extract_interfaces(tree, content),
                'enums': self._extract_enums(tree, content),
                'annotations': self._extract_annotation_types(tree, content),
                'complexity': self._calculate_complexity(tree),
                'metrics': self._calculate_metrics(tree, content),
                'dependencies': self._analyze_dependencies(tree)
            }
            
            return analysis
            
        except javalang.parser.JavaSyntaxError as e:
            logger.warning(f"Java syntax error: {e}")
            # Fallback to regex-based parsing
            return self._fallback_analysis(content)
        except Exception as e:
            logger.error(f"Error analyzing Java file: {e}")
            return self._fallback_analysis(content)
    
    def _extract_package(self, tree: javalang.tree.CompilationUnit) -> Optional[str]:
        """Extract package declaration"""
        if tree.package:
            return tree.package.name
        return None
    
    def _extract_imports(self, tree: javalang.tree.CompilationUnit) -> List[str]:
        """Extract import statements"""
        imports = []
        for import_decl in tree.imports:
            if import_decl.static:
                imports.append(f"static {import_decl.path}")
            else:
                imports.append(import_decl.path)
        return imports
    
    def _extract_classes(self, tree: javalang.tree.CompilationUnit, 
                        content: str) -> List[JavaClass]:
        """Extract class definitions"""
        classes = []
        lines = content.split('\n')
        
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            java_class = self._parse_class_node(node, path, lines)
            if java_class:
                classes.append(java_class)
        
        return classes
    
    def _parse_class_node(self, node: javalang.tree.ClassDeclaration,
                         path: Tuple, lines: List[str]) -> JavaClass:
        """Parse a class node into JavaClass object"""
        # Get line numbers
        start_line = node.position.line if node.position else 1
        end_line = self._find_end_line(node, lines, start_line)
        
        # Extract class content
        class_content = '\n'.join(lines[start_line - 1:end_line])
        
        # Extract modifiers
        modifiers = node.modifiers if node.modifiers else []
        
        # Extract annotations
        annotations = []
        if node.annotations:
            for ann in node.annotations:
                annotations.append(f"@{ann.name}")
        
        # Extract extends and implements
        extends = node.extends.name if node.extends else None
        implements = [impl.name for impl in node.implements] if node.implements else []
        
        # Extract type parameters (generics)
        type_params = []
        if node.type_parameters:
            for param in node.type_parameters:
                type_params.append(param.name)
        
        # Extract fields
        fields = self._extract_fields(node)
        
        # Extract methods
        methods = self._extract_methods(node)
        
        # Extract constructors
        constructors = self._extract_constructors(node)
        
        # Extract inner classes
        inner_classes = self._extract_inner_classes(node, lines)
        
        # Extract static blocks
        static_blocks = self._extract_static_blocks(node, lines)
        
        # Extract Javadoc
        javadoc = self._extract_javadoc(lines, start_line)
        
        return JavaClass(
            package=self._extract_package_from_path(path),
            imports=self.imports,
            class_name=node.name,
            class_type=JavaElementType.CLASS,
            extends=extends,
            implements=implements,
            modifiers=modifiers,
            annotations=annotations,
            fields=fields,
            methods=methods,
            constructors=constructors,
            inner_classes=inner_classes,
            static_blocks=static_blocks,
            content=class_content,
            start_line=start_line,
            end_line=end_line,
            javadoc=javadoc,
            type_parameters=type_params
        )
    
    def _extract_fields(self, node: javalang.tree.ClassDeclaration) -> List[JavaElement]:
        """Extract field declarations"""
        fields = []
        
        for field in node.fields:
            for declarator in field.declarators:
                field_element = JavaElement(
                    element_type=JavaElementType.FIELD,
                    name=declarator.name,
                    content=self._reconstruct_field(field, declarator),
                    start_line=field.position.line if field.position else 0,
                    end_line=field.position.line if field.position else 0,
                    modifiers=field.modifiers if field.modifiers else [],
                    annotations=[f"@{ann.name}" for ann in field.annotations] if field.annotations else [],
                    metadata={
                        'type': field.type.name if hasattr(field.type, 'name') else str(field.type),
                        'static': 'static' in (field.modifiers or []),
                        'final': 'final' in (field.modifiers or []),
                        'visibility': self._get_visibility(field.modifiers)
                    }
                )
                fields.append(field_element)
        
        return fields
    
    def _extract_methods(self, node: javalang.tree.ClassDeclaration) -> List[JavaElement]:
        """Extract method declarations"""
        methods = []
        
        for method in node.methods:
            method_element = JavaElement(
                element_type=JavaElementType.METHOD,
                name=method.name,
                content=self._reconstruct_method_signature(method),
                start_line=method.position.line if method.position else 0,
                end_line=0,  # Will be calculated later
                modifiers=method.modifiers if method.modifiers else [],
                annotations=[f"@{ann.name}" for ann in method.annotations] if method.annotations else [],
                metadata={
                    'return_type': method.return_type.name if method.return_type and hasattr(method.return_type, 'name') else str(method.return_type) if method.return_type else 'void',
                    'parameters': self._extract_parameters(method),
                    'throws': method.throws if method.throws else [],
                    'abstract': 'abstract' in (method.modifiers or []),
                    'static': 'static' in (method.modifiers or []),
                    'synchronized': 'synchronized' in (method.modifiers or []),
                    'visibility': self._get_visibility(method.modifiers),
                    'complexity': self._calculate_method_complexity(method)
                }
            )
            methods.append(method_element)
        
        return methods
    
    def _extract_constructors(self, node: javalang.tree.ClassDeclaration) -> List[JavaElement]:
        """Extract constructor declarations"""
        constructors = []
        
        for constructor in node.constructors:
            constructor_element = JavaElement(
                element_type=JavaElementType.CONSTRUCTOR,
                name=constructor.name,
                content=self._reconstruct_constructor_signature(constructor),
                start_line=constructor.position.line if constructor.position else 0,
                end_line=0,  # Will be calculated later
                modifiers=constructor.modifiers if constructor.modifiers else [],
                annotations=[f"@{ann.name}" for ann in constructor.annotations] if constructor.annotations else [],
                metadata={
                    'parameters': self._extract_parameters(constructor),
                    'throws': constructor.throws if constructor.throws else [],
                    'visibility': self._get_visibility(constructor.modifiers)
                }
            )
            constructors.append(constructor_element)
        
        return constructors
    
    def _extract_inner_classes(self, node: javalang.tree.ClassDeclaration,
                              lines: List[str]) -> List[JavaClass]:
        """Extract inner class declarations"""
        inner_classes = []
        
        # Look for nested class declarations
        for inner_node in node.body:
            if isinstance(inner_node, javalang.tree.ClassDeclaration):
                inner_class = self._parse_class_node(inner_node, (), lines)
                if inner_class:
                    inner_classes.append(inner_class)
        
        return inner_classes
    
    def _extract_static_blocks(self, node: javalang.tree.ClassDeclaration,
                              lines: List[str]) -> List[JavaElement]:
        """Extract static initialization blocks"""
        static_blocks = []
        
        # Look for static blocks in class body
        for item in node.body:
            if isinstance(item, javalang.tree.StaticInitializer):
                static_element = JavaElement(
                    element_type=JavaElementType.STATIC_BLOCK,
                    name="static",
                    content="static { ... }",
                    start_line=item.position.line if item.position else 0,
                    end_line=0,
                    modifiers=["static"],
                    annotations=[],
                    metadata={'is_static': True}
                )
                static_blocks.append(static_element)
        
        return static_blocks
    
    def _extract_interfaces(self, tree: javalang.tree.CompilationUnit,
                           content: str) -> List[JavaClass]:
        """Extract interface definitions"""
        interfaces = []
        lines = content.split('\n')
        
        for path, node in tree.filter(javalang.tree.InterfaceDeclaration):
            interface = self._parse_interface_node(node, path, lines)
            if interface:
                interfaces.append(interface)
        
        return interfaces
    
    def _parse_interface_node(self, node: javalang.tree.InterfaceDeclaration,
                             path: Tuple, lines: List[str]) -> JavaClass:
        """Parse interface node"""
        start_line = node.position.line if node.position else 1
        end_line = self._find_end_line(node, lines, start_line)
        
        interface_content = '\n'.join(lines[start_line - 1:end_line])
        
        return JavaClass(
            package=self._extract_package_from_path(path),
            imports=self.imports,
            class_name=node.name,
            class_type=JavaElementType.INTERFACE,
            extends=[ext.name for ext in node.extends] if node.extends else [],
            implements=[],
            modifiers=node.modifiers if node.modifiers else [],
            annotations=[f"@{ann.name}" for ann in node.annotations] if node.annotations else [],
            fields=self._extract_interface_fields(node),
            methods=self._extract_interface_methods(node),
            constructors=[],
            inner_classes=[],
            static_blocks=[],
            content=interface_content,
            start_line=start_line,
            end_line=end_line,
            javadoc=self._extract_javadoc(lines, start_line),
            type_parameters=[]
        )
    
    def _extract_enums(self, tree: javalang.tree.CompilationUnit,
                      content: str) -> List[JavaClass]:
        """Extract enum definitions"""
        enums = []
        lines = content.split('\n')
        
        for path, node in tree.filter(javalang.tree.EnumDeclaration):
            enum = self._parse_enum_node(node, path, lines)
            if enum:
                enums.append(enum)
        
        return enums
    
    def _parse_enum_node(self, node: javalang.tree.EnumDeclaration,
                        path: Tuple, lines: List[str]) -> JavaClass:
        """Parse enum node"""
        start_line = node.position.line if node.position else 1
        end_line = self._find_end_line(node, lines, start_line)
        
        enum_content = '\n'.join(lines[start_line - 1:end_line])
        
        # Extract enum constants
        constants = []
        for constant in node.body.constants:
            const_element = JavaElement(
                element_type=JavaElementType.FIELD,
                name=constant.name,
                content=constant.name,
                start_line=constant.position.line if constant.position else 0,
                end_line=constant.position.line if constant.position else 0,
                modifiers=["public", "static", "final"],
                annotations=[],
                metadata={'is_enum_constant': True}
            )
            constants.append(const_element)
        
        return JavaClass(
            package=self._extract_package_from_path(path),
            imports=self.imports,
            class_name=node.name,
            class_type=JavaElementType.ENUM,
            extends=None,
            implements=[impl.name for impl in node.implements] if node.implements else [],
            modifiers=node.modifiers if node.modifiers else [],
            annotations=[f"@{ann.name}" for ann in node.annotations] if node.annotations else [],
            fields=constants,
            methods=self._extract_enum_methods(node),
            constructors=self._extract_enum_constructors(node),
            inner_classes=[],
            static_blocks=[],
            content=enum_content,
            start_line=start_line,
            end_line=end_line,
            javadoc=self._extract_javadoc(lines, start_line),
            type_parameters=[]
        )
    
    def _extract_annotation_types(self, tree: javalang.tree.CompilationUnit,
                                 content: str) -> List[JavaClass]:
        """Extract annotation type definitions"""
        annotations = []
        lines = content.split('\n')
        
        for path, node in tree.filter(javalang.tree.AnnotationDeclaration):
            annotation = JavaClass(
                package=self._extract_package_from_path(path),
                imports=self.imports,
                class_name=node.name,
                class_type=JavaElementType.ANNOTATION,
                extends=None,
                implements=[],
                modifiers=node.modifiers if node.modifiers else [],
                annotations=[],
                fields=[],
                methods=[],
                constructors=[],
                inner_classes=[],
                static_blocks=[],
                content="",
                start_line=node.position.line if node.position else 1,
                end_line=0,
                javadoc=None,
                type_parameters=[]
            )
            annotations.append(annotation)
        
        return annotations
    
    def _calculate_complexity(self, tree: javalang.tree.CompilationUnit) -> Dict[str, Any]:
        """Calculate code complexity metrics"""
        complexity = {
            'cyclomatic_complexity': 0,
            'class_count': 0,
            'method_count': 0,
            'field_count': 0,
            'line_count': 0,
            'max_depth': 0,
            'coupling': 0
        }
        
        # Count classes
        for _, node in tree.filter(javalang.tree.TypeDeclaration):
            complexity['class_count'] += 1
        
        # Count and analyze methods
        for _, method in tree.filter(javalang.tree.MethodDeclaration):
            complexity['method_count'] += 1
            complexity['cyclomatic_complexity'] += self._calculate_method_complexity(method)
        
        # Count fields
        for _, field in tree.filter(javalang.tree.FieldDeclaration):
            complexity['field_count'] += 1
        
        # Calculate coupling (simplified - count unique types referenced)
        referenced_types = set()
        for _, node in tree:
            if hasattr(node, 'type') and hasattr(node.type, 'name'):
                referenced_types.add(node.type.name)
        complexity['coupling'] = len(referenced_types)
        
        return complexity
    
    def _calculate_method_complexity(self, method: javalang.tree.MethodDeclaration) -> int:
        """Calculate cyclomatic complexity of a method"""
        complexity = 1  # Base complexity
        
        if not method.body:
            return complexity
        
        # Count decision points
        for _, node in method.filter(javalang.tree.IfStatement):
            complexity += 1
        for _, node in method.filter(javalang.tree.WhileStatement):
            complexity += 1
        for _, node in method.filter(javalang.tree.ForStatement):
            complexity += 1
        for _, node in method.filter(javalang.tree.DoStatement):
            complexity += 1
        for _, node in method.filter(javalang.tree.CatchClause):
            complexity += 1
        for _, node in method.filter(javalang.tree.SwitchStatementCase):
            complexity += 1
        for _, node in method.filter(javalang.tree.TernaryExpression):
            complexity += 1
        
        # Count logical operators
        for _, node in method.filter(javalang.tree.BinaryOperation):
            if node.operator in ['&&', '||']:
                complexity += 1
        
        return complexity
    
    def _calculate_metrics(self, tree: javalang.tree.CompilationUnit,
                          content: str) -> Dict[str, Any]:
        """Calculate various code metrics"""
        lines = content.split('\n')
        
        return {
            'total_lines': len(lines),
            'code_lines': sum(1 for line in lines if line.strip() and not line.strip().startswith('//')),
            'comment_lines': sum(1 for line in lines if line.strip().startswith('//')),
            'blank_lines': sum(1 for line in lines if not line.strip()),
            'import_count': len(list(tree.filter(javalang.tree.Import))),
            'annotation_count': self._count_annotations(tree),
            'lambda_count': self._count_lambdas(tree),
            'stream_api_usage': self._detect_stream_api(tree)
        }
    
    def _count_annotations(self, tree: javalang.tree.CompilationUnit) -> int:
        """Count total annotations used"""
        count = 0
        for _, node in tree:
            if hasattr(node, 'annotations') and node.annotations:
                count += len(node.annotations)
        return count
    
    def _count_lambdas(self, tree: javalang.tree.CompilationUnit) -> int:
        """Count lambda expressions"""
        count = 0
        for _, node in tree.filter(javalang.tree.LambdaExpression):
            count += 1
        return count
    
    def _detect_stream_api(self, tree: javalang.tree.CompilationUnit) -> bool:
        """Detect if Stream API is used"""
        for _, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member in ['stream', 'parallelStream', 'filter', 'map', 
                              'reduce', 'collect', 'forEach', 'flatMap']:
                return True
        return False
    
    def _analyze_dependencies(self, tree: javalang.tree.CompilationUnit) -> Dict[str, List[str]]:
        """Analyze class dependencies"""
        dependencies = {
            'imports': [],
            'extends': [],
            'implements': [],
            'field_types': [],
            'method_return_types': [],
            'method_parameter_types': [],
            'annotations_used': []
        }
        
        # Imports
        for import_decl in tree.imports:
            dependencies['imports'].append(import_decl.path)
        
        # Class relationships
        for _, node in tree.filter(javalang.tree.ClassDeclaration):
            if node.extends:
                dependencies['extends'].append(node.extends.name)
            for impl in node.implements or []:
                dependencies['implements'].append(impl.name)
        
        # Field types
        for _, field in tree.filter(javalang.tree.FieldDeclaration):
            if hasattr(field.type, 'name'):
                dependencies['field_types'].append(field.type.name)
        
        # Method signatures
        for _, method in tree.filter(javalang.tree.MethodDeclaration):
            if method.return_type and hasattr(method.return_type, 'name'):
                dependencies['method_return_types'].append(method.return_type.name)
            for param in method.parameters or []:
                if hasattr(param.type, 'name'):
                    dependencies['method_parameter_types'].append(param.type.name)
        
        # Annotations
        for _, node in tree:
            if hasattr(node, 'annotations') and node.annotations:
                for ann in node.annotations:
                    dependencies['annotations_used'].append(ann.name)
        
        # Remove duplicates
        for key in dependencies:
            dependencies[key] = list(set(dependencies[key]))
        
        return dependencies
    
    def _extract_javadoc(self, lines: List[str], start_line: int) -> Optional[str]:
        """Extract Javadoc comment before element"""
        if start_line <= 1:
            return None
        
        # Look backwards for Javadoc
        javadoc_lines = []
        i = start_line - 2  # Start from line before element
        
        while i >= 0:
            line = lines[i].strip()
            if line.endswith('*/'):
                javadoc_lines.insert(0, lines[i])
                if '/**' in line:
                    break
                i -= 1
                while i >= 0:
                    javadoc_lines.insert(0, lines[i])
                    if '/**' in lines[i]:
                        break
                    i -= 1
                break
            elif not line or line.startswith('//'):
                i -= 1
            else:
                break
        
        if javadoc_lines:
            return '\n'.join(javadoc_lines)
        
        return None
    
    def _find_end_line(self, node: Any, lines: List[str], start_line: int) -> int:
        """Find the end line of a node"""
        # Simple heuristic: count braces
        brace_count = 0
        in_string = False
        in_comment = False
        
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            
            for j, char in enumerate(line):
                # Handle strings
                if char == '"' and (j == 0 or line[j-1] != '\\'):
                    in_string = not in_string
                
                # Handle comments
                if not in_string:
                    if j < len(line) - 1 and line[j:j+2] == '//':
                        break  # Rest of line is comment
                    elif j < len(line) - 1 and line[j:j+2] == '/*':
                        in_comment = True
                    elif j < len(line) - 1 and line[j:j+2] == '*/':
                        in_comment = False
                
                # Count braces
                if not in_string and not in_comment:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        
                        if brace_count == 0:
                            return i + 1
        
        return len(lines)
    
    def _reconstruct_field(self, field: javalang.tree.FieldDeclaration,
                          declarator: javalang.tree.VariableDeclarator) -> str:
        """Reconstruct field declaration"""
        parts = []
        
        if field.modifiers:
            parts.extend(field.modifiers)
        
        parts.append(str(field.type.name if hasattr(field.type, 'name') else field.type))
        parts.append(declarator.name)
        
        if declarator.initializer:
            parts.append('=')
            parts.append('...')  # Simplified
        
        return ' '.join(parts) + ';'
    
    def _reconstruct_method_signature(self, method: javalang.tree.MethodDeclaration) -> str:
        """Reconstruct method signature"""
        parts = []
        
        if method.modifiers:
            parts.extend(method.modifiers)
        
        if method.return_type:
            parts.append(str(method.return_type.name if hasattr(method.return_type, 'name') else method.return_type))
        else:
            parts.append('void')
        
        parts.append(method.name)
        
        # Parameters
        params = []
        for param in method.parameters or []:
            param_str = f"{param.type.name if hasattr(param.type, 'name') else param.type} {param.name}"
            params.append(param_str)
        
        parts.append(f"({', '.join(params)})")
        
        # Throws clause
        if method.throws:
            parts.append('throws')
            parts.append(', '.join(method.throws))
        
        return ' '.join(parts)
    
    def _reconstruct_constructor_signature(self, constructor: javalang.tree.ConstructorDeclaration) -> str:
        """Reconstruct constructor signature"""
        parts = []
        
        if constructor.modifiers:
            parts.extend(constructor.modifiers)
        
        parts.append(constructor.name)
        
        # Parameters
        params = []
        for param in constructor.parameters or []:
            param_str = f"{param.type.name if hasattr(param.type, 'name') else param.type} {param.name}"
            params.append(param_str)
        
        parts.append(f"({', '.join(params)})")
        
        # Throws clause
        if constructor.throws:
            parts.append('throws')
            parts.append(', '.join(constructor.throws))
        
        return ' '.join(parts)
    
    def _extract_parameters(self, method_or_constructor: Any) -> List[Dict[str, str]]:
        """Extract parameter information"""
        params = []
        
        for param in method_or_constructor.parameters or []:
            params.append({
                'name': param.name,
                'type': param.type.name if hasattr(param.type, 'name') else str(param.type),
                'modifiers': param.modifiers if param.modifiers else []
            })
        
        return params
    
    def _get_visibility(self, modifiers: Optional[List[str]]) -> str:
        """Get visibility level from modifiers"""
        if not modifiers:
            return 'package-private'
        
        for mod in modifiers:
            if mod in ['public', 'private', 'protected']:
                return mod
        
        return 'package-private'
    
    def _extract_package_from_path(self, path: Tuple) -> Optional[str]:
        """Extract package from node path"""
        # In javalang, path contains parent nodes
        # This is simplified - would need proper implementation
        return self.current_package
    
    def _extract_interface_fields(self, node: javalang.tree.InterfaceDeclaration) -> List[JavaElement]:
        """Extract interface field declarations (constants)"""
        fields = []
        
        for field in node.fields:
            for declarator in field.declarators:
                field_element = JavaElement(
                    element_type=JavaElementType.FIELD,
                    name=declarator.name,
                    content=self._reconstruct_field(field, declarator),
                    start_line=field.position.line if field.position else 0,
                    end_line=field.position.line if field.position else 0,
                    modifiers=['public', 'static', 'final'],  # Interface fields are always public static final
                    annotations=[],
                    metadata={'is_interface_constant': True}
                )
                fields.append(field_element)
        
        return fields
    
    def _extract_interface_methods(self, node: javalang.tree.InterfaceDeclaration) -> List[JavaElement]:
        """Extract interface method declarations"""
        methods = []
        
        for method in node.methods:
            # Determine if default or static method
            is_default = 'default' in (method.modifiers or [])
            is_static = 'static' in (method.modifiers or [])
            
            method_element = JavaElement(
                element_type=JavaElementType.METHOD,
                name=method.name,
                content=self._reconstruct_method_signature(method),
                start_line=method.position.line if method.position else 0,
                end_line=0,
                modifiers=method.modifiers if method.modifiers else ['public', 'abstract'],
                annotations=[],
                metadata={
                    'is_default': is_default,
                    'is_static': is_static,
                    'is_interface_method': True
                }
            )
            methods.append(method_element)
        
        return methods
    
    def _extract_enum_methods(self, node: javalang.tree.EnumDeclaration) -> List[JavaElement]:
        """Extract enum method declarations"""
        methods = []
        
        if hasattr(node.body, 'methods'):
            for method in node.body.methods:
                method_element = JavaElement(
                    element_type=JavaElementType.METHOD,
                    name=method.name,
                    content=self._reconstruct_method_signature(method),
                    start_line=method.position.line if method.position else 0,
                    end_line=0,
                    modifiers=method.modifiers if method.modifiers else [],
                    annotations=[],
                    metadata={'is_enum_method': True}
                )
                methods.append(method_element)
        
        return methods
    
    def _extract_enum_constructors(self, node: javalang.tree.EnumDeclaration) -> List[JavaElement]:
        """Extract enum constructor declarations"""
        constructors = []
        
        if hasattr(node.body, 'constructors'):
            for constructor in node.body.constructors:
                constructor_element = JavaElement(
                    element_type=JavaElementType.CONSTRUCTOR,
                    name=node.name,
                    content=self._reconstruct_constructor_signature(constructor),
                    start_line=constructor.position.line if constructor.position else 0,
                    end_line=0,
                    modifiers=['private'],  # Enum constructors are always private
                    annotations=[],
                    metadata={'is_enum_constructor': True}
                )
                constructors.append(constructor_element)
        
        return constructors
    
    def _fallback_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback analysis using regex when parsing fails"""
        logger.info("Using fallback regex-based Java analysis")
        
        lines = content.split('\n')
        
        # Extract package
        package = None
        package_match = re.search(r'^package\s+([\w\.]+);', content, re.MULTILINE)
        if package_match:
            package = package_match.group(1)
        
        # Extract imports
        imports = re.findall(r'^import\s+(?:static\s+)?([\w\.\*]+);', content, re.MULTILINE)
        
        # Extract class names
        classes = re.findall(r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)', content)
        interfaces = re.findall(r'(?:public\s+)?interface\s+(\w+)', content)
        enums = re.findall(r'(?:public\s+)?enum\s+(\w+)', content)
        
        # Extract method signatures
        methods = re.findall(
            r'(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(?:synchronized)?\s*'
            r'(?:[\w<>\[\]]+)\s+(\w+)\s*\([^)]*\)',
            content
        )
        
        return {
            'package': package,
            'imports': imports,
            'classes': [{'name': c, 'type': 'class'} for c in classes],
            'interfaces': [{'name': i, 'type': 'interface'} for i in interfaces],
            'enums': [{'name': e, 'type': 'enum'} for e in enums],
            'methods': methods,
            'annotations': [],
            'complexity': {'cyclomatic_complexity': len(methods)},
            'metrics': {
                'total_lines': len(lines),
                'import_count': len(imports),
                'class_count': len(classes) + len(interfaces) + len(enums)
            },
            'dependencies': {'imports': imports}
        }

class JavaChunker(BaseChunker):
    """Chunker specialized for Java source files"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, ChunkerConfig(max_tokens=max_tokens))
        self.analyzer = JavaSyntaxAnalyzer()
        
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from Java source file
        
        Args:
            content: Java source code
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Analyze Java structure
            analysis = self.analyzer.analyze_java_file(content)
            
            chunks = []
            
            # Create package/import chunk
            header_chunk = self._create_header_chunk(content, analysis, file_context)
            if header_chunk:
                chunks.append(header_chunk)
            
            # Create chunks for each class
            for java_class in analysis.get('classes', []):
                class_chunks = self._create_class_chunks(java_class, file_context)
                chunks.extend(class_chunks)
            
            # Create chunks for interfaces
            for interface in analysis.get('interfaces', []):
                interface_chunks = self._create_interface_chunks(interface, file_context)
                chunks.extend(interface_chunks)
            
            # Create chunks for enums
            for enum in analysis.get('enums', []):
                enum_chunks = self._create_enum_chunks(enum, file_context)
                chunks.extend(enum_chunks)
            
            # If no structured chunks created, fallback to line-based
            if not chunks:
                chunks = self._fallback_chunking(content, file_context)
            
            logger.info(f"Created {len(chunks)} chunks for Java file {file_context.path}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking Java file {file_context.path}: {e}")
            return self._fallback_chunking(content, file_context)
    
    def _create_header_chunk(self, content: str, analysis: Dict[str, Any],
                           file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for package and imports"""
        header_lines = []
        
        # Add package declaration
        if analysis.get('package'):
            header_lines.append(f"package {analysis['package']};")
            header_lines.append("")
        
        # Add imports
        for import_stmt in analysis.get('imports', []):
            if import_stmt.startswith('static '):
                header_lines.append(f"import static {import_stmt[7:]};")
            else:
                header_lines.append(f"import {import_stmt};")
        
        if not header_lines:
            return None
        
        header_content = '\n'.join(header_lines)
        
        return self.create_chunk(
            content=header_content,
            chunk_type='java_header',
            metadata={
                'package': analysis.get('package'),
                'import_count': len(analysis.get('imports', [])),
                'imports': analysis.get('imports', [])
            },
            file_path=str(file_context.path)
        )
    
    def _create_class_chunks(self, java_class: JavaClass,
                           file_context: FileContext) -> List[Chunk]:
        """Create chunks for a Java class"""
        chunks = []
        
        # Create class declaration chunk
        class_chunk = self._create_class_declaration_chunk(java_class, file_context)
        if class_chunk:
            chunks.append(class_chunk)
        
        # Create field chunk if there are many fields
        if len(java_class.fields) > 10:
            field_chunk = self._create_fields_chunk(java_class, file_context)
            if field_chunk:
                chunks.append(field_chunk)
        
        # Create chunks for methods
        for method in java_class.methods:
            method_chunk = self._create_method_chunk(method, java_class, file_context)
            if method_chunk:
                chunks.append(method_chunk)
        
        # Create chunks for constructors
        for constructor in java_class.constructors:
            constructor_chunk = self._create_constructor_chunk(
                constructor, java_class, file_context
            )
            if constructor_chunk:
                chunks.append(constructor_chunk)
        
        # Create chunks for inner classes
        for inner_class in java_class.inner_classes:
            inner_chunks = self._create_class_chunks(inner_class, file_context)
            chunks.extend(inner_chunks)
        
        return chunks
    
    def _create_class_declaration_chunk(self, java_class: JavaClass,
                                       file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for class declaration and structure"""
        lines = []
        
        # Add Javadoc if present
        if java_class.javadoc:
            lines.append(java_class.javadoc)
        
        # Add annotations
        for annotation in java_class.annotations:
            lines.append(annotation)
        
        # Build class declaration
        class_decl_parts = []
        if java_class.modifiers:
            class_decl_parts.extend(java_class.modifiers)
        
        class_decl_parts.append('class' if java_class.class_type == JavaElementType.CLASS else str(java_class.class_type.value))
        class_decl_parts.append(java_class.class_name)
        
        # Add generics
        if java_class.type_parameters:
            class_decl_parts.append(f"<{', '.join(java_class.type_parameters)}>")
        
        # Add extends
        if java_class.extends:
            class_decl_parts.append(f"extends {java_class.extends}")
        
        # Add implements
        if java_class.implements:
            class_decl_parts.append(f"implements {', '.join(java_class.implements)}")
        
        class_decl = ' '.join(class_decl_parts) + " {"
        lines.append(class_decl)
        
        # Add fields (if not too many)
        if len(java_class.fields) <= 10:
            for field in java_class.fields:
                lines.append(f"    {field.content}")
        else:
            lines.append(f"    // {len(java_class.fields)} fields omitted")
        
        # Add method signatures
        lines.append("")
        lines.append("    // Methods:")
        for method in java_class.methods:
            lines.append(f"    {method.content};")
        
        lines.append("}")
        
        chunk_content = '\n'.join(lines)
        
        if self.count_tokens(chunk_content) <= self.max_tokens:
            return self.create_chunk(
                content=chunk_content,
                chunk_type='java_class_declaration',
                metadata={
                    'class_name': java_class.class_name,
                    'class_type': java_class.class_type.value,
                    'package': java_class.package,
                    'extends': java_class.extends,
                    'implements': java_class.implements,
                    'field_count': len(java_class.fields),
                    'method_count': len(java_class.methods),
                    'constructor_count': len(java_class.constructors),
                    'has_inner_classes': len(java_class.inner_classes) > 0
                },
                file_path=str(file_context.path),
                start_line=java_class.start_line,
                end_line=java_class.end_line
            )
        
        return None
    
    def _create_fields_chunk(self, java_class: JavaClass,
                           file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for class fields"""
        lines = [f"// Fields for class {java_class.class_name}"]
        
        for field in java_class.fields:
            field_line = ' '.join(field.modifiers) + ' ' if field.modifiers else ''
            field_line += field.content
            lines.append(field_line)
        
        chunk_content = '\n'.join(lines)
        
        return self.create_chunk(
            content=chunk_content,
            chunk_type='java_fields',
            metadata={
                'class_name': java_class.class_name,
                'field_count': len(java_class.fields),
                'static_fields': sum(1 for f in java_class.fields if 'static' in f.metadata),
                'final_fields': sum(1 for f in java_class.fields if 'final' in f.metadata)
            },
            file_path=str(file_context.path)
        )
    
    def _create_method_chunk(self, method: JavaElement, java_class: JavaClass,
                           file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for a method with context"""
        lines = []
        
        # Add context comment
        lines.append(f"// Package: {java_class.package}")
        lines.append(f"// Class: {java_class.class_name}")
        if java_class.extends:
            lines.append(f"// Extends: {java_class.extends}")
        if java_class.implements:
            lines.append(f"// Implements: {', '.join(java_class.implements)}")
        lines.append("")
        
        # Add method annotations
        for annotation in method.annotations:
            lines.append(annotation)
        
        # Add method signature and body placeholder
        method_signature = ' '.join(method.modifiers) + ' ' if method.modifiers else ''
        method_signature += method.content
        
        lines.append(method_signature + " {")
        lines.append("    // Method implementation")
        lines.append("}")
        
        chunk_content = '\n'.join(lines)
        
        return self.create_chunk(
            content=chunk_content,
            chunk_type='java_method',
            metadata={
                'class_name': java_class.class_name,
                'method_name': method.name,
                'visibility': method.metadata.get('visibility', 'package-private'),
                'is_static': method.metadata.get('static', False),
                'is_abstract': method.metadata.get('abstract', False),
                'return_type': method.metadata.get('return_type', 'void'),
                'parameter_count': len(method.metadata.get('parameters', [])),
                'throws': method.metadata.get('throws', []),
                'complexity': method.metadata.get('complexity', 1)
            },
            file_path=str(file_context.path),
            start_line=method.start_line,
            end_line=method.end_line
        )
    
    def _create_constructor_chunk(self, constructor: JavaElement, java_class: JavaClass,
                                file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for a constructor"""
        lines = []
        
        # Add context
        lines.append(f"// Class: {java_class.class_name}")
        lines.append("")
        
        # Add constructor
        for annotation in constructor.annotations:
            lines.append(annotation)
        
        constructor_signature = ' '.join(constructor.modifiers) + ' ' if constructor.modifiers else ''
        constructor_signature += constructor.content
        
        lines.append(constructor_signature + " {")
        lines.append("    // Constructor implementation")
        lines.append("}")
        
        chunk_content = '\n'.join(lines)
        
        return self.create_chunk(
            content=chunk_content,
            chunk_type='java_constructor',
            metadata={
                'class_name': java_class.class_name,
                'visibility': constructor.metadata.get('visibility', 'package-private'),
                'parameter_count': len(constructor.metadata.get('parameters', [])),
                'throws': constructor.metadata.get('throws', [])
            },
            file_path=str(file_context.path),
            start_line=constructor.start_line,
            end_line=constructor.end_line
        )
    
    def _create_interface_chunks(self, interface: JavaClass,
                               file_context: FileContext) -> List[Chunk]:
        """Create chunks for an interface"""
        chunks = []
        
        # Interface declaration chunk
        lines = []
        
        if interface.javadoc:
            lines.append(interface.javadoc)
        
        for annotation in interface.annotations:
            lines.append(annotation)
        
        interface_decl = ' '.join(interface.modifiers) if interface.modifiers else ''
        interface_decl += f" interface {interface.class_name}"
        
        if interface.extends:
            interface_decl += f" extends {', '.join(interface.extends)}"
        
        lines.append(interface_decl + " {")
        
        # Add constants
        for field in interface.fields:
            lines.append(f"    {field.content}")
        
        # Add method signatures
        for method in interface.methods:
            lines.append(f"    {method.content};")
        
        lines.append("}")
        
        chunk_content = '\n'.join(lines)
        
        if self.count_tokens(chunk_content) <= self.max_tokens:
            chunks.append(self.create_chunk(
                content=chunk_content,
                chunk_type='java_interface',
                metadata={
                    'interface_name': interface.class_name,
                    'extends': interface.extends,
                    'constant_count': len(interface.fields),
                    'method_count': len(interface.methods),
                    'package': interface.package
                },
                file_path=str(file_context.path),
                start_line=interface.start_line,
                end_line=interface.end_line
            ))
        
        return chunks
    
    def _create_enum_chunks(self, enum: JavaClass,
                          file_context: FileContext) -> List[Chunk]:
        """Create chunks for an enum"""
        chunks = []
        
        lines = []
        
        if enum.javadoc:
            lines.append(enum.javadoc)
        
        for annotation in enum.annotations:
            lines.append(annotation)
        
        enum_decl = ' '.join(enum.modifiers) if enum.modifiers else ''
        enum_decl += f" enum {enum.class_name}"
        
        if enum.implements:
            enum_decl += f" implements {', '.join(enum.implements)}"
        
        lines.append(enum_decl + " {")
        
        # Add enum constants
        constants = [f.name for f in enum.fields if f.metadata.get('is_enum_constant')]
        if constants:
            lines.append(f"    {', '.join(constants)};")
        
        # Add fields
        regular_fields = [f for f in enum.fields if not f.metadata.get('is_enum_constant')]
        for field in regular_fields:
            lines.append(f"    {field.content}")
        
        # Add methods
        for method in enum.methods:
            lines.append(f"    {method.content};")
        
        lines.append("}")
        
        chunk_content = '\n'.join(lines)
        
        if self.count_tokens(chunk_content) <= self.max_tokens:
            chunks.append(self.create_chunk(
                content=chunk_content,
                chunk_type='java_enum',
                metadata={
                    'enum_name': enum.class_name,
                    'constant_count': len(constants),
                    'implements': enum.implements,
                    'method_count': len(enum.methods),
                    'package': enum.package
                },
                file_path=str(file_context.path),
                start_line=enum.start_line,
                end_line=enum.end_line
            ))
        
        return chunks
    
    def _fallback_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback line-based chunking for Java files"""
        logger.warning(f"Using fallback chunking for Java file {file_context.path}")
        
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        in_method = False
        brace_count = 0
        
        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line)
            
            # Track method boundaries
            if re.match(r'.*\s+(public|private|protected).*\s+\w+\s*\([^)]*\)', line):
                in_method = True
                brace_count = 0
            
            if in_method:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0 and '{' in line:
                    in_method = False
            
            # Check if we should start new chunk
            should_split = (
                current_tokens + line_tokens > self.max_tokens or
                (not in_method and brace_count == 0 and current_tokens > self.max_tokens * 0.7)
            )
            
            if should_split and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='java_fallback',
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
                chunk_type='java_fallback',
                metadata={
                    'is_fallback': True,
                    'line_count': len(current_chunk)
                },
                file_path=str(file_context.path)
            ))
        
        return chunks