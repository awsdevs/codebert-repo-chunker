"""
JavaScript/TypeScript chunker for intelligent semantic chunking
Handles ES6+, TypeScript, React/JSX, Vue, Node.js, and modern JavaScript patterns
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from src.utils.logger import get_logger
from enum import Enum
import esprima
import subprocess

from src.core.base_chunker import BaseChunker, Chunk, ChunkerConfig
from src.core.file_context import FileContext
from config.settings import settings

logger = get_logger(__name__)

class JSElementType(Enum):
    """Types of JavaScript/TypeScript elements"""
    MODULE = "module"
    IMPORT = "import"
    EXPORT = "export"
    CLASS = "class"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    ENUM = "enum"
    FUNCTION = "function"
    ARROW_FUNCTION = "arrow_function"
    ASYNC_FUNCTION = "async_function"
    GENERATOR = "generator"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    GETTER = "getter"
    SETTER = "setter"
    VARIABLE = "variable"
    CONSTANT = "constant"
    COMPONENT = "component"
    HOOK = "hook"
    MIDDLEWARE = "middleware"
    REDUCER = "reducer"
    ACTION = "action"
    DECORATOR = "decorator"

@dataclass
class JSElement:
    """Represents a JavaScript/TypeScript code element"""
    element_type: JSElementType
    name: str
    content: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    is_exported: bool
    is_default: bool
    is_async: bool
    is_generator: bool
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    decorators: List[str]
    parent: Optional['JSElement'] = None
    children: List['JSElement'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class JSModule:
    """Represents a JavaScript/TypeScript module"""
    file_type: str  # 'javascript', 'typescript', 'jsx', 'tsx'
    module_type: str  # 'commonjs', 'esm', 'umd', 'amd'
    imports: List[Dict[str, Any]]
    exports: List[Dict[str, Any]]
    classes: List[JSElement]
    functions: List[JSElement]
    components: List[JSElement]
    interfaces: List[JSElement]
    types: List[JSElement]
    variables: List[JSElement]
    constants: List[JSElement]
    hooks: List[JSElement]
    has_jsx: bool
    has_typescript: bool
    framework: Optional[str]  # 'react', 'vue', 'angular', 'svelte', etc.
    dependencies: Set[str]

class JavaScriptAnalyzer:
    """Analyzes JavaScript/TypeScript syntax and structure"""
    
    # React patterns
    REACT_PATTERNS = {
        'component_class': re.compile(r'class\s+(\w+)\s+extends\s+(?:React\.)?(?:Component|PureComponent)'),
        'component_function': re.compile(r'(?:export\s+)?(?:default\s+)?(?:const|function)\s+(\w+)\s*[=:]\s*(?:\([^)]*\)|[^=]+)\s*=>\s*(?:\(|<|{)'),
        'hook': re.compile(r'(?:const|let)\s+(use[A-Z]\w*)\s*='),
        'hook_call': re.compile(r'\buse[A-Z]\w*\s*\('),
        'jsx_element': re.compile(r'<([A-Z]\w*)[^>]*>'),
        'jsx_fragment': re.compile(r'<(?:React\.)?Fragment>|<>'),
        'memo': re.compile(r'React\.memo\s*\('),
        'forward_ref': re.compile(r'React\.forwardRef\s*\('),
        'context': re.compile(r'React\.createContext\s*\('),
        'styled_component': re.compile(r'styled\.\w+`|styled\('),
    }
    
    # Vue patterns
    VUE_PATTERNS = {
        'component': re.compile(r'export\s+default\s+(?:defineComponent\s*\()?{'),
        'composition_api': re.compile(r'setup\s*\([^)]*\)\s*{'),
        'ref': re.compile(r'ref\s*\([^)]*\)'),
        'reactive': re.compile(r'reactive\s*\([^)]*\)'),
        'computed': re.compile(r'computed\s*\([^)]*\)'),
        'watch': re.compile(r'watch\s*\([^)]*\)'),
        'template': re.compile(r'<template[^>]*>'),
        'script_setup': re.compile(r'<script\s+setup[^>]*>'),
    }
    
    # Angular patterns
    ANGULAR_PATTERNS = {
        'component': re.compile(r'@Component\s*\({'),
        'service': re.compile(r'@Injectable\s*\({'),
        'directive': re.compile(r'@Directive\s*\({'),
        'pipe': re.compile(r'@Pipe\s*\({'),
        'module': re.compile(r'@NgModule\s*\({'),
        'input': re.compile(r'@Input\s*\(\)'),
        'output': re.compile(r'@Output\s*\(\)'),
    }
    
    # Node.js patterns
    NODE_PATTERNS = {
        'express_router': re.compile(r'(?:const|let|var)\s+\w+\s*=\s*(?:express\.)?Router\s*\('),
        'express_middleware': re.compile(r'app\.(?:use|get|post|put|delete|patch)\s*\('),
        'require': re.compile(r'require\s*\([\'"`]([^\'"`]+)[\'"`]\)'),
        'module_exports': re.compile(r'module\.exports\s*='),
        'exports': re.compile(r'exports\.\w+\s*='),
        'async_handler': re.compile(r'async\s+\([^)]*\)\s*=>\s*{'),
        'event_emitter': re.compile(r'\.(?:on|once|emit)\s*\('),
    }
    
    # TypeScript patterns
    TYPESCRIPT_PATTERNS = {
        'interface': re.compile(r'(?:export\s+)?interface\s+(\w+)(?:<[^>]+>)?\s*{'),
        'type_alias': re.compile(r'(?:export\s+)?type\s+(\w+)(?:<[^>]+>)?\s*='),
        'enum': re.compile(r'(?:export\s+)?enum\s+(\w+)\s*{'),
        'namespace': re.compile(r'(?:export\s+)?namespace\s+(\w+)\s*{'),
        'decorator': re.compile(r'@(\w+)(?:\([^)]*\))?'),
        'generic': re.compile(r'<([^>]+)>'),
        'type_annotation': re.compile(r':\s*([^=,;{}\n]+)'),
        'as_const': re.compile(r'as\s+const'),
        'readonly': re.compile(r'\breadonly\s+'),
        'abstract': re.compile(r'\babstract\s+'),
    }
    
    # Modern JavaScript patterns
    MODERN_JS_PATTERNS = {
        'arrow_function': re.compile(r'(?:const|let|var)?\s*(\w+)?\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=]+)\s*=>'),
        'async_function': re.compile(r'async\s+(?:function\s+)?(\w+)?\s*\('),
        'generator': re.compile(r'function\s*\*\s*(\w+)?\s*\('),
        'class': re.compile(r'(?:export\s+)?(?:default\s+)?class\s+(\w+)'),
        'method': re.compile(r'(?:async\s+)?(?:static\s+)?(\w+)\s*\([^)]*\)\s*{'),
        'getter': re.compile(r'get\s+(\w+)\s*\(\)\s*{'),
        'setter': re.compile(r'set\s+(\w+)\s*\([^)]*\)\s*{'),
        'destructuring': re.compile(r'(?:const|let|var)\s*{([^}]+)}\s*='),
        'spread': re.compile(r'\.\.\.(\w+)'),
        'template_literal': re.compile(r'`[^`]*\${[^}]+}[^`]*`'),
        'optional_chaining': re.compile(r'\?\.\w+'),
        'nullish_coalescing': re.compile(r'\?\?'),
    }
    
    def __init__(self):
        self.current_module = None
        self.parse_errors = []
        
    def analyze_javascript_file(self, content: str, file_path: Path) -> Dict[str, Any]:
        """
        Analyze JavaScript/TypeScript file structure
        
        Args:
            content: File content
            file_path: Path to file
            
        Returns:
            Analysis results
        """
        file_type = self._detect_file_type(file_path, content)
        
        # Try parsing with esprima for JavaScript
        if file_type in ['javascript', 'jsx']:
            try:
                return self._analyze_with_esprima(content, file_type)
            except Exception as e:
                logger.warning(f"Esprima parsing failed: {e}, using pattern matching")
        
        # For TypeScript or if esprima fails, use pattern matching
        return self._analyze_with_patterns(content, file_type)
    
    def _detect_file_type(self, file_path: Path, content: str) -> str:
        """Detect JavaScript variant"""
        ext = file_path.suffix.lower()
        
        if ext == '.ts':
            return 'typescript'
        elif ext == '.tsx':
            return 'tsx'
        elif ext == '.jsx':
            return 'jsx'
        elif ext in ['.js', '.mjs', '.cjs']:
            # Check for JSX in content
            if self.REACT_PATTERNS['jsx_element'].search(content):
                return 'jsx'
            return 'javascript'
        
        return 'javascript'
    
    def _analyze_with_esprima(self, content: str, file_type: str) -> Dict[str, Any]:
        """Analyze using esprima parser"""
        try:
            # Parse with JSX support if needed
            if file_type in ['jsx', 'tsx']:
                ast = esprima.parse(content, {'jsx': True, 'tolerant': True, 'loc': True})
            else:
                ast = esprima.parse(content, {'tolerant': True, 'loc': True})
            
            return self._extract_from_ast(ast, content)
            
        except Exception as e:
            logger.error(f"AST parsing failed: {e}")
            raise
    
    def _extract_from_ast(self, ast: Any, content: str) -> Dict[str, Any]:
        """Extract structure from AST"""
        module = JSModule(
            file_type='javascript',
            module_type='esm',
            imports=[],
            exports=[],
            classes=[],
            functions=[],
            components=[],
            interfaces=[],
            types=[],
            variables=[],
            constants=[],
            hooks=[],
            has_jsx=False,
            has_typescript=False,
            framework=None,
            dependencies=set()
        )
        
        # Walk AST and extract elements
        self._walk_ast(ast, module, content)
        
        # Detect framework
        module.framework = self._detect_framework(module, content)
        
        # Analyze complexity
        complexity = self._calculate_complexity(module)
        
        return {
            'module': module,
            'complexity': complexity,
            'metrics': self._calculate_metrics(module, content)
        }
    
    def _walk_ast(self, node: Any, module: JSModule, content: str, parent: Optional[JSElement] = None):
        """Recursively walk AST nodes"""
        if not node:
            return
        
        node_type = node.type if hasattr(node, 'type') else None
        
        if node_type == 'ImportDeclaration':
            self._extract_import(node, module)
        
        elif node_type == 'ExportNamedDeclaration' or node_type == 'ExportDefaultDeclaration':
            self._extract_export(node, module)
        
        elif node_type == 'ClassDeclaration' or node_type == 'ClassExpression':
            self._extract_class(node, module, content)
        
        elif node_type == 'FunctionDeclaration' or node_type == 'FunctionExpression':
            self._extract_function(node, module, content)
        
        elif node_type == 'ArrowFunctionExpression':
            self._extract_arrow_function(node, module, content)
        
        elif node_type == 'VariableDeclaration':
            self._extract_variable(node, module, content)
        
        # Recursively process child nodes
        if hasattr(node, 'body'):
            if isinstance(node.body, list):
                for child in node.body:
                    self._walk_ast(child, module, content, parent)
            else:
                self._walk_ast(node.body, module, content, parent)
    
    def _extract_import(self, node: Any, module: JSModule):
        """Extract import statement"""
        import_info = {
            'source': node.source.value if hasattr(node, 'source') else None,
            'specifiers': [],
            'line': node.loc.start.line if hasattr(node, 'loc') else 0
        }
        
        if hasattr(node, 'specifiers'):
            for spec in node.specifiers:
                if spec.type == 'ImportDefaultSpecifier':
                    import_info['specifiers'].append({
                        'type': 'default',
                        'name': spec.local.name
                    })
                elif spec.type == 'ImportSpecifier':
                    import_info['specifiers'].append({
                        'type': 'named',
                        'imported': spec.imported.name,
                        'local': spec.local.name
                    })
                elif spec.type == 'ImportNamespaceSpecifier':
                    import_info['specifiers'].append({
                        'type': 'namespace',
                        'name': spec.local.name
                    })
        
        module.imports.append(import_info)
        
        # Track dependency
        if import_info['source']:
            module.dependencies.add(import_info['source'])
    
    def _extract_export(self, node: Any, module: JSModule):
        """Extract export statement"""
        export_info = {
            'type': 'default' if node.type == 'ExportDefaultDeclaration' else 'named',
            'declaration': None,
            'line': node.loc.start.line if hasattr(node, 'loc') else 0
        }
        
        if hasattr(node, 'declaration'):
            if hasattr(node.declaration, 'name'):
                export_info['declaration'] = node.declaration.name
            elif hasattr(node.declaration, 'id'):
                export_info['declaration'] = node.declaration.id.name
        
        module.exports.append(export_info)
    
    def _extract_class(self, node: Any, module: JSModule, content: str):
        """Extract class declaration"""
        class_element = JSElement(
            element_type=JSElementType.CLASS,
            name=node.id.name if hasattr(node, 'id') else 'anonymous',
            content=self._get_node_content(node, content),
            start_line=node.loc.start.line if hasattr(node, 'loc') else 0,
            end_line=node.loc.end.line if hasattr(node, 'loc') else 0,
            start_column=node.loc.start.column if hasattr(node, 'loc') else 0,
            end_column=node.loc.end.column if hasattr(node, 'loc') else 0,
            is_exported=False,
            is_default=False,
            is_async=False,
            is_generator=False,
            parameters=[],
            return_type=None,
            decorators=[],
            metadata={
                'extends': node.superClass.name if hasattr(node, 'superClass') and hasattr(node.superClass, 'name') else None
            }
        )
        
        # Check if it's a React component
        if class_element.metadata.get('extends') in ['Component', 'PureComponent', 'React.Component', 'React.PureComponent']:
            module.components.append(class_element)
        else:
            module.classes.append(class_element)
        
        # Extract class methods
        if hasattr(node, 'body') and hasattr(node.body, 'body'):
            for item in node.body.body:
                if item.type == 'MethodDefinition':
                    self._extract_method(item, class_element, content)
    
    def _extract_method(self, node: Any, parent: JSElement, content: str):
        """Extract method from class"""
        method_element = JSElement(
            element_type=self._get_method_type(node),
            name=node.key.name if hasattr(node.key, 'name') else 'anonymous',
            content=self._get_node_content(node, content),
            start_line=node.loc.start.line if hasattr(node, 'loc') else 0,
            end_line=node.loc.end.line if hasattr(node, 'loc') else 0,
            start_column=node.loc.start.column if hasattr(node, 'loc') else 0,
            end_column=node.loc.end.column if hasattr(node, 'loc') else 0,
            is_exported=False,
            is_default=False,
            is_async=getattr(node.value, 'async') if hasattr(node.value, 'async') else False,
            is_generator=node.value.generator if hasattr(node.value, 'generator') else False,
            parameters=self._extract_parameters(node.value),
            return_type=None,
            decorators=[],
            parent=parent,
            metadata={
                'static': node.static if hasattr(node, 'static') else False,
                'kind': node.kind if hasattr(node, 'kind') else 'method'
            }
        )
        
        parent.children.append(method_element)
    
    def _get_method_type(self, node: Any) -> JSElementType:
        """Determine method type"""
        if hasattr(node, 'kind'):
            if node.kind == 'constructor':
                return JSElementType.CONSTRUCTOR
            elif node.kind == 'get':
                return JSElementType.GETTER
            elif node.kind == 'set':
                return JSElementType.SETTER
        
        return JSElementType.METHOD
    
    def _extract_function(self, node: Any, module: JSModule, content: str):
        """Extract function declaration"""
        function_element = JSElement(
            element_type=JSElementType.ASYNC_FUNCTION if getattr(node, 'async', False) else 
                        JSElementType.GENERATOR if getattr(node, 'generator', False) else 
                        JSElementType.FUNCTION,
            name=node.id.name if hasattr(node, 'id') and node.id else 'anonymous',
            content=self._get_node_content(node, content),
            start_line=node.loc.start.line if hasattr(node, 'loc') else 0,
            end_line=node.loc.end.line if hasattr(node, 'loc') else 0,
            start_column=node.loc.start.column if hasattr(node, 'loc') else 0,
            end_column=node.loc.end.column if hasattr(node, 'loc') else 0,
            is_exported=False,
            is_default=False,
            is_async=getattr(node, 'async') if hasattr(node, 'async') else False,
            is_generator=getattr(node, 'generator') if hasattr(node, 'generator') else False,
            parameters=self._extract_parameters(node),
            return_type=None,
            decorators=[],
            metadata={}
        )
        
        # Check if it's a React hook
        if function_element.name.startswith('use') and function_element.name[3].isupper():
            module.hooks.append(function_element)
        # Check if it's a React component
        elif self._is_react_component(function_element, content):
            module.components.append(function_element)
        else:
            module.functions.append(function_element)
    
    def _extract_arrow_function(self, node: Any, module: JSModule, content: str):
        """Extract arrow function expression"""
        # Arrow functions are often assigned to variables
        # This would be handled in _extract_variable
        pass
    
    def _extract_variable(self, node: Any, module: JSModule, content: str):
        """Extract variable declaration"""
        for declarator in node.declarations:
            if declarator.init and hasattr(declarator.init, 'type'):
                # Check if it's an arrow function
                if declarator.init.type == 'ArrowFunctionExpression':
                    element = JSElement(
                        element_type=JSElementType.ARROW_FUNCTION,
                        name=declarator.id.name if hasattr(declarator.id, 'name') else 'anonymous',
                        content=self._get_node_content(node, content),
                        start_line=node.loc.start.line if hasattr(node, 'loc') else 0,
                        end_line=node.loc.end.line if hasattr(node, 'loc') else 0,
                        start_column=node.loc.start.column if hasattr(node, 'loc') else 0,
                        end_column=node.loc.end.column if hasattr(node, 'loc') else 0,
                        is_exported=False,
                        is_default=False,
                        is_async=getattr(declarator.init, 'async', False) if hasattr(declarator.init, 'async') else False,
                        is_generator=False,
                        parameters=self._extract_parameters(declarator.init),
                        return_type=None,
                        decorators=[],
                        metadata={'kind': node.kind}  # const, let, var
                    )
                    
                    # Check if it's a React component or hook
                    if element.name.startswith('use') and element.name[3].isupper():
                        module.hooks.append(element)
                    elif self._is_react_component(element, content):
                        module.components.append(element)
                    else:
                        module.functions.append(element)
                else:
                    # Regular variable
                    element = JSElement(
                        element_type=JSElementType.CONSTANT if node.kind == 'const' else JSElementType.VARIABLE,
                        name=declarator.id.name if hasattr(declarator.id, 'name') else 'anonymous',
                        content=self._get_node_content(node, content),
                        start_line=node.loc.start.line if hasattr(node, 'loc') else 0,
                        end_line=node.loc.end.line if hasattr(node, 'loc') else 0,
                        start_column=node.loc.start.column if hasattr(node, 'loc') else 0,
                        end_column=node.loc.end.column if hasattr(node, 'loc') else 0,
                        is_exported=False,
                        is_default=False,
                        is_async=False,
                        is_generator=False,
                        parameters=[],
                        return_type=None,
                        decorators=[],
                        metadata={'kind': node.kind}
                    )
                    
                    if node.kind == 'const':
                        module.constants.append(element)
                    else:
                        module.variables.append(element)
    
    def _extract_parameters(self, node: Any) -> List[Dict[str, Any]]:
        """Extract function parameters"""
        params = []
        
        if hasattr(node, 'params'):
            for param in node.params:
                param_info = {
                    'name': None,
                    'type': None,
                    'default': None,
                    'rest': False
                }
                
                if hasattr(param, 'name'):
                    param_info['name'] = param.name
                elif hasattr(param, 'left'):
                    # Destructuring parameter
                    param_info['name'] = '{...}'  # Simplified
                
                if param.type == 'RestElement':
                    param_info['rest'] = True
                    if hasattr(param.argument, 'name'):
                        param_info['name'] = param.argument.name
                
                params.append(param_info)
        
        return params
    
    def _get_node_content(self, node: Any, content: str) -> str:
        """Extract node content from source"""
        if hasattr(node, 'loc'):
            lines = content.split('\n')
            start_line = node.loc.start.line - 1
            end_line = node.loc.end.line
            
            if start_line < len(lines) and end_line <= len(lines):
                return '\n'.join(lines[start_line:end_line])
        
        return ''
    
    def _is_react_component(self, element: JSElement, content: str) -> bool:
        """Check if function/class is a React component"""
        # Component names start with capital letter
        if element.name and element.name[0].isupper():
            return True
        
        # Check for JSX in content
        element_content = element.content or content
        if self.REACT_PATTERNS['jsx_element'].search(element_content):
            return True
        
        # Check for React.createElement
        if 'React.createElement' in element_content or 'jsx(' in element_content:
            return True
        
        return False
    
    def _analyze_with_patterns(self, content: str, file_type: str) -> Dict[str, Any]:
        """Analyze using pattern matching (fallback)"""
        module = JSModule(
            file_type=file_type,
            module_type=self._detect_module_type(content),
            imports=self._extract_imports_pattern(content),
            exports=self._extract_exports_pattern(content),
            classes=self._extract_classes_pattern(content),
            functions=self._extract_functions_pattern(content),
            components=self._extract_components_pattern(content),
            interfaces=self._extract_interfaces_pattern(content) if 'ts' in file_type else [],
            types=self._extract_types_pattern(content) if 'ts' in file_type else [],
            variables=[],
            constants=[],
            hooks=self._extract_hooks_pattern(content),
            has_jsx='jsx' in file_type or bool(self.REACT_PATTERNS['jsx_element'].search(content)),
            has_typescript='ts' in file_type,
            framework=self._detect_framework_pattern(content),
            dependencies=self._extract_dependencies_pattern(content)
        )
        
        complexity = self._calculate_complexity(module)
        
        return {
            'module': module,
            'complexity': complexity,
            'metrics': self._calculate_metrics(module, content)
        }
    
    def _detect_module_type(self, content: str) -> str:
        """Detect module system type"""
        if 'export ' in content or 'import ' in content:
            return 'esm'
        elif 'module.exports' in content or 'require(' in content:
            return 'commonjs'
        elif 'define(' in content and 'require(' in content:
            return 'amd'
        else:
            return 'none'
    
    def _extract_imports_pattern(self, content: str) -> List[Dict[str, Any]]:
        """Extract imports using patterns"""
        imports = []
        
        # ES6 imports
        es6_import = re.compile(
            r'import\s+(?:'
            r'(?P<default>\w+)|'
            r'{(?P<named>[^}]+)}|'
            r'\*\s+as\s+(?P<namespace>\w+)|'
            r'(?P<default2>\w+)\s*,\s*{(?P<named2>[^}]+)}'
            r')\s+from\s+[\'"`](?P<source>[^\'"`]+)[\'"`]'
        )
        
        for match in es6_import.finditer(content):
            imports.append({
                'source': match.group('source'),
                'default': match.group('default') or match.group('default2'),
                'named': match.group('named') or match.group('named2'),
                'namespace': match.group('namespace')
            })
        
        # CommonJS requires
        require_pattern = re.compile(r'(?:const|let|var)\s+(\w+)\s*=\s*require\([\'"`]([^\'"`]+)[\'"`]\)')
        for match in require_pattern.finditer(content):
            imports.append({
                'source': match.group(2),
                'default': match.group(1),
                'type': 'commonjs'
            })
        
        return imports
    
    def _extract_exports_pattern(self, content: str) -> List[Dict[str, Any]]:
        """Extract exports using patterns"""
        exports = []
        
        # ES6 exports
        export_patterns = [
            (re.compile(r'export\s+default\s+(\w+)'), 'default'),
            (re.compile(r'export\s+(?:const|let|var|function|class)\s+(\w+)'), 'named'),
            (re.compile(r'export\s+{([^}]+)}'), 'named_list'),
            (re.compile(r'export\s+\*\s+from\s+[\'"`]([^\'"`]+)[\'"`]'), 're-export'),
        ]
        
        for pattern, export_type in export_patterns:
            for match in pattern.finditer(content):
                exports.append({
                    'type': export_type,
                    'name': match.group(1) if match.groups() else None
                })
        
        # CommonJS exports
        if 'module.exports' in content:
            exports.append({'type': 'commonjs', 'name': 'module.exports'})
        
        return exports
    
    def _extract_classes_pattern(self, content: str) -> List[JSElement]:
        """Extract classes using patterns"""
        classes = []
        lines = content.split('\n')
        
        for match in self.MODERN_JS_PATTERNS['class'].finditer(content):
            class_name = match.group(1)
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            # Find class end
            end_line = self._find_block_end(lines, start_line - 1) + 1
            
            classes.append(JSElement(
                element_type=JSElementType.CLASS,
                name=class_name,
                content='',
                start_line=start_line,
                end_line=end_line,
                start_column=0,
                end_column=0,
                is_exported='export' in match.group(0),
                is_default='default' in match.group(0),
                is_async=False,
                is_generator=False,
                parameters=[],
                return_type=None,
                decorators=self._extract_decorators_at_line(content, start_line),
                metadata={}
            ))
        
        return classes
    
    def _extract_functions_pattern(self, content: str) -> List[JSElement]:
        """Extract functions using patterns"""
        functions = []
        lines = content.split('\n')
        
        # Regular functions
        func_pattern = re.compile(
            r'(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*(\*?)\s*(\w+)?\s*\([^)]*\)'
        )
        
        for match in func_pattern.finditer(content):
            is_generator = bool(match.group(1))
            func_name = match.group(2) or 'anonymous'
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            functions.append(JSElement(
                element_type=JSElementType.GENERATOR if is_generator else
                           JSElementType.ASYNC_FUNCTION if 'async' in match.group(0) else
                           JSElementType.FUNCTION,
                name=func_name,
                content='',
                start_line=start_line,
                end_line=start_line,  # Will be updated
                start_column=0,
                end_column=0,
                is_exported='export' in match.group(0),
                is_default='default' in match.group(0),
                is_async='async' in match.group(0),
                is_generator=is_generator,
                parameters=[],
                return_type=None,
                decorators=[],
                metadata={}
            ))
        
        # Arrow functions
        for match in self.MODERN_JS_PATTERNS['arrow_function'].finditer(content):
            func_name = match.group(1) or 'anonymous'
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            functions.append(JSElement(
                element_type=JSElementType.ARROW_FUNCTION,
                name=func_name,
                content='',
                start_line=start_line,
                end_line=start_line,
                start_column=0,
                end_column=0,
                is_exported=False,
                is_default=False,
                is_async='async' in match.group(0),
                is_generator=False,
                parameters=[],
                return_type=None,
                decorators=[],
                metadata={}
            ))
        
        return functions
    
    def _extract_components_pattern(self, content: str) -> List[JSElement]:
        """Extract React components using patterns"""
        components = []
        
        # Class components
        for match in self.REACT_PATTERNS['component_class'].finditer(content):
            comp_name = match.group(1)
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            components.append(JSElement(
                element_type=JSElementType.COMPONENT,
                name=comp_name,
                content='',
                start_line=start_line,
                end_line=start_line,
                start_column=0,
                end_column=0,
                is_exported=False,
                is_default=False,
                is_async=False,
                is_generator=False,
                parameters=[],
                return_type=None,
                decorators=[],
                metadata={'type': 'class_component'}
            ))
        
        # Function components (capital letter)
        func_comp_pattern = re.compile(
            r'(?:export\s+)?(?:default\s+)?(?:const|function)\s+([A-Z]\w*)\s*[=:]\s*(?:\([^)]*\)|[^=]+)\s*=>'
        )
        
        for match in func_comp_pattern.finditer(content):
            comp_name = match.group(1)
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            components.append(JSElement(
                element_type=JSElementType.COMPONENT,
                name=comp_name,
                content='',
                start_line=start_line,
                end_line=start_line,
                start_column=0,
                end_column=0,
                is_exported='export' in match.group(0),
                is_default='default' in match.group(0),
                is_async=False,
                is_generator=False,
                parameters=[],
                return_type=None,
                decorators=[],
                metadata={'type': 'function_component'}
            ))
        
        return components
    
    def _extract_hooks_pattern(self, content: str) -> List[JSElement]:
        """Extract React hooks"""
        hooks = []
        
        hook_pattern = re.compile(r'(?:const|let)\s+(use[A-Z]\w*)\s*=')
        
        for match in hook_pattern.finditer(content):
            hook_name = match.group(1)
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            hooks.append(JSElement(
                element_type=JSElementType.HOOK,
                name=hook_name,
                content='',
                start_line=start_line,
                end_line=start_line,
                start_column=0,
                end_column=0,
                is_exported=False,
                is_default=False,
                is_async=False,
                is_generator=False,
                parameters=[],
                return_type=None,
                decorators=[],
                metadata={}
            ))
        
        return hooks
    
    def _extract_interfaces_pattern(self, content: str) -> List[JSElement]:
        """Extract TypeScript interfaces"""
        interfaces = []
        
        for match in self.TYPESCRIPT_PATTERNS['interface'].finditer(content):
            interface_name = match.group(1)
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            interfaces.append(JSElement(
                element_type=JSElementType.INTERFACE,
                name=interface_name,
                content='',
                start_line=start_line,
                end_line=start_line,
                start_column=0,
                end_column=0,
                is_exported='export' in match.group(0),
                is_default=False,
                is_async=False,
                is_generator=False,
                parameters=[],
                return_type=None,
                decorators=[],
                metadata={}
            ))
        
        return interfaces
    
    def _extract_types_pattern(self, content: str) -> List[JSElement]:
        """Extract TypeScript type aliases"""
        types = []
        
        for match in self.TYPESCRIPT_PATTERNS['type_alias'].finditer(content):
            type_name = match.group(1)
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            types.append(JSElement(
                element_type=JSElementType.TYPE_ALIAS,
                name=type_name,
                content='',
                start_line=start_line,
                end_line=start_line,
                start_column=0,
                end_column=0,
                is_exported='export' in match.group(0),
                is_default=False,
                is_async=False,
                is_generator=False,
                parameters=[],
                return_type=None,
                decorators=[],
                metadata={}
            ))
        
        return types
    
    def _extract_decorators_at_line(self, content: str, line_num: int) -> List[str]:
        """Extract decorators before a given line"""
        decorators = []
        lines = content.split('\n')
        
        if line_num > 1:
            # Check previous lines for decorators
            for i in range(line_num - 2, -1, -1):
                line = lines[i].strip()
                if line.startswith('@'):
                    match = self.TYPESCRIPT_PATTERNS['decorator'].match(line)
                    if match:
                        decorators.insert(0, match.group(1))
                elif line and not line.startswith('//'):
                    break
        
        return decorators
    
    def _extract_dependencies_pattern(self, content: str) -> Set[str]:
        """Extract all dependencies"""
        deps = set()
        
        # From imports
        import_pattern = re.compile(r'(?:import|require)\s*\([\'"`]([^\'"`]+)[\'"`]\)')
        for match in import_pattern.finditer(content):
            deps.add(match.group(1))
        
        # From ES6 imports
        es6_import = re.compile(r'from\s+[\'"`]([^\'"`]+)[\'"`]')
        for match in es6_import.finditer(content):
            deps.add(match.group(1))
        
        return deps
    
    def _detect_framework_pattern(self, content: str) -> Optional[str]:
        """Detect JavaScript framework"""
        # React
        if ('import React' in content or 'from \'react\'' in content or 
            'from "react"' in content or any(p.search(content) for p in self.REACT_PATTERNS.values())):
            return 'react'
        
        # Vue
        if ('from \'vue\'' in content or 'from "vue"' in content or
            any(p.search(content) for p in self.VUE_PATTERNS.values())):
            return 'vue'
        
        # Angular
        if ('from \'@angular' in content or 'from "@angular' in content or
            any(p.search(content) for p in self.ANGULAR_PATTERNS.values())):
            return 'angular'
        
        # Svelte
        if '<script' in content and '<style' in content and ('$:' in content or 'export let' in content):
            return 'svelte'
        
        # Express/Node
        if 'express' in content or any(p.search(content) for p in self.NODE_PATTERNS.values()):
            return 'express'
        
        return None
    
    def _detect_framework(self, module: JSModule, content: str) -> Optional[str]:
        """Detect framework from module structure"""
        # Check dependencies
        react_deps = {'react', 'react-dom', 'react-native'}
        vue_deps = {'vue', '@vue/composition-api'}
        angular_deps = {'@angular/core', '@angular/common'}
        
        for dep in module.dependencies:
            if any(r in dep for r in react_deps):
                return 'react'
            elif any(v in dep for v in vue_deps):
                return 'vue'
            elif any(a in dep for a in angular_deps):
                return 'angular'
        
        return self._detect_framework_pattern(content)
    
    def _find_block_end(self, lines: List[str], start: int) -> int:
        """Find the end of a code block"""
        brace_count = 0
        in_string = False
        string_char = None
        
        for i in range(start, len(lines)):
            line = lines[i]
            
            for char in line:
                # Handle strings
                if char in ['"', "'", '`'] and not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char and in_string:
                    in_string = False
                    string_char = None
                
                # Count braces
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and i > start:
                            return i
        
        return len(lines) - 1
    
    def _calculate_complexity(self, module: JSModule) -> Dict[str, Any]:
        """Calculate code complexity metrics"""
        return {
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(module),
            'class_count': len(module.classes),
            'function_count': len(module.functions),
            'component_count': len(module.components),
            'hook_count': len(module.hooks),
            'import_count': len(module.imports),
            'export_count': len(module.exports),
            'has_typescript': module.has_typescript,
            'has_jsx': module.has_jsx,
            'framework': module.framework
        }
    
    def _calculate_cyclomatic_complexity(self, module: JSModule) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        # Add complexity for each function/method
        all_functions = module.functions + module.hooks + [m for c in module.classes for m in c.children if m.element_type == JSElementType.METHOD]
        
        for func in all_functions:
            complexity += 1  # Each function adds 1
            
            # Count decision points in function content
            if func.content:
                complexity += func.content.count(' if ')
                complexity += func.content.count(' else ')
                complexity += func.content.count(' for ')
                complexity += func.content.count(' while ')
                complexity += func.content.count(' case ')
                complexity += func.content.count(' catch ')
                complexity += func.content.count(' ? ')  # Ternary operator
                complexity += func.content.count(' && ')
                complexity += func.content.count(' || ')
        
        return complexity
    
    def _calculate_metrics(self, module: JSModule, content: str) -> Dict[str, Any]:
        """Calculate various metrics"""
        lines = content.split('\n')
        
        return {
            'total_lines': len(lines),
            'code_lines': sum(1 for line in lines if line.strip() and not line.strip().startswith('//')),
            'comment_lines': sum(1 for line in lines if line.strip().startswith('//')),
            'blank_lines': sum(1 for line in lines if not line.strip()),
            'max_line_length': max((len(line) for line in lines), default=0),
            'avg_line_length': sum(len(line) for line in lines) / max(len(lines), 1),
            'dependency_count': len(module.dependencies),
            'uses_async': any(f.is_async for f in module.functions),
            'uses_generators': any(f.is_generator for f in module.functions),
            'uses_decorators': bool(module.has_typescript and '@' in content),
            'module_type': module.module_type
        }

class JavaScriptChunker(BaseChunker):
    """Chunker specialized for JavaScript/TypeScript files"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, ChunkerConfig(max_tokens=max_tokens))
        self.analyzer = JavaScriptAnalyzer()
        
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from JavaScript/TypeScript file
        
        Args:
            content: Source code
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Analyze file structure
            analysis = self.analyzer.analyze_javascript_file(content, file_context.path)
            
            chunks = []
            module = analysis['module']
            
            # Create import/header chunk
            header_chunk = self._create_header_chunk(content, module, file_context)
            if header_chunk:
                chunks.append(header_chunk)
            
            # Create chunks for components (highest priority for React apps)
            for component in module.components:
                comp_chunks = self._create_component_chunks(component, module, content, file_context)
                chunks.extend(comp_chunks)
            
            # Create chunks for classes
            for class_elem in module.classes:
                class_chunks = self._create_class_chunks(class_elem, module, content, file_context)
                chunks.extend(class_chunks)
            
            # Create chunks for hooks
            for hook in module.hooks:
                hook_chunk = self._create_hook_chunk(hook, module, content, file_context)
                if hook_chunk:
                    chunks.append(hook_chunk)
            
            # Create chunks for functions
            for function in module.functions:
                func_chunk = self._create_function_chunk(function, module, content, file_context)
                if func_chunk:
                    chunks.append(func_chunk)
            
            # Create chunks for TypeScript interfaces and types
            if module.has_typescript:
                for interface in module.interfaces:
                    interface_chunk = self._create_interface_chunk(interface, module, content, file_context)
                    if interface_chunk:
                        chunks.append(interface_chunk)
                
                for type_alias in module.types:
                    type_chunk = self._create_type_chunk(type_alias, module, content, file_context)
                    if type_chunk:
                        chunks.append(type_chunk)
            
            # If no chunks created, fall back to line-based chunking
            if not chunks:
                chunks = self._fallback_chunking(content, file_context)
            
            logger.info(f"Created {len(chunks)} chunks for {file_context.path}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking JavaScript file {file_context.path}: {e}")
            return self._fallback_chunking(content, file_context)
    
    def _create_header_chunk(self, content: str, module: JSModule,
                           file_context: FileContext) -> Optional[Chunk]:
        """Create header chunk with imports and module setup"""
        lines = content.split('\n')
        header_lines = []
        
        # Find the end of imports section
        import_end = 0
        for i, line in enumerate(lines):
            if re.match(r'^\s*(import|require|export|from)\s+', line):
                import_end = i + 1
            elif line.strip() and not line.strip().startswith('//'):
                # First non-import, non-comment line
                break
        
        if import_end == 0:
            return None
        
        header_lines = lines[:import_end]
        header_content = '\n'.join(header_lines)
        
        if not header_content.strip():
            return None
        
        return self.create_chunk(
            content=header_content,
            chunk_type='javascript_imports',
            metadata={
                'module_type': module.module_type,
                'import_count': len(module.imports),
                'dependency_count': len(module.dependencies),
                'framework': module.framework,
                'has_typescript': module.has_typescript
            },
            file_path=str(file_context.path)
        )
    
    def _create_component_chunks(self, component: JSElement, module: JSModule,
                                content: str, file_context: FileContext) -> List[Chunk]:
        """Create chunks for React/Vue components"""
        chunks = []
        lines = content.split('\n')
        
        # Get component content
        if component.end_line > 0:
            comp_lines = lines[component.start_line - 1:component.end_line]
        else:
            comp_lines = self._extract_element_lines(lines, component.start_line - 1)
        
        comp_content = '\n'.join(comp_lines)
        
        # Add context (imports for hooks/utilities used)
        context_lines = [
            f"// Component: {component.name}",
            f"// Framework: {module.framework or 'Unknown'}",
            ""
        ]
        
        full_content = '\n'.join(context_lines) + comp_content
        
        if self.count_tokens(full_content) <= self.max_tokens:
            chunks.append(self.create_chunk(
                content=full_content,
                chunk_type='javascript_component',
                metadata={
                    'component_name': component.name,
                    'framework': module.framework,
                    'is_exported': component.is_exported,
                    'is_default': component.is_default,
                    'component_type': component.metadata.get('type', 'unknown'),
                    'has_hooks': any(h in comp_content for h in ['useState', 'useEffect', 'useContext'])
                },
                file_path=str(file_context.path),
                start_line=component.start_line,
                end_line=component.end_line
            ))
        else:
            # Split large component
            chunks.extend(self._split_large_component(comp_content, component, module, file_context))
        
        return chunks
    
    def _create_class_chunks(self, class_elem: JSElement, module: JSModule,
                           content: str, file_context: FileContext) -> List[Chunk]:
        """Create chunks for classes"""
        chunks = []
        lines = content.split('\n')
        
        # Get class content
        if class_elem.end_line > 0:
            class_lines = lines[class_elem.start_line - 1:class_elem.end_line]
        else:
            class_lines = self._extract_element_lines(lines, class_elem.start_line - 1)
        
        class_content = '\n'.join(class_lines)
        
        # Check if it fits in one chunk
        if self.count_tokens(class_content) <= self.max_tokens:
            chunks.append(self.create_chunk(
                content=class_content,
                chunk_type='javascript_class',
                metadata={
                    'class_name': class_elem.name,
                    'is_exported': class_elem.is_exported,
                    'extends': class_elem.metadata.get('extends'),
                    'method_count': len(class_elem.children),
                    'has_constructor': any(m.element_type == JSElementType.CONSTRUCTOR for m in class_elem.children)
                },
                file_path=str(file_context.path),
                start_line=class_elem.start_line,
                end_line=class_elem.end_line
            ))
        else:
            # Split into methods
            chunks.extend(self._split_class_into_methods(class_elem, class_content, file_context))
        
        return chunks
    
    def _create_hook_chunk(self, hook: JSElement, module: JSModule,
                          content: str, file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for React hook"""
        lines = content.split('\n')
        
        if hook.end_line > 0:
            hook_lines = lines[hook.start_line - 1:hook.end_line]
        else:
            hook_lines = self._extract_element_lines(lines, hook.start_line - 1)
        
        hook_content = '\n'.join(hook_lines)
        
        # Add context
        context = [
            f"// Custom Hook: {hook.name}",
            "// Dependencies for context:",
            "import { useState, useEffect, useCallback, useMemo } from 'react';",
            ""
        ]
        
        full_content = '\n'.join(context) + hook_content
        
        return self.create_chunk(
            content=full_content,
            chunk_type='javascript_hook',
            metadata={
                'hook_name': hook.name,
                'is_exported': hook.is_exported,
                'uses_state': 'useState' in hook_content,
                'uses_effect': 'useEffect' in hook_content
            },
            file_path=str(file_context.path),
            start_line=hook.start_line,
            end_line=hook.end_line
        )
    
    def _create_function_chunk(self, function: JSElement, module: JSModule,
                              content: str, file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for function"""
        lines = content.split('\n')
        
        if function.end_line > 0:
            func_lines = lines[function.start_line - 1:function.end_line]
        else:
            func_lines = self._extract_element_lines(lines, function.start_line - 1)
        
        func_content = '\n'.join(func_lines)
        
        return self.create_chunk(
            content=func_content,
            chunk_type='javascript_function',
            metadata={
                'function_name': function.name,
                'is_async': function.is_async,
                'is_generator': function.is_generator,
                'is_arrow': function.element_type == JSElementType.ARROW_FUNCTION,
                'is_exported': function.is_exported,
                'is_default': function.is_default,
                'parameter_count': len(function.parameters)
            },
            file_path=str(file_context.path),
            start_line=function.start_line,
            end_line=function.end_line
        )
    
    def _create_interface_chunk(self, interface: JSElement, module: JSModule,
                               content: str, file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for TypeScript interface"""
        lines = content.split('\n')
        
        if interface.end_line > 0:
            interface_lines = lines[interface.start_line - 1:interface.end_line]
        else:
            interface_lines = self._extract_element_lines(lines, interface.start_line - 1)
        
        interface_content = '\n'.join(interface_lines)
        
        return self.create_chunk(
            content=interface_content,
            chunk_type='typescript_interface',
            metadata={
                'interface_name': interface.name,
                'is_exported': interface.is_exported
            },
            file_path=str(file_context.path),
            start_line=interface.start_line,
            end_line=interface.end_line
        )
    
    def _create_type_chunk(self, type_alias: JSElement, module: JSModule,
                         content: str, file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for TypeScript type alias"""
        lines = content.split('\n')
        
        if type_alias.end_line > 0:
            type_lines = lines[type_alias.start_line - 1:type_alias.end_line]
        else:
            # Type aliases are usually single line
            type_lines = [lines[type_alias.start_line - 1]]
        
        type_content = '\n'.join(type_lines)
        
        return self.create_chunk(
            content=type_content,
            chunk_type='typescript_type',
            metadata={
                'type_name': type_alias.name,
                'is_exported': type_alias.is_exported
            },
            file_path=str(file_context.path),
            start_line=type_alias.start_line,
            end_line=type_alias.end_line or type_alias.start_line
        )
    
    def _extract_element_lines(self, lines: List[str], start: int) -> List[str]:
        """Extract lines for an element when end line is unknown"""
        element_lines = []
        brace_count = 0
        in_element = False
        
        for i in range(start, len(lines)):
            line = lines[i]
            element_lines.append(line)
            
            # Track braces
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_element = True
                elif char == '}':
                    brace_count -= 1
            
            # Element ends when braces balance
            if in_element and brace_count == 0:
                break
        
        return element_lines
    
    def _split_large_component(self, content: str, component: JSElement,
                              module: JSModule, file_context: FileContext) -> List[Chunk]:
        """Split large component into smaller chunks"""
        chunks = []
        lines = content.split('\n')
        
        # Keep component declaration
        declaration_lines = []
        for i, line in enumerate(lines):
            declaration_lines.append(line)
            if '{' in line:
                break
        
        declaration = '\n'.join(declaration_lines)
        
        # Find logical sections (render method, lifecycle methods, etc.)
        sections = self._find_component_sections(lines[len(declaration_lines):])
        
        for section in sections:
            section_content = declaration + '\n' + section['content']
            
            if self.count_tokens(section_content) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=section_content,
                    chunk_type='javascript_component_part',
                    metadata={
                        'component_name': component.name,
                        'section': section['name'],
                        'framework': module.framework
                    },
                    file_path=str(file_context.path)
                ))
        
        return chunks
    
    def _find_component_sections(self, lines: List[str]) -> List[Dict[str, str]]:
        """Find logical sections in a component"""
        sections = []
        current_section = {'name': 'main', 'content': ''}
        
        for line in lines:
            # Detect section markers
            if 'render(' in line or 'return (' in line or 'return <' in line:
                if current_section['content']:
                    sections.append(current_section)
                current_section = {'name': 'render', 'content': line}
            elif 'componentDidMount' in line:
                if current_section['content']:
                    sections.append(current_section)
                current_section = {'name': 'lifecycle', 'content': line}
            elif 'useEffect' in line:
                if current_section['content']:
                    sections.append(current_section)
                current_section = {'name': 'effects', 'content': line}
            else:
                current_section['content'] += '\n' + line
        
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def _split_class_into_methods(self, class_elem: JSElement, content: str,
                                file_context: FileContext) -> List[Chunk]:
        """Split class into method chunks"""
        chunks = []
        
        # Class declaration chunk
        class_decl = f"class {class_elem.name}"
        if class_elem.metadata.get('extends'):
            class_decl += f" extends {class_elem.metadata['extends']}"
        class_decl += " {"
        
        # Add each method as a chunk
        for method in class_elem.children:
            method_context = [
                f"// Class: {class_elem.name}",
                class_decl,
                ""
            ]
            
            method_content = '\n'.join(method_context) + method.content + "\n}"
            
            chunks.append(self.create_chunk(
                content=method_content,
                chunk_type='javascript_method',
                metadata={
                    'class_name': class_elem.name,
                    'method_name': method.name,
                    'method_type': method.element_type.value,
                    'is_static': method.metadata.get('static', False),
                    'is_async': method.is_async
                },
                file_path=str(file_context.path),
                start_line=method.start_line,
                end_line=method.end_line
            ))
        
        return chunks
    
    def _fallback_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback to line-based chunking"""
        logger.warning(f"Using fallback chunking for {file_context.path}")
        
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
                    chunk_type='javascript_fallback',
                    metadata={
                        'is_fallback': True,
                        'line_count': len(current_chunk)
                    },
                    file_path=str(file_context.path),
                    start_line=chunk_start,
                    end_line=i - 1
                ))
                current_chunk = []
                current_tokens = 0
                chunk_start = i
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add remaining
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='javascript_fallback',
                metadata={
                    'is_fallback': True,
                    'line_count': len(current_chunk)
                },
                file_path=str(file_context.path),
                start_line=chunk_start,
                end_line=len(lines) - 1
            ))
        
        return chunks