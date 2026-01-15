"""
Python-specific chunker for intelligent semantic chunking
Handles classes, functions, async code, decorators, type hints, and modern Python patterns
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from enum import Enum
import tokenize
import io

from src.core.base_chunker import BaseChunker, Chunk, ChunkerConfig
from src.core.file_context import FileContext
from config.settings import settings

logger = logging.getLogger(__name__)

class PythonElementType(Enum):
    """Types of Python elements"""
    MODULE = "module"
    IMPORT = "import"
    CLASS = "class"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    METHOD = "method"
    ASYNC_METHOD = "async_method"
    PROPERTY = "property"
    STATIC_METHOD = "static_method"
    CLASS_METHOD = "class_method"
    DECORATOR = "decorator"
    CONSTRUCTOR = "constructor"
    GENERATOR = "generator"
    ASYNC_GENERATOR = "async_generator"
    LAMBDA = "lambda"
    COMPREHENSION = "comprehension"
    GLOBAL_VAR = "global_var"
    CONSTANT = "constant"
    TYPE_ALIAS = "type_alias"
    DATACLASS = "dataclass"
    ENUM_CLASS = "enum_class"
    PROTOCOL = "protocol"
    TYPED_DICT = "typed_dict"
    NAMED_TUPLE = "named_tuple"
    CONTEXT_MANAGER = "context_manager"

@dataclass
class PythonElement:
    """Represents a Python code element"""
    element_type: PythonElementType
    name: str
    content: str
    start_line: int
    end_line: int
    indent_level: int
    decorators: List[str]
    docstring: Optional[str]
    type_hints: Dict[str, str]
    is_async: bool
    is_generator: bool
    is_abstract: bool
    parent: Optional['PythonElement'] = None
    children: List['PythonElement'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PythonModule:
    """Represents a Python module structure"""
    module_name: str
    docstring: Optional[str]
    imports: List[Dict[str, Any]]
    from_imports: List[Dict[str, Any]]
    classes: List[PythonElement]
    functions: List[PythonElement]
    global_vars: List[PythonElement]
    constants: List[PythonElement]
    type_aliases: List[Dict[str, Any]]
    __all__: Optional[List[str]]
    is_main: bool
    is_package: bool
    encoding: str
    python_version: Optional[str]
    dependencies: Set[str]
    frameworks: Set[str]

class PythonASTAnalyzer(ast.NodeVisitor):
    """Analyzes Python AST for structure extraction"""
    
    # Common decorators to recognize
    COMMON_DECORATORS = {
        # Built-in
        'property', 'staticmethod', 'classmethod', 'abstractmethod',
        'cached_property', 'final', 'override', 'deprecated',
        
        # Dataclasses
        'dataclass', 'dataclasses.dataclass', 'field',
        
        # Typing
        'overload', 'no_type_check', 'type_check_only',
        
        # Pytest
        'pytest.fixture', 'pytest.mark.parametrize', 'pytest.mark.skip',
        'pytest.mark.skipif', 'pytest.mark.xfail',
        
        # Django
        'login_required', 'permission_required', 'require_http_methods',
        'csrf_exempt', 'cache_page', 'transaction.atomic',
        
        # Flask
        'app.route', 'login_required', 'cache.cached', 'cross_origin',
        
        # FastAPI
        'app.get', 'app.post', 'app.put', 'app.delete', 'app.patch',
        'Depends', 'Security', 'File', 'Form',
        
        # Celery
        'task', 'shared_task', 'periodic_task',
        
        # Click
        'click.command', 'click.option', 'click.argument', 'click.group',
        
        # Functools
        'lru_cache', 'cache', 'wraps', 'singledispatch',
        
        # Asyncio
        'asyncio.coroutine', 'async_generator',
        
        # Pydantic
        'validator', 'root_validator', 'field_validator',
    }
    
    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        'django': ['django', 'models.Model', 'views', 'urls', 'admin.site'],
        'flask': ['Flask', 'flask', 'Blueprint', 'render_template', 'request'],
        'fastapi': ['FastAPI', 'APIRouter', 'BaseModel', 'Field', 'Query'],
        'pytest': ['pytest', 'conftest', 'fixture', 'parametrize'],
        'numpy': ['numpy', 'np.', 'array', 'ndarray'],
        'pandas': ['pandas', 'pd.', 'DataFrame', 'Series'],
        'tensorflow': ['tensorflow', 'tf.', 'keras'],
        'pytorch': ['torch', 'nn.Module', 'tensor'],
        'scikit-learn': ['sklearn', 'fit', 'predict', 'transform'],
        'sqlalchemy': ['SQLAlchemy', 'Base', 'Column', 'Integer', 'String'],
        'asyncio': ['asyncio', 'async def', 'await', 'gather'],
        'celery': ['Celery', 'task', 'shared_task'],
        'requests': ['requests.get', 'requests.post'],
        'boto3': ['boto3', 'client', 'resource'],
        'pydantic': ['BaseModel', 'Field', 'validator'],
    }
    
    def __init__(self):
        self.module = None
        self.current_class = None
        self.current_function = None
        self.elements = []
        self.imports = []
        self.from_imports = []
        self.global_vars = []
        self.constants = []
        self.type_aliases = []
        
    def analyze_module(self, source_code: str, file_path: Path) -> PythonModule:
        """
        Analyze Python module structure
        
        Args:
            source_code: Python source code
            file_path: Path to Python file
            
        Returns:
            PythonModule object with analyzed structure
        """
        try:
            # Parse AST
            tree = ast.parse(source_code)
            
            # Initialize module
            self.module = PythonModule(
                module_name=file_path.stem,
                docstring=ast.get_docstring(tree),
                imports=[],
                from_imports=[],
                classes=[],
                functions=[],
                global_vars=[],
                constants=[],
                type_aliases=[],
                __all__=None,
                is_main='if __name__ ==' in source_code,
                is_package=file_path.name == '__init__.py',
                encoding=self._detect_encoding(source_code),
                python_version=self._detect_python_version(source_code),
                dependencies=set(),
                frameworks=self._detect_frameworks(source_code)
            )
            
            # Visit AST nodes
            self.visit(tree)
            
            # Post-processing
            self._resolve_relationships()
            self._detect_patterns()
            
            return self.module
            
        except SyntaxError as e:
            logger.error(f"Python syntax error in {file_path}: {e}")
            return self._create_fallback_module(source_code, file_path)
    
    def visit_Import(self, node: ast.Import):
        """Visit import statement"""
        for alias in node.names:
            import_info = {
                'module': alias.name,
                'alias': alias.asname,
                'line': node.lineno,
                'col': node.col_offset
            }
            self.module.imports.append(import_info)
            self.module.dependencies.add(alias.name.split('.')[0])
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from...import statement"""
        module_name = node.module or ''
        level = node.level  # Relative import level
        
        for alias in node.names:
            from_import_info = {
                'module': module_name,
                'name': alias.name,
                'alias': alias.asname,
                'level': level,
                'line': node.lineno,
                'col': node.col_offset
            }
            self.module.from_imports.append(from_import_info)
            
            if module_name:
                self.module.dependencies.add(module_name.split('.')[0])
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition"""
        # Extract decorators
        decorators = self._extract_decorators(node)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Determine class type
        element_type = self._determine_class_type(node, decorators)
        
        # Extract base classes
        bases = [self._get_name(base) for base in node.bases]
        
        # Create class element
        class_element = PythonElement(
            element_type=element_type,
            name=node.name,
            content='',  # Will be filled later
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            indent_level=node.col_offset // 4,
            decorators=decorators,
            docstring=docstring,
            type_hints={},
            is_async=False,
            is_generator=False,
            is_abstract='ABC' in bases or 'abc.ABC' in bases,
            metadata={
                'bases': bases,
                'keywords': {kw.arg: self._get_name(kw.value) for kw in node.keywords if kw.arg},
                'metaclass': next((kw.value for kw in node.keywords if kw.arg == 'metaclass'), None)
            }
        )
        
        # Store current class context
        parent_class = self.current_class
        self.current_class = class_element
        
        # Visit class body
        for item in node.body:
            if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                method = self._extract_method(item, class_element)
                class_element.children.append(method)
            elif isinstance(item, ast.Assign):
                # Class variables
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_element.metadata.setdefault('class_vars', []).append(target.id)
            elif isinstance(item, ast.AnnAssign):
                # Type-annotated class variables
                if isinstance(item.target, ast.Name):
                    var_name = item.target.id
                    type_hint = self._get_type_hint(item.annotation)
                    class_element.type_hints[var_name] = type_hint
        
        # Restore parent class context
        self.current_class = parent_class
        
        # Add to module
        self.module.classes.append(class_element)
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition"""
        if self.current_class is None:  # Module-level function
            function_element = self._extract_function(node)
            self.module.functions.append(function_element)
        
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition"""
        if self.current_class is None:  # Module-level function
            function_element = self._extract_function(node)
            function_element.is_async = True
            function_element.element_type = PythonElementType.ASYNC_FUNCTION
            self.module.functions.append(function_element)
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        """Visit assignment statement"""
        if self.current_class is None and self.current_function is None:
            # Module-level assignment
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    
                    # Check if it's a constant (UPPER_CASE)
                    if var_name.isupper():
                        const_element = PythonElement(
                            element_type=PythonElementType.CONSTANT,
                            name=var_name,
                            content='',
                            start_line=node.lineno,
                            end_line=node.lineno,
                            indent_level=node.col_offset // 4,
                            decorators=[],
                            docstring=None,
                            type_hints={},
                            is_async=False,
                            is_generator=False,
                            is_abstract=False,
                            metadata={'value': self._get_value_repr(node.value)}
                        )
                        self.module.constants.append(const_element)
                    else:
                        # Regular global variable
                        var_element = PythonElement(
                            element_type=PythonElementType.GLOBAL_VAR,
                            name=var_name,
                            content='',
                            start_line=node.lineno,
                            end_line=node.lineno,
                            indent_level=node.col_offset // 4,
                            decorators=[],
                            docstring=None,
                            type_hints={},
                            is_async=False,
                            is_generator=False,
                            is_abstract=False,
                            metadata={'value': self._get_value_repr(node.value)}
                        )
                        self.module.global_vars.append(var_element)
                    
                    # Check for __all__
                    if var_name == '__all__':
                        if isinstance(node.value, ast.List):
                            self.module.__all__ = [
                                elt.s for elt in node.value.elts 
                                if isinstance(elt, ast.Str)
                            ]
        
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Visit annotated assignment"""
        if self.current_class is None and self.current_function is None:
            # Module-level type-annotated variable
            if isinstance(node.target, ast.Name):
                var_name = node.target.id
                type_hint = self._get_type_hint(node.annotation)
                
                # Check for type alias
                if self._is_type_alias(node):
                    self.module.type_aliases.append({
                        'name': var_name,
                        'type': type_hint,
                        'line': node.lineno
                    })
                else:
                    var_element = PythonElement(
                        element_type=PythonElementType.GLOBAL_VAR,
                        name=var_name,
                        content='',
                        start_line=node.lineno,
                        end_line=node.lineno,
                        indent_level=node.col_offset // 4,
                        decorators=[],
                        docstring=None,
                        type_hints={var_name: type_hint},
                        is_async=False,
                        is_generator=False,
                        is_abstract=False,
                        metadata={'has_type_hint': True}
                    )
                    self.module.global_vars.append(var_element)
        
        self.generic_visit(node)
    
    def _extract_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> PythonElement:
        """Extract function information"""
        # Extract decorators
        decorators = self._extract_decorators(node)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract type hints
        type_hints = self._extract_function_type_hints(node)
        
        # Determine if generator
        is_generator = self._is_generator_function(node)
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        # Determine element type
        if is_async and is_generator:
            element_type = PythonElementType.ASYNC_GENERATOR
        elif is_generator:
            element_type = PythonElementType.GENERATOR
        elif is_async:
            element_type = PythonElementType.ASYNC_FUNCTION
        else:
            element_type = PythonElementType.FUNCTION
        
        return PythonElement(
            element_type=element_type,
            name=node.name,
            content='',
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            indent_level=node.col_offset // 4,
            decorators=decorators,
            docstring=docstring,
            type_hints=type_hints,
            is_async=is_async,
            is_generator=is_generator,
            is_abstract=False,
            metadata={
                'args': self._extract_arguments(node.args),
                'returns': type_hints.get('return', None),
                'complexity': self._calculate_complexity(node)
            }
        )
    
    def _extract_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                       class_element: PythonElement) -> PythonElement:
        """Extract method from class"""
        method = self._extract_function(node)
        
        # Determine method type based on decorators and name
        if 'property' in method.decorators:
            method.element_type = PythonElementType.PROPERTY
        elif 'staticmethod' in method.decorators:
            method.element_type = PythonElementType.STATIC_METHOD
        elif 'classmethod' in method.decorators:
            method.element_type = PythonElementType.CLASS_METHOD
        elif node.name.startswith('__') and node.name.endswith('__'):
            method.metadata['is_magic'] = True
            if node.name == '__init__':
                method.element_type = PythonElementType.CONSTRUCTOR
            elif node.name in ['__enter__', '__exit__', '__aenter__', '__aexit__']:
                method.element_type = PythonElementType.CONTEXT_MANAGER
        elif isinstance(node, ast.AsyncFunctionDef):
            method.element_type = PythonElementType.ASYNC_METHOD
        else:
            method.element_type = PythonElementType.METHOD
        
        method.parent = class_element
        return method
    
    def _extract_decorators(self, node: Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Extract decorator names"""
        decorators = []
        
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(self._get_name(decorator))
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
                elif isinstance(decorator.func, ast.Attribute):
                    decorators.append(self._get_name(decorator.func))
        
        return decorators
    
    def _extract_function_type_hints(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, str]:
        """Extract function type hints"""
        type_hints = {}
        
        # Arguments
        for arg in node.args.args:
            if arg.annotation:
                type_hints[arg.arg] = self._get_type_hint(arg.annotation)
        
        # Keyword-only arguments
        for arg in node.args.kwonlyargs:
            if arg.annotation:
                type_hints[arg.arg] = self._get_type_hint(arg.annotation)
        
        # Return type
        if node.returns:
            type_hints['return'] = self._get_type_hint(node.returns)
        
        return type_hints
    
    def _extract_arguments(self, args: ast.arguments) -> Dict[str, Any]:
        """Extract function arguments"""
        return {
            'args': [arg.arg for arg in args.args],
            'defaults': [self._get_value_repr(d) for d in args.defaults],
            'kwonlyargs': [arg.arg for arg in args.kwonlyargs],
            'kw_defaults': [self._get_value_repr(d) if d else None for d in args.kw_defaults],
            'vararg': args.vararg.arg if args.vararg else None,
            'kwarg': args.kwarg.arg if args.kwarg else None
        }
    
    def _determine_class_type(self, node: ast.ClassDef, decorators: List[str]) -> PythonElementType:
        """Determine the type of class"""
        if 'dataclass' in decorators or 'dataclasses.dataclass' in decorators:
            return PythonElementType.DATACLASS
        
        # Check base classes
        bases = [self._get_name(base) for base in node.bases]
        
        if 'Enum' in bases or 'enum.Enum' in bases:
            return PythonElementType.ENUM_CLASS
        elif 'Protocol' in bases or 'typing.Protocol' in bases:
            return PythonElementType.PROTOCOL
        elif 'TypedDict' in bases or 'typing.TypedDict' in bases:
            return PythonElementType.TYPED_DICT
        elif 'NamedTuple' in bases or 'typing.NamedTuple' in bases:
            return PythonElementType.NAMED_TUPLE
        
        return PythonElementType.CLASS
    
    def _is_generator_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if function is a generator"""
        for child in ast.walk(node):
            if isinstance(child, ast.Yield) or isinstance(child, ast.YieldFrom):
                return True
        return False
    
    def _is_type_alias(self, node: ast.AnnAssign) -> bool:
        """Check if annotated assignment is a type alias"""
        # Simple heuristic: TypeAlias annotation or capitalized name
        if isinstance(node.annotation, ast.Name) and node.annotation.id == 'TypeAlias':
            return True
        
        if isinstance(node.target, ast.Name):
            name = node.target.id
            # Type aliases often start with capital letter
            if name[0].isupper() and '=' in ast.unparse(node):
                return True
        
        return False
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from various AST nodes"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, ast.Subscript):
            return self._get_name(node.value)
        elif hasattr(node, 'id'):
            return node.id
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else ''
    
    def _get_type_hint(self, node: ast.AST) -> str:
        """Get type hint as string"""
        if hasattr(ast, 'unparse'):
            return ast.unparse(node)
        else:
            # Fallback for older Python versions
            return self._get_name(node)
    
    def _get_value_repr(self, node: ast.AST) -> Any:
        """Get representation of a value node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return [self._get_value_repr(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._get_value_repr(elt) for elt in node.elts)
        elif isinstance(node, ast.Dict):
            return {
                self._get_value_repr(k): self._get_value_repr(v)
                for k, v in zip(node.keys, node.values)
            }
        else:
            return None
    
    def _calculate_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With) or isinstance(child, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _detect_encoding(self, source_code: str) -> str:
        """Detect source file encoding"""
        # Check for encoding declaration in first two lines
        lines = source_code.split('\n', 2)[:2]
        
        for line in lines:
            match = re.search(r'coding[=:]\s*([-\w.]+)', line)
            if match:
                return match.group(1)
        
        return 'utf-8'
    
    def _detect_python_version(self, source_code: str) -> Optional[str]:
        """Detect required Python version from code patterns"""
        # Check for version-specific syntax
        patterns = {
            '3.10': ['match ', 'case '],
            '3.9': ['dict[', 'list[', 'tuple['],
            '3.8': [':=', 'f"{', "f'{"],
            '3.7': ['__annotations__', 'dataclass'],
            '3.6': ['f"', "f'"],
            '3.5': ['async def', 'await '],
        }
        
        for version, indicators in patterns.items():
            if any(ind in source_code for ind in indicators):
                return version
        
        return None
    
    def _detect_frameworks(self, source_code: str) -> Set[str]:
        """Detect frameworks used in code"""
        frameworks = set()
        
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            if any(pattern in source_code for pattern in patterns):
                frameworks.add(framework)
        
        return frameworks
    
    def _resolve_relationships(self):
        """Resolve parent-child relationships between elements"""
        # This is handled during visiting, but could be enhanced here
        pass
    
    def _detect_patterns(self):
        """Detect common Python patterns"""
        # Could detect patterns like:
        # - Factory patterns
        # - Singleton patterns
        # - Context managers
        # - Decorators patterns
        pass
    
    def _create_fallback_module(self, source_code: str, file_path: Path) -> PythonModule:
        """Create fallback module when AST parsing fails"""
        return PythonModule(
            module_name=file_path.stem,
            docstring=None,
            imports=[],
            from_imports=[],
            classes=[],
            functions=[],
            global_vars=[],
            constants=[],
            type_aliases=[],
            __all__=None,
            is_main='if __name__ ==' in source_code,
            is_package=file_path.name == '__init__.py',
            encoding='utf-8',
            python_version=None,
            dependencies=set(),
            frameworks=set()
        )

class PythonChunker(BaseChunker):
    """Chunker specialized for Python source files"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, ChunkerConfig(max_tokens=max_tokens))
        self.analyzer = PythonASTAnalyzer()
        
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from Python source file
        
        Args:
            content: Python source code
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Analyze Python structure
            module = self.analyzer.analyze_module(content, file_context.path)
            
            chunks = []
            
            # Create module header chunk (imports and module docstring)
            header_chunk = self._create_header_chunk(content, module, file_context)
            if header_chunk:
                chunks.append(header_chunk)
            
            # Create chunks for classes
            for class_elem in module.classes:
                class_chunks = self._create_class_chunks(class_elem, module, content, file_context)
                chunks.extend(class_chunks)
            
            # Create chunks for functions
            for function_elem in module.functions:
                func_chunk = self._create_function_chunk(function_elem, module, content, file_context)
                if func_chunk:
                    chunks.append(func_chunk)
            
            # Create chunk for global variables and constants
            if module.global_vars or module.constants:
                globals_chunk = self._create_globals_chunk(module, content, file_context)
                if globals_chunk:
                    chunks.append(globals_chunk)
            
            # Create chunk for type definitions
            if module.type_aliases:
                types_chunk = self._create_types_chunk(module, content, file_context)
                if types_chunk:
                    chunks.append(types_chunk)
            
            # If no chunks created, fall back to line-based chunking
            if not chunks:
                chunks = self._fallback_chunking(content, file_context)
            
            # Add metadata about the module to all chunks
            for chunk in chunks:
                chunk.metadata['annotations'] = chunk.metadata.get('annotations', {})
                chunk.metadata['annotations']['module_name'] = module.module_name
                chunk.metadata['annotations']['is_package'] = module.is_package
                chunk.metadata['annotations']['frameworks'] = list(module.frameworks)
            
            logger.info(f"Created {len(chunks)} chunks for Python file {file_context.path}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking Python file {file_context.path}: {e}")
            return self._fallback_chunking(content, file_context)
    
    def _create_header_chunk(self, content: str, module: PythonModule,
                           file_context: FileContext) -> Optional[Chunk]:
        """Create header chunk with imports and module docstring"""
        lines = content.split('\n')
        header_lines = []
        
        # Add shebang if present
        if lines and lines[0].startswith('#!'):
            header_lines.append(lines[0])
        
        # Add encoding declaration if present
        for line in lines[:2]:
            if 'coding' in line and line.startswith('#'):
                header_lines.append(line)
                break
        
        # Add module docstring
        if module.docstring:
            header_lines.append('"""')
            header_lines.extend(module.docstring.split('\n'))
            header_lines.append('"""')
            header_lines.append('')
        
        # Add __all__ if present
        if module.__all__:
            header_lines.append(f"__all__ = {module.__all__}")
            header_lines.append('')
        
        # Add imports
        import_lines = self._extract_import_lines(content)
        if import_lines:
            header_lines.extend(import_lines)
        
        if not header_lines:
            return None
        
        header_content = '\n'.join(header_lines)
        
        return self.create_chunk(
            content=header_content,
            chunk_type='python_header',
            metadata={
                'has_docstring': bool(module.docstring),
                'import_count': len(module.imports) + len(module.from_imports),
                'dependencies': list(module.dependencies),
                'has___all__': module.__all__ is not None,
                'python_version': module.python_version
            },
            file_path=str(file_context.path)
        )
    
    def _create_class_chunks(self, class_elem: PythonElement, module: PythonModule,
                           content: str, file_context: FileContext) -> List[Chunk]:
        """Create chunks for a Python class"""
        chunks = []
        lines = content.split('\n')
        
        # Extract class content
        class_lines = lines[class_elem.start_line - 1:class_elem.end_line]
        class_content = '\n'.join(class_lines)
        
        # Check if entire class fits in one chunk
        if self.count_tokens(class_content) <= self.max_tokens:
            chunks.append(self.create_chunk(
                content=class_content,
                chunk_type='python_class',
                metadata={
                    'class_name': class_elem.name,
                    'class_type': class_elem.element_type.value,
                    'is_abstract': class_elem.is_abstract,
                    'bases': class_elem.metadata.get('bases', []),
                    'decorators': class_elem.decorators,
                    'method_count': len(class_elem.children),
                    'has_docstring': class_elem.docstring is not None
                },
                file_path=str(file_context.path),
                start_line=class_elem.start_line,
                end_line=class_elem.end_line
            ))
        else:
            # Split class into smaller chunks
            chunks.extend(self._split_large_class(class_elem, class_content, file_context))
        
        return chunks
    
    def _split_large_class(self, class_elem: PythonElement, content: str,
                         file_context: FileContext) -> List[Chunk]:
        """Split a large class into method-level chunks"""
        chunks = []
        lines = content.split('\n')
        
        # Create class declaration chunk
        class_decl_lines = []
        for i, line in enumerate(lines):
            class_decl_lines.append(line)
            if line.strip() and not line.strip().startswith('@') and ':' in line:
                # Found class declaration line
                break
        
        # Add class docstring if present
        if class_elem.docstring:
            # Find and include docstring lines
            in_docstring = False
            for i in range(len(class_decl_lines), min(len(class_decl_lines) + 20, len(lines))):
                line = lines[i]
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                    class_decl_lines.append(line)
                    if not in_docstring:
                        break
                elif in_docstring:
                    class_decl_lines.append(line)
        
        class_declaration = '\n'.join(class_decl_lines)
        
        # Create chunk for class declaration with class variables
        class_vars_chunk = self.create_chunk(
            content=class_declaration,
            chunk_type='python_class_declaration',
            metadata={
                'class_name': class_elem.name,
                'class_type': class_elem.element_type.value,
                'bases': class_elem.metadata.get('bases', []),
                'has_docstring': class_elem.docstring is not None
            },
            file_path=str(file_context.path)
        )
        chunks.append(class_vars_chunk)
        
        # Create chunks for each method
        for method in class_elem.children:
            method_chunk = self._create_method_chunk(method, class_elem, content, file_context)
            if method_chunk:
                chunks.append(method_chunk)
        
        return chunks
    
    def _create_method_chunk(self, method: PythonElement, class_elem: PythonElement,
                           content: str, file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for a class method"""
        lines = content.split('\n')
        
        # Extract method content
        method_lines = lines[method.start_line - 1:method.end_line]
        method_content = '\n'.join(method_lines)
        
        # Add class context
        context_lines = [
            f"# Class: {class_elem.name}",
            f"# Method: {method.name}",
        ]
        
        if method.decorators:
            context_lines.append(f"# Decorators: {', '.join(method.decorators)}")
        
        context_lines.append("")
        
        full_content = '\n'.join(context_lines) + method_content
        
        return self.create_chunk(
            content=full_content,
            chunk_type='python_method',
            metadata={
                'class_name': class_elem.name,
                'method_name': method.name,
                'method_type': method.element_type.value,
                'is_async': method.is_async,
                'is_generator': method.is_generator,
                'decorators': method.decorators,
                'has_docstring': method.docstring is not None,
                'complexity': method.metadata.get('complexity', 1)
            },
            file_path=str(file_context.path),
            start_line=method.start_line,
            end_line=method.end_line
        )
    
    def _create_function_chunk(self, function: PythonElement, module: PythonModule,
                             content: str, file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for a module-level function"""
        lines = content.split('\n')
        
        # Include decorators if present
        start_line = function.start_line - 1
        if function.decorators:
            # Look back for decorator lines
            while start_line > 0 and lines[start_line - 1].strip().startswith('@'):
                start_line -= 1
        
        func_lines = lines[start_line:function.end_line]
        func_content = '\n'.join(func_lines)
        
        return self.create_chunk(
            content=func_content,
            chunk_type='python_function',
            metadata={
                'function_name': function.name,
                'function_type': function.element_type.value,
                'is_async': function.is_async,
                'is_generator': function.is_generator,
                'decorators': function.decorators,
                'has_docstring': function.docstring is not None,
                'has_type_hints': bool(function.type_hints),
                'complexity': function.metadata.get('complexity', 1),
                'parameters': function.metadata.get('args', {})
            },
            file_path=str(file_context.path),
            start_line=function.start_line,
            end_line=function.end_line
        )
    
    def _create_globals_chunk(self, module: PythonModule, content: str,
                            file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for global variables and constants"""
        lines = []
        
        lines.append("# Global Variables and Constants")
        lines.append("")
        
        # Add constants
        if module.constants:
            lines.append("# Constants")
            for const in module.constants:
                const_line = f"{const.name} = {const.metadata.get('value', '...')}"
                lines.append(const_line)
            lines.append("")
        
        # Add global variables
        if module.global_vars:
            lines.append("# Global Variables")
            for var in module.global_vars:
                if var.type_hints:
                    type_hint = list(var.type_hints.values())[0]
                    var_line = f"{var.name}: {type_hint}"
                else:
                    var_line = f"{var.name} = {var.metadata.get('value', '...')}"
                lines.append(var_line)
        
        if len(lines) <= 2:  # Only header
            return None
        
        return self.create_chunk(
            content='\n'.join(lines),
            chunk_type='python_globals',
            metadata={
                'constant_count': len(module.constants),
                'global_var_count': len(module.global_vars)
            },
            file_path=str(file_context.path)
        )
    
    def _create_types_chunk(self, module: PythonModule, content: str,
                          file_context: FileContext) -> Optional[Chunk]:
        """Create chunk for type definitions"""
        lines = []
        
        lines.append("# Type Definitions")
        lines.append("from typing import *")
        lines.append("")
        
        for type_alias in module.type_aliases:
            lines.append(f"{type_alias['name']} = {type_alias['type']}")
        
        return self.create_chunk(
            content='\n'.join(lines),
            chunk_type='python_types',
            metadata={
                'type_count': len(module.type_aliases)
            },
            file_path=str(file_context.path)
        )
    
    def _extract_import_lines(self, content: str) -> List[str]:
        """Extract import lines from content"""
        lines = content.split('\n')
        import_lines = []
        
        for line in lines:
            # Stop at first non-import, non-comment line
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            elif stripped.startswith('import ') or stripped.startswith('from '):
                import_lines.append(line)
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                # Skip module docstring
                continue
            elif import_lines:
                # We've seen imports and now hit non-import line
                break
        
        return import_lines
    
    def _fallback_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback to simple chunking when AST parsing fails"""
        logger.warning(f"Using fallback chunking for Python file {file_context.path}")
        
        chunks = []
        lines = content.split('\n')
        
        # Try to identify functions and classes using regex
        function_pattern = re.compile(r'^(async\s+)?def\s+(\w+)\s*\(')
        class_pattern = re.compile(r'^class\s+(\w+)')
        
        current_chunk = []
        current_tokens = 0
        current_indent = 0
        chunk_type = 'python_code'
        chunk_name = None
        chunk_start = 0
        
        for i, line in enumerate(lines):
            indent = len(line) - len(line.lstrip())
            
            # Check for new function or class
            func_match = function_pattern.match(line.strip())
            class_match = class_pattern.match(line.strip())
            
            if (func_match or class_match) and current_chunk:
                # Save current chunk
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type=chunk_type,
                    metadata={
                        'name': chunk_name,
                        'is_fallback': True
                    },
                    file_path=str(file_context.path),
                    start_line=chunk_start,
                    end_line=i - 1
                ))
                
                # Start new chunk
                current_chunk = []
                current_tokens = 0
                chunk_start = i
                
                if func_match:
                    chunk_type = 'python_function_fallback'
                    chunk_name = func_match.group(2)
                elif class_match:
                    chunk_type = 'python_class_fallback'
                    chunk_name = class_match.group(1)
            
            # Add line to current chunk
            line_tokens = self.count_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                # Save current chunk
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type=chunk_type,
                    metadata={
                        'name': chunk_name,
                        'is_fallback': True
                    },
                    file_path=str(file_context.path),
                    start_line=chunk_start,
                    end_line=i - 1
                ))
                
                current_chunk = [line]
                current_tokens = line_tokens
                chunk_start = i
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
            
            current_indent = indent
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type=chunk_type,
                metadata={
                    'name': chunk_name,
                    'is_fallback': True
                },
                file_path=str(file_context.path),
                start_line=chunk_start,
                end_line=len(lines) - 1
            ))
        
        return chunks