"""
PL/SQL-specific chunker for intelligent semantic chunking
Handles packages, procedures, functions, triggers, types, and complex Oracle database objects
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from enum import Enum

from src.core.base_chunker import BaseChunker, Chunk
from src.core.file_context import FileContext
from config.settings import settings

logger = logging.getLogger(__name__)

class PLSQLObjectType(Enum):
    """Types of PL/SQL objects"""
    PACKAGE_SPEC = "package_spec"
    PACKAGE_BODY = "package_body"
    PROCEDURE = "procedure"
    FUNCTION = "function"
    TRIGGER = "trigger"
    TYPE_SPEC = "type_spec"
    TYPE_BODY = "type_body"
    VIEW = "view"
    MATERIALIZED_VIEW = "materialized_view"
    ANONYMOUS_BLOCK = "anonymous_block"
    DECLARE_BLOCK = "declare_block"
    CURSOR = "cursor"
    EXCEPTION = "exception"
    RECORD = "record"
    TABLE_TYPE = "table_type"
    VARRAY = "varray"
    OBJECT_TYPE = "object_type"
    SEQUENCE = "sequence"
    SYNONYM = "synonym"
    INDEX = "index"
    TABLE = "table"
    CONSTRAINT = "constraint"

@dataclass
class PLSQLElement:
    """Represents a PL/SQL code element"""
    element_type: PLSQLObjectType
    name: str
    schema: Optional[str]
    content: str
    start_line: int
    end_line: int
    parent: Optional['PLSQLElement']
    children: List['PLSQLElement']
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    dependencies: Set[str]
    grants: List[str]
    pragmas: List[str]
    exceptions: List[str]
    is_autonomous: bool
    is_deterministic: bool
    is_pipelined: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PLSQLStructure:
    """Represents overall PL/SQL file structure"""
    objects: List[PLSQLElement]
    total_lines: int
    total_objects: int
    has_package_spec: bool
    has_package_body: bool
    has_anonymous_blocks: bool
    uses_dbms_packages: Set[str]
    uses_cursors: bool
    uses_dynamic_sql: bool
    uses_bulk_collect: bool
    uses_forall: bool
    uses_ref_cursors: bool
    uses_exceptions: bool
    dependencies: Set[str]
    compilation_errors: List[str]
    statistics: Dict[str, Any]

class PLSQLAnalyzer:
    """Analyzes PL/SQL code structure"""
    
    # PL/SQL keywords for syntax highlighting
    KEYWORDS = {
        'DECLARE', 'BEGIN', 'END', 'EXCEPTION', 'WHEN', 'THEN', 'RAISE',
        'IF', 'ELSIF', 'ELSE', 'CASE', 'LOOP', 'WHILE', 'FOR', 'EXIT',
        'CONTINUE', 'RETURN', 'GOTO', 'NULL', 'PRAGMA', 'AUTONOMOUS_TRANSACTION',
        'CREATE', 'REPLACE', 'OR', 'AS', 'IS', 'TYPE', 'SUBTYPE',
        'CURSOR', 'PROCEDURE', 'FUNCTION', 'PACKAGE', 'BODY', 'TRIGGER',
        'VIEW', 'SEQUENCE', 'SYNONYM', 'INDEX', 'TABLE', 'CONSTRAINT',
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'INTO', 'FROM',
        'WHERE', 'GROUP', 'BY', 'HAVING', 'ORDER', 'CONNECT', 'START',
        'WITH', 'UNION', 'INTERSECT', 'MINUS', 'EXISTS', 'NOT', 'IN',
        'BETWEEN', 'LIKE', 'AND', 'OR', 'ANY', 'ALL', 'SOME',
        'COMMIT', 'ROLLBACK', 'SAVEPOINT', 'LOCK', 'GRANT', 'REVOKE'
    }
    
    # Common DBMS packages
    DBMS_PACKAGES = {
        'DBMS_OUTPUT', 'DBMS_SQL', 'DBMS_LOB', 'DBMS_RANDOM', 'DBMS_UTILITY',
        'DBMS_SCHEDULER', 'DBMS_LOCK', 'DBMS_ALERT', 'DBMS_PIPE', 'DBMS_JOB',
        'DBMS_SESSION', 'DBMS_TRANSACTION', 'DBMS_METADATA', 'DBMS_STATS',
        'DBMS_CRYPTO', 'DBMS_REDEFINITION', 'DBMS_REPAIR', 'DBMS_SPACE',
        'UTL_FILE', 'UTL_HTTP', 'UTL_SMTP', 'UTL_TCP', 'UTL_COMPRESS',
        'UTL_ENCODE', 'UTL_RAW', 'UTL_MATCH', 'UTL_I18N'
    }
    
    # Regular expressions for PL/SQL parsing
    PATTERNS = {
        # Package specification
        'package_spec': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:EDITIONABLE\s+|NONEDITIONABLE\s+)?'
            r'PACKAGE\s+(?:BODY\s+)?(?:"?(\w+)"?\.)?"?(\w+)"?'
            r'(?:\s+AUTHID\s+(?:DEFINER|CURRENT_USER))?'
            r'\s+(?:AS|IS)',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # Package body
        'package_body': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:EDITIONABLE\s+|NONEDITIONABLE\s+)?'
            r'PACKAGE\s+BODY\s+(?:"?(\w+)"?\.)?"?(\w+)"?'
            r'\s+(?:AS|IS)',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # Procedure
        'procedure': re.compile(
            r'(?:CREATE\s+(?:OR\s+REPLACE\s+)?)?'
            r'PROCEDURE\s+(?:"?(\w+)"?\.)?"?(\w+)"?'
            r'\s*\([^)]*\)?\s*(?:AS|IS)',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # Function
        'function': re.compile(
            r'(?:CREATE\s+(?:OR\s+REPLACE\s+)?)?'
            r'FUNCTION\s+(?:"?(\w+)"?\.)?"?(\w+)"?'
            r'\s*\([^)]*\)?\s*RETURN\s+\w+\s+(?:AS|IS)',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # Trigger
        'trigger': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:EDITIONABLE\s+|NONEDITIONABLE\s+)?'
            r'TRIGGER\s+(?:"?(\w+)"?\.)?"?(\w+)"?'
            r'\s+(?:BEFORE|AFTER|INSTEAD\s+OF)',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # Type specification
        'type_spec': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:EDITIONABLE\s+|NONEDITIONABLE\s+)?'
            r'TYPE\s+(?:BODY\s+)?(?:"?(\w+)"?\.)?"?(\w+)"?'
            r'\s+(?:AS|IS|UNDER)',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # View
        'view': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:FORCE\s+)?(?:EDITIONABLE\s+|NONEDITIONABLE\s+)?'
            r'VIEW\s+(?:"?(\w+)"?\.)?"?(\w+)"?',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # Materialized view
        'mview': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?MATERIALIZED\s+VIEW\s+(?:"?(\w+)"?\.)?"?(\w+)"?',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # Anonymous block
        'anonymous_block': re.compile(
            r'(?:DECLARE\s+.*?)?\s*BEGIN\s+',
            re.IGNORECASE | re.MULTILINE | re.DOTALL
        ),
        
        # Cursor declaration
        'cursor': re.compile(
            r'CURSOR\s+(\w+)\s*(?:\([^)]*\))?\s+IS\s*',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # Type definitions
        'type_record': re.compile(
            r'TYPE\s+(\w+)\s+IS\s+RECORD\s*\(',
            re.IGNORECASE | re.MULTILINE
        ),
        
        'type_table': re.compile(
            r'TYPE\s+(\w+)\s+IS\s+TABLE\s+OF',
            re.IGNORECASE | re.MULTILINE
        ),
        
        'type_varray': re.compile(
            r'TYPE\s+(\w+)\s+IS\s+(?:VARRAY|VARYING\s+ARRAY)',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # Exception declaration
        'exception': re.compile(
            r'(\w+)\s+EXCEPTION;',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # Pragma directives
        'pragma': re.compile(
            r'PRAGMA\s+(\w+)',
            re.IGNORECASE | re.MULTILINE
        ),
        
        # DBMS package usage
        'dbms_usage': re.compile(
            r'(DBMS_\w+|UTL_\w+)\.(\w+)',
            re.IGNORECASE
        ),
        
        # Dynamic SQL
        'dynamic_sql': re.compile(
            r'EXECUTE\s+IMMEDIATE|DBMS_SQL',
            re.IGNORECASE
        ),
        
        # Bulk operations
        'bulk_collect': re.compile(
            r'BULK\s+COLLECT\s+INTO',
            re.IGNORECASE
        ),
        
        'forall': re.compile(
            r'FORALL\s+\w+\s+IN',
            re.IGNORECASE
        ),
        
        # REF CURSOR
        'ref_cursor': re.compile(
            r'REF\s+CURSOR|SYS_REFCURSOR',
            re.IGNORECASE
        ),
        
        # Grant statements
        'grant': re.compile(
            r'GRANT\s+\w+\s+(?:ON\s+)?(?:\w+\.)?\w+\s+TO',
            re.IGNORECASE
        ),
        
        # Dependencies (tables, views, etc.)
        'dependency': re.compile(
            r'(?:FROM|JOIN|INTO|UPDATE|DELETE\s+FROM)\s+(?:"?(\w+)"?\.)?"?(\w+)"?',
            re.IGNORECASE
        ),
        
        # Comments
        'single_comment': re.compile(r'--.*$', re.MULTILINE),
        'multi_comment': re.compile(r'/\*.*?\*/', re.DOTALL),
        
        # String literals
        'string_literal': re.compile(r"'[^']*'"),
        
        # END statements
        'end_statement': re.compile(
            r'END\s+(?:IF|LOOP|CASE|(\w+))?\s*;',
            re.IGNORECASE | re.MULTILINE
        )
    }
    
    def __init__(self):
        self.objects = []
        self.current_object = None
        self.object_stack = []
        self.dependencies = set()
        self.dbms_packages = set()
        
    def analyze_plsql(self, content: str, file_path: Optional[Path] = None) -> PLSQLStructure:
        """
        Analyze PL/SQL code structure
        
        Args:
            content: PL/SQL source code
            file_path: Optional file path
            
        Returns:
            PLSQLStructure analysis
        """
        try:
            lines = content.split('\n')
            
            # Remove comments for parsing (but preserve line numbers)
            clean_content = self._remove_comments(content)
            
            # Parse objects
            self._parse_objects(clean_content, lines)
            
            # Extract dependencies
            self._extract_dependencies(content)
            
            # Extract DBMS package usage
            self._extract_dbms_usage(content)
            
            # Check for specific features
            has_dynamic_sql = bool(self.PATTERNS['dynamic_sql'].search(content))
            has_bulk_collect = bool(self.PATTERNS['bulk_collect'].search(content))
            has_forall = bool(self.PATTERNS['forall'].search(content))
            has_ref_cursors = bool(self.PATTERNS['ref_cursor'].search(content))
            
            # Create structure
            structure = PLSQLStructure(
                objects=self.objects,
                total_lines=len(lines),
                total_objects=len(self.objects),
                has_package_spec=any(o.element_type == PLSQLObjectType.PACKAGE_SPEC for o in self.objects),
                has_package_body=any(o.element_type == PLSQLObjectType.PACKAGE_BODY for o in self.objects),
                has_anonymous_blocks=any(o.element_type == PLSQLObjectType.ANONYMOUS_BLOCK for o in self.objects),
                uses_dbms_packages=self.dbms_packages,
                uses_cursors=any(o.element_type == PLSQLObjectType.CURSOR for o in self.objects),
                uses_dynamic_sql=has_dynamic_sql,
                uses_bulk_collect=has_bulk_collect,
                uses_forall=has_forall,
                uses_ref_cursors=has_ref_cursors,
                uses_exceptions=bool(self.PATTERNS['exception'].search(content)),
                dependencies=self.dependencies,
                compilation_errors=[],
                statistics=self._calculate_statistics()
            )
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing PL/SQL: {e}")
            return self._create_error_structure(str(e))
    
    def _remove_comments(self, content: str) -> str:
        """Remove comments while preserving line structure"""
        # Remove multi-line comments
        content = self.PATTERNS['multi_comment'].sub('', content)
        
        # Remove single-line comments
        content = self.PATTERNS['single_comment'].sub('', content)
        
        return content
    
    def _parse_objects(self, content: str, lines: List[str]):
        """Parse PL/SQL objects from content"""
        # Parse packages
        self._parse_packages(content, lines)
        
        # Parse standalone procedures and functions
        self._parse_procedures_functions(content, lines)
        
        # Parse triggers
        self._parse_triggers(content, lines)
        
        # Parse types
        self._parse_types(content, lines)
        
        # Parse views
        self._parse_views(content, lines)
        
        # Parse anonymous blocks
        self._parse_anonymous_blocks(content, lines)
    
    def _parse_packages(self, content: str, lines: List[str]):
        """Parse package specifications and bodies"""
        # Parse package specifications
        for match in self.PATTERNS['package_spec'].finditer(content):
            if 'BODY' not in match.group(0).upper():
                schema = match.group(1)
                name = match.group(2)
                start_pos = match.start()
                
                # Find END statement
                end_pattern = re.compile(
                    rf'END\s+(?:{re.escape(name)}|\s*);',
                    re.IGNORECASE | re.MULTILINE
                )
                end_match = end_pattern.search(content, start_pos)
                
                if end_match:
                    end_pos = end_match.end()
                    object_content = content[start_pos:end_pos]
                    
                    package = PLSQLElement(
                        element_type=PLSQLObjectType.PACKAGE_SPEC,
                        name=name,
                        schema=schema,
                        content=object_content,
                        start_line=content[:start_pos].count('\n') + 1,
                        end_line=content[:end_pos].count('\n') + 1,
                        parent=None,
                        children=[],
                        parameters=[],
                        return_type=None,
                        dependencies=set(),
                        grants=[],
                        pragmas=[],
                        exceptions=[],
                        is_autonomous=False,
                        is_deterministic=False,
                        is_pipelined=False,
                        metadata={'type': 'specification'}
                    )
                    
                    # Parse package contents
                    self._parse_package_contents(package, object_content)
                    
                    self.objects.append(package)
        
        # Parse package bodies
        for match in self.PATTERNS['package_body'].finditer(content):
            schema = match.group(1)
            name = match.group(2)
            start_pos = match.start()
            
            # Find END statement
            end_pattern = re.compile(
                rf'END\s+(?:{re.escape(name)}|\s*);',
                re.IGNORECASE | re.MULTILINE
            )
            end_match = end_pattern.search(content, start_pos)
            
            if end_match:
                end_pos = end_match.end()
                object_content = content[start_pos:end_pos]
                
                package = PLSQLElement(
                    element_type=PLSQLObjectType.PACKAGE_BODY,
                    name=name,
                    schema=schema,
                    content=object_content,
                    start_line=content[:start_pos].count('\n') + 1,
                    end_line=content[:end_pos].count('\n') + 1,
                    parent=None,
                    children=[],
                    parameters=[],
                    return_type=None,
                    dependencies=set(),
                    grants=[],
                    pragmas=[],
                    exceptions=[],
                    is_autonomous=False,
                    is_deterministic=False,
                    is_pipelined=False,
                    metadata={'type': 'body'}
                )
                
                # Parse package body contents
                self._parse_package_body_contents(package, object_content)
                
                self.objects.append(package)
    
    def _parse_package_contents(self, package: PLSQLElement, content: str):
        """Parse package specification contents"""
        # Parse cursors
        for match in self.PATTERNS['cursor'].finditer(content):
            cursor_name = match.group(1)
            cursor = PLSQLElement(
                element_type=PLSQLObjectType.CURSOR,
                name=cursor_name,
                schema=package.schema,
                content=match.group(0),
                start_line=0,
                end_line=0,
                parent=package,
                children=[],
                parameters=[],
                return_type=None,
                dependencies=set(),
                grants=[],
                pragmas=[],
                exceptions=[],
                is_autonomous=False,
                is_deterministic=False,
                is_pipelined=False,
                metadata={}
            )
            package.children.append(cursor)
        
        # Parse type definitions
        for pattern_name, pattern in [
            ('type_record', self.PATTERNS['type_record']),
            ('type_table', self.PATTERNS['type_table']),
            ('type_varray', self.PATTERNS['type_varray'])
        ]:
            for match in pattern.finditer(content):
                type_name = match.group(1)
                type_elem = PLSQLElement(
                    element_type=PLSQLObjectType.RECORD if 'record' in pattern_name else
                                PLSQLObjectType.TABLE_TYPE if 'table' in pattern_name else
                                PLSQLObjectType.VARRAY,
                    name=type_name,
                    schema=package.schema,
                    content=match.group(0),
                    start_line=0,
                    end_line=0,
                    parent=package,
                    children=[],
                    parameters=[],
                    return_type=None,
                    dependencies=set(),
                    grants=[],
                    pragmas=[],
                    exceptions=[],
                    is_autonomous=False,
                    is_deterministic=False,
                    is_pipelined=False,
                    metadata={'type_kind': pattern_name}
                )
                package.children.append(type_elem)
        
        # Parse exceptions
        for match in self.PATTERNS['exception'].finditer(content):
            exception_name = match.group(1)
            package.exceptions.append(exception_name)
        
        # Parse pragmas
        for match in self.PATTERNS['pragma'].finditer(content):
            pragma_type = match.group(1)
            package.pragmas.append(pragma_type)
            
            if pragma_type == 'AUTONOMOUS_TRANSACTION':
                package.is_autonomous = True
    
    def _parse_package_body_contents(self, package: PLSQLElement, content: str):
        """Parse package body contents"""
        # Parse procedures and functions within the package
        self._parse_internal_procedures_functions(package, content)
        
        # Parse initialization block if present
        init_pattern = re.compile(
            r'BEGIN\s+(.*?)(?:END\s+\w+\s*;|$)',
            re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        
        init_match = init_pattern.search(content)
        if init_match:
            # Check if this is the package initialization block
            # (appears after all procedure/function definitions)
            init_content = init_match.group(1)
            if init_content.strip():
                package.metadata['has_initialization'] = True
                package.metadata['initialization_block'] = init_content
    
    def _parse_internal_procedures_functions(self, parent: PLSQLElement, content: str):
        """Parse procedures and functions within a package body"""
        # Parse procedures
        for match in self.PATTERNS['procedure'].finditer(content):
            if not match.group(0).upper().startswith('CREATE'):
                proc_name = match.group(2) or match.group(1)
                if proc_name:
                    procedure = PLSQLElement(
                        element_type=PLSQLObjectType.PROCEDURE,
                        name=proc_name,
                        schema=parent.schema,
                        content='',
                        start_line=0,
                        end_line=0,
                        parent=parent,
                        children=[],
                        parameters=self._extract_parameters(match.group(0)),
                        return_type=None,
                        dependencies=set(),
                        grants=[],
                        pragmas=[],
                        exceptions=[],
                        is_autonomous=False,
                        is_deterministic=False,
                        is_pipelined=False,
                        metadata={'is_internal': True}
                    )
                    parent.children.append(procedure)
        
        # Parse functions
        for match in self.PATTERNS['function'].finditer(content):
            if not match.group(0).upper().startswith('CREATE'):
                func_name = match.group(2) or match.group(1)
                if func_name:
                    # Extract return type
                    return_match = re.search(
                        r'RETURN\s+(\w+)',
                        match.group(0),
                        re.IGNORECASE
                    )
                    return_type = return_match.group(1) if return_match else None
                    
                    function = PLSQLElement(
                        element_type=PLSQLObjectType.FUNCTION,
                        name=func_name,
                        schema=parent.schema,
                        content='',
                        start_line=0,
                        end_line=0,
                        parent=parent,
                        children=[],
                        parameters=self._extract_parameters(match.group(0)),
                        return_type=return_type,
                        dependencies=set(),
                        grants=[],
                        pragmas=[],
                        exceptions=[],
                        is_autonomous=False,
                        is_deterministic='DETERMINISTIC' in match.group(0).upper(),
                        is_pipelined='PIPELINED' in match.group(0).upper(),
                        metadata={'is_internal': True}
                    )
                    parent.children.append(function)
    
    def _parse_procedures_functions(self, content: str, lines: List[str]):
        """Parse standalone procedures and functions"""
        # Parse standalone procedures
        for match in self.PATTERNS['procedure'].finditer(content):
            if match.group(0).upper().startswith('CREATE'):
                schema = match.group(1)
                name = match.group(2)
                start_pos = match.start()
                
                # Find END statement
                end_pattern = re.compile(
                    rf'END\s+(?:{re.escape(name)}|\s*);',
                    re.IGNORECASE | re.MULTILINE
                )
                end_match = end_pattern.search(content, start_pos)
                
                if end_match:
                    end_pos = end_match.end()
                    object_content = content[start_pos:end_pos]
                    
                    procedure = PLSQLElement(
                        element_type=PLSQLObjectType.PROCEDURE,
                        name=name,
                        schema=schema,
                        content=object_content,
                        start_line=content[:start_pos].count('\n') + 1,
                        end_line=content[:end_pos].count('\n') + 1,
                        parent=None,
                        children=[],
                        parameters=self._extract_parameters(object_content),
                        return_type=None,
                        dependencies=set(),
                        grants=[],
                        pragmas=self._extract_pragmas(object_content),
                        exceptions=[],
                        is_autonomous='AUTONOMOUS_TRANSACTION' in object_content.upper(),
                        is_deterministic=False,
                        is_pipelined=False,
                        metadata={'is_standalone': True}
                    )
                    
                    self.objects.append(procedure)
        
        # Parse standalone functions
        for match in self.PATTERNS['function'].finditer(content):
            if match.group(0).upper().startswith('CREATE'):
                schema = match.group(1)
                name = match.group(2)
                start_pos = match.start()
                
                # Find END statement
                end_pattern = re.compile(
                    rf'END\s+(?:{re.escape(name)}|\s*);',
                    re.IGNORECASE | re.MULTILINE
                )
                end_match = end_pattern.search(content, start_pos)
                
                if end_match:
                    end_pos = end_match.end()
                    object_content = content[start_pos:end_pos]
                    
                    # Extract return type
                    return_match = re.search(
                        r'RETURN\s+(\w+)',
                        object_content,
                        re.IGNORECASE
                    )
                    return_type = return_match.group(1) if return_match else None
                    
                    function = PLSQLElement(
                        element_type=PLSQLObjectType.FUNCTION,
                        name=name,
                        schema=schema,
                        content=object_content,
                        start_line=content[:start_pos].count('\n') + 1,
                        end_line=content[:end_pos].count('\n') + 1,
                        parent=None,
                        children=[],
                        parameters=self._extract_parameters(object_content),
                        return_type=return_type,
                        dependencies=set(),
                        grants=[],
                        pragmas=self._extract_pragmas(object_content),
                        exceptions=[],
                        is_autonomous='AUTONOMOUS_TRANSACTION' in object_content.upper(),
                        is_deterministic='DETERMINISTIC' in object_content.upper(),
                        is_pipelined='PIPELINED' in object_content.upper(),
                        metadata={'is_standalone': True}
                    )
                    
                    self.objects.append(function)
    
    def _parse_triggers(self, content: str, lines: List[str]):
        """Parse triggers"""
        for match in self.PATTERNS['trigger'].finditer(content):
            schema = match.group(1)
            name = match.group(2)
            start_pos = match.start()
            
            # Find END statement
            end_pattern = re.compile(
                rf'END\s+(?:{re.escape(name)}|\s*);',
                re.IGNORECASE | re.MULTILINE
            )
            end_match = end_pattern.search(content, start_pos)
            
            if end_match:
                end_pos = end_match.end()
                object_content = content[start_pos:end_pos]
                
                # Extract trigger details
                trigger_info = self._extract_trigger_info(object_content)
                
                trigger = PLSQLElement(
                    element_type=PLSQLObjectType.TRIGGER,
                    name=name,
                    schema=schema,
                    content=object_content,
                    start_line=content[:start_pos].count('\n') + 1,
                    end_line=content[:end_pos].count('\n') + 1,
                    parent=None,
                    children=[],
                    parameters=[],
                    return_type=None,
                    dependencies=set([trigger_info.get('table_name')]) if trigger_info.get('table_name') else set(),
                    grants=[],
                    pragmas=self._extract_pragmas(object_content),
                    exceptions=[],
                    is_autonomous='AUTONOMOUS_TRANSACTION' in object_content.upper(),
                    is_deterministic=False,
                    is_pipelined=False,
                    metadata=trigger_info
                )
                
                self.objects.append(trigger)
    
    def _parse_types(self, content: str, lines: List[str]):
        """Parse type specifications and bodies"""
        for match in self.PATTERNS['type_spec'].finditer(content):
            schema = match.group(1)
            name = match.group(2)
            is_body = 'BODY' in match.group(0).upper()
            start_pos = match.start()
            
            # Find END statement or semicolon
            if 'OBJECT' in content[start_pos:start_pos+200].upper():
                # Object type - look for END
                end_pattern = re.compile(
                    rf'END\s*;',
                    re.IGNORECASE | re.MULTILINE
                )
                end_match = end_pattern.search(content, start_pos)
            else:
                # Simple type - look for semicolon
                end_match = re.search(r';', content[start_pos:])
                if end_match:
                    end_match = type('Match', (), {
                        'end': lambda: start_pos + end_match.end()
                    })()
            
            if end_match:
                end_pos = end_match.end()
                object_content = content[start_pos:end_pos]
                
                type_elem = PLSQLElement(
                    element_type=PLSQLObjectType.TYPE_BODY if is_body else PLSQLObjectType.TYPE_SPEC,
                    name=name,
                    schema=schema,
                    content=object_content,
                    start_line=content[:start_pos].count('\n') + 1,
                    end_line=content[:end_pos].count('\n') + 1,
                    parent=None,
                    children=[],
                    parameters=[],
                    return_type=None,
                    dependencies=set(),
                    grants=[],
                    pragmas=[],
                    exceptions=[],
                    is_autonomous=False,
                    is_deterministic=False,
                    is_pipelined=False,
                    metadata={'is_body': is_body}
                )
                
                self.objects.append(type_elem)
    
    def _parse_views(self, content: str, lines: List[str]):
        """Parse views and materialized views"""
        # Parse regular views
        for match in self.PATTERNS['view'].finditer(content):
            schema = match.group(1)
            name = match.group(2)
            start_pos = match.start()
            
            # Find AS SELECT
            as_match = re.search(
                r'AS\s+SELECT',
                content[start_pos:],
                re.IGNORECASE
            )
            
            if as_match:
                # Find end (semicolon or next CREATE)
                end_pattern = re.compile(
                    r';|(?=CREATE\s)',
                    re.IGNORECASE | re.MULTILINE
                )
                end_match = end_pattern.search(content, start_pos + as_match.start())
                
                if end_match:
                    end_pos = end_match.start() + 1
                    object_content = content[start_pos:end_pos]
                    
                    view = PLSQLElement(
                        element_type=PLSQLObjectType.VIEW,
                        name=name,
                        schema=schema,
                        content=object_content,
                        start_line=content[:start_pos].count('\n') + 1,
                        end_line=content[:end_pos].count('\n') + 1,
                        parent=None,
                        children=[],
                        parameters=[],
                        return_type=None,
                        dependencies=self._extract_view_dependencies(object_content),
                        grants=[],
                        pragmas=[],
                        exceptions=[],
                        is_autonomous=False,
                        is_deterministic=False,
                        is_pipelined=False,
                        metadata={'is_force': 'FORCE' in match.group(0).upper()}
                    )
                    
                    self.objects.append(view)
    
    def _parse_anonymous_blocks(self, content: str, lines: List[str]):
        """Parse anonymous PL/SQL blocks"""
        for match in self.PATTERNS['anonymous_block'].finditer(content):
            start_pos = match.start()
            
            # Find matching END
            end_pattern = re.compile(
                r'END\s*;',
                re.IGNORECASE | re.MULTILINE
            )
            
            # Track nested BEGIN/END blocks
            block_depth = 1
            search_pos = start_pos + len(match.group(0))
            
            while block_depth > 0 and search_pos < len(content):
                # Look for BEGIN or END
                begin_match = re.search(r'\bBEGIN\b', content[search_pos:], re.IGNORECASE)
                end_match = end_pattern.search(content[search_pos:])
                
                if end_match and (not begin_match or end_match.start() < begin_match.start()):
                    block_depth -= 1
                    if block_depth == 0:
                        end_pos = search_pos + end_match.end()
                        break
                    search_pos += end_match.end()
                elif begin_match:
                    block_depth += 1
                    search_pos += begin_match.end()
                else:
                    break
            
            if block_depth == 0:
                object_content = content[start_pos:end_pos]
                
                anonymous = PLSQLElement(
                    element_type=PLSQLObjectType.ANONYMOUS_BLOCK,
                    name=f"anonymous_block_{len(self.objects) + 1}",
                    schema=None,
                    content=object_content,
                    start_line=content[:start_pos].count('\n') + 1,
                    end_line=content[:end_pos].count('\n') + 1,
                    parent=None,
                    children=[],
                    parameters=[],
                    return_type=None,
                    dependencies=set(),
                    grants=[],
                    pragmas=[],
                    exceptions=[],
                    is_autonomous=False,
                    is_deterministic=False,
                    is_pipelined=False,
                    metadata={'has_declare': 'DECLARE' in match.group(0).upper()}
                )
                
                self.objects.append(anonymous)
    
    def _extract_parameters(self, content: str) -> List[Dict[str, Any]]:
        """Extract parameters from procedure/function signature"""
        parameters = []
        
        # Find parameter list
        param_match = re.search(
            r'\(([^)]+)\)',
            content,
            re.IGNORECASE | re.DOTALL
        )
        
        if param_match:
            param_str = param_match.group(1)
            
            # Split by comma (but not within nested parentheses)
            params = self._split_parameters(param_str)
            
            for param in params:
                param = param.strip()
                if param:
                    # Parse parameter
                    param_info = self._parse_parameter(param)
                    if param_info:
                        parameters.append(param_info)
        
        return parameters
    
    def _split_parameters(self, param_str: str) -> List[str]:
        """Split parameter string by commas, respecting parentheses"""
        params = []
        current_param = ''
        paren_depth = 0
        
        for char in param_str:
            if char == ',' and paren_depth == 0:
                params.append(current_param)
                current_param = ''
            else:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                current_param += char
        
        if current_param:
            params.append(current_param)
        
        return params
    
    def _parse_parameter(self, param: str) -> Optional[Dict[str, Any]]:
        """Parse a single parameter"""
        # Pattern for parameter: name [IN|OUT|IN OUT] type [DEFAULT value]
        param_pattern = re.compile(
            r'(\w+)\s*(?:(IN|OUT|IN\s+OUT)\s+)?(\w+(?:\([^)]+\))?)'
            r'(?:\s+DEFAULT\s+(.+))?',
            re.IGNORECASE
        )
        
        match = param_pattern.match(param.strip())
        if match:
            return {
                'name': match.group(1),
                'mode': match.group(2) or 'IN',
                'type': match.group(3),
                'default': match.group(4)
            }
        
        return None
    
    def _extract_pragmas(self, content: str) -> List[str]:
        """Extract pragma directives"""
        pragmas = []
        
        for match in self.PATTERNS['pragma'].finditer(content):
            pragmas.append(match.group(1))
        
        return pragmas
    
    def _extract_trigger_info(self, content: str) -> Dict[str, Any]:
        """Extract trigger information"""
        info = {}
        
        # Trigger timing (BEFORE/AFTER/INSTEAD OF)
        timing_match = re.search(
            r'(BEFORE|AFTER|INSTEAD\s+OF)',
            content,
            re.IGNORECASE
        )
        if timing_match:
            info['timing'] = timing_match.group(1)
        
        # Trigger event (INSERT/UPDATE/DELETE)
        event_match = re.search(
            r'(INSERT|UPDATE|DELETE)(?:\s+OR\s+(INSERT|UPDATE|DELETE))*',
            content,
            re.IGNORECASE
        )
        if event_match:
            info['events'] = event_match.group(0)
        
        # Table name
        table_match = re.search(
            r'ON\s+(?:"?(\w+)"?\.)?"?(\w+)"?',
            content,
            re.IGNORECASE
        )
        if table_match:
            info['table_schema'] = table_match.group(1)
            info['table_name'] = table_match.group(2)
        
        # FOR EACH ROW
        if re.search(r'FOR\s+EACH\s+ROW', content, re.IGNORECASE):
            info['is_row_level'] = True
        else:
            info['is_row_level'] = False
        
        # WHEN clause
        when_match = re.search(
            r'WHEN\s+\(([^)]+)\)',
            content,
            re.IGNORECASE
        )
        if when_match:
            info['when_condition'] = when_match.group(1)
        
        return info
    
    def _extract_view_dependencies(self, content: str) -> Set[str]:
        """Extract table dependencies from view"""
        dependencies = set()
        
        for match in self.PATTERNS['dependency'].finditer(content):
            schema = match.group(1)
            table = match.group(2)
            
            if schema:
                dependencies.add(f"{schema}.{table}")
            else:
                dependencies.add(table)
        
        return dependencies
    
    def _extract_dependencies(self, content: str):
        """Extract all dependencies from content"""
        for match in self.PATTERNS['dependency'].finditer(content):
            schema = match.group(1)
            object_name = match.group(2)
            
            if schema:
                self.dependencies.add(f"{schema}.{object_name}")
            else:
                self.dependencies.add(object_name)
    
    def _extract_dbms_usage(self, content: str):
        """Extract DBMS package usage"""
        for match in self.PATTERNS['dbms_usage'].finditer(content):
            package = match.group(1)
            if package.upper() in self.DBMS_PACKAGES:
                self.dbms_packages.add(package.upper())
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about the PL/SQL code"""
        stats = {
            'object_types': defaultdict(int),
            'total_procedures': 0,
            'total_functions': 0,
            'total_triggers': 0,
            'total_packages': 0,
            'total_cursors': 0,
            'total_exceptions': 0,
            'autonomous_transactions': 0,
            'deterministic_functions': 0,
            'pipelined_functions': 0
        }
        
        for obj in self.objects:
            stats['object_types'][obj.element_type.value] += 1
            
            if obj.element_type == PLSQLObjectType.PROCEDURE:
                stats['total_procedures'] += 1
            elif obj.element_type == PLSQLObjectType.FUNCTION:
                stats['total_functions'] += 1
            elif obj.element_type == PLSQLObjectType.TRIGGER:
                stats['total_triggers'] += 1
            elif obj.element_type in [PLSQLObjectType.PACKAGE_SPEC, PLSQLObjectType.PACKAGE_BODY]:
                stats['total_packages'] += 1
            
            # Count features
            if obj.is_autonomous:
                stats['autonomous_transactions'] += 1
            if obj.is_deterministic:
                stats['deterministic_functions'] += 1
            if obj.is_pipelined:
                stats['pipelined_functions'] += 1
            
            # Count children
            for child in obj.children:
                if child.element_type == PLSQLObjectType.CURSOR:
                    stats['total_cursors'] += 1
            
            stats['total_exceptions'] += len(obj.exceptions)
        
        return dict(stats)
    
    def _create_error_structure(self, error_msg: str) -> PLSQLStructure:
        """Create error structure for invalid PL/SQL"""
        return PLSQLStructure(
            objects=[],
            total_lines=0,
            total_objects=0,
            has_package_spec=False,
            has_package_body=False,
            has_anonymous_blocks=False,
            uses_dbms_packages=set(),
            uses_cursors=False,
            uses_dynamic_sql=False,
            uses_bulk_collect=False,
            uses_forall=False,
            uses_ref_cursors=False,
            uses_exceptions=False,
            dependencies=set(),
            compilation_errors=[error_msg],
            statistics={}
        )

class PLSQLChunker(BaseChunker):
    """Chunker specialized for PL/SQL code"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, max_tokens)
        self.analyzer = PLSQLAnalyzer()
        
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from PL/SQL file
        
        Args:
            content: PL/SQL source code
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Analyze PL/SQL structure
            structure = self.analyzer.analyze_plsql(content, file_context.path)
            
            chunks = []
            
            # Process each object
            for obj in structure.objects:
                obj_chunks = self._chunk_object(obj, structure)
                chunks.extend(obj_chunks)
            
            # If no objects found, fall back to line-based chunking
            if not chunks:
                chunks = self._fallback_chunking(content, file_context)
            
            # Add metadata to all chunks
            for chunk in chunks:
                chunk.metadata['language'] = 'plsql'
                chunk.metadata['uses_dbms'] = list(structure.uses_dbms_packages)
                chunk.metadata['has_dynamic_sql'] = structure.uses_dynamic_sql
            
            logger.info(f"Created {len(chunks)} chunks for PL/SQL file {file_context.path}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking PL/SQL file {file_context.path}: {e}")
            return self._fallback_chunking(content, file_context)
    
    def _chunk_object(self, obj: PLSQLElement, structure: PLSQLStructure) -> List[Chunk]:
        """Create chunks for a PL/SQL object"""
        chunks = []
        
        if obj.element_type == PLSQLObjectType.PACKAGE_SPEC:
            chunks.extend(self._chunk_package_spec(obj))
        elif obj.element_type == PLSQLObjectType.PACKAGE_BODY:
            chunks.extend(self._chunk_package_body(obj))
        elif obj.element_type in [PLSQLObjectType.PROCEDURE, PLSQLObjectType.FUNCTION]:
            chunks.extend(self._chunk_procedure_function(obj))
        elif obj.element_type == PLSQLObjectType.TRIGGER:
            chunks.extend(self._chunk_trigger(obj))
        elif obj.element_type in [PLSQLObjectType.TYPE_SPEC, PLSQLObjectType.TYPE_BODY]:
            chunks.extend(self._chunk_type(obj))
        elif obj.element_type == PLSQLObjectType.VIEW:
            chunks.extend(self._chunk_view(obj))
        else:
            # Generic object chunking
            chunks.extend(self._chunk_generic_object(obj))
        
        return chunks
    
    def _chunk_package_spec(self, package: PLSQLElement) -> List[Chunk]:
        """Chunk package specification"""
        chunks = []
        
        # Package header with all declarations
        if self.count_tokens(package.content) <= self.max_tokens:
            chunks.append(self.create_chunk(
                content=package.content,
                chunk_type='plsql_package_spec',
                metadata={
                    'package_name': package.name,
                    'schema': package.schema,
                    'cursor_count': sum(1 for c in package.children if c.element_type == PLSQLObjectType.CURSOR),
                    'exception_count': len(package.exceptions),
                    'has_pragmas': bool(package.pragmas)
                },
                file_path=package.name + '.pks',
                start_line=package.start_line,
                end_line=package.end_line
            ))
        else:
            # Split large package spec
            chunks.extend(self._split_large_package(package))
        
        return chunks
    
    def _chunk_package_body(self, package: PLSQLElement) -> List[Chunk]:
        """Chunk package body"""
        chunks = []
        
        # Package header
        header_lines = []
        header_lines.append(f"CREATE OR REPLACE PACKAGE BODY {package.name} AS")
        header_lines.append("")
        
        # Private declarations (if any)
        if package.metadata.get('has_private_declarations'):
            header_lines.append("-- Private declarations")
            header_lines.append("")
        
        header_content = '\n'.join(header_lines)
        
        chunks.append(self.create_chunk(
            content=header_content,
            chunk_type='plsql_package_body_header',
            metadata={
                'package_name': package.name,
                'schema': package.schema,
                'procedure_count': sum(1 for c in package.children if c.element_type == PLSQLObjectType.PROCEDURE),
                'function_count': sum(1 for c in package.children if c.element_type == PLSQLObjectType.FUNCTION)
            },
            file_path=package.name + '.pkb'
        ))
        
        # Chunk each procedure/function in the body
        for child in package.children:
            if child.element_type in [PLSQLObjectType.PROCEDURE, PLSQLObjectType.FUNCTION]:
                child_chunks = self._chunk_procedure_function(child, parent_package=package.name)
                chunks.extend(child_chunks)
        
        # Initialization block if present
        if package.metadata.get('has_initialization'):
            init_content = f"-- Package initialization\nBEGIN\n{package.metadata['initialization_block']}\nEND {package.name};"
            
            chunks.append(self.create_chunk(
                content=init_content,
                chunk_type='plsql_package_init',
                metadata={
                    'package_name': package.name,
                    'is_initialization': True
                },
                file_path=package.name + '.pkb'
            ))
        
        return chunks
    
    def _chunk_procedure_function(self, obj: PLSQLElement, parent_package: Optional[str] = None) -> List[Chunk]:
        """Chunk procedure or function"""
        chunks = []
        
        # Check size
        if self.count_tokens(obj.content) <= self.max_tokens:
            chunks.append(self.create_chunk(
                content=obj.content,
                chunk_type=f'plsql_{obj.element_type.value}',
                metadata={
                    'name': obj.name,
                    'schema': obj.schema,
                    'package': parent_package,
                    'parameters': obj.parameters,
                    'return_type': obj.return_type,
                    'is_autonomous': obj.is_autonomous,
                    'is_deterministic': obj.is_deterministic,
                    'is_pipelined': obj.is_pipelined
                },
                file_path=f"{parent_package or obj.schema or 'unknown'}.{obj.name}",
                start_line=obj.start_line,
                end_line=obj.end_line
            ))
        else:
            # Split large procedure/function
            chunks.extend(self._split_large_procedure_function(obj, parent_package))
        
        return chunks
    
    def _chunk_trigger(self, trigger: PLSQLElement) -> List[Chunk]:
        """Chunk trigger"""
        chunks = []
        
        chunks.append(self.create_chunk(
            content=trigger.content,
            chunk_type='plsql_trigger',
            metadata={
                'trigger_name': trigger.name,
                'schema': trigger.schema,
                'timing': trigger.metadata.get('timing'),
                'events': trigger.metadata.get('events'),
                'table_name': trigger.metadata.get('table_name'),
                'is_row_level': trigger.metadata.get('is_row_level'),
                'when_condition': trigger.metadata.get('when_condition'),
                'is_autonomous': trigger.is_autonomous
            },
            file_path=f"{trigger.name}.trg",
            start_line=trigger.start_line,
            end_line=trigger.end_line
        ))
        
        return chunks
    
    def _chunk_type(self, type_obj: PLSQLElement) -> List[Chunk]:
        """Chunk type specification or body"""
        chunks = []
        
        chunks.append(self.create_chunk(
            content=type_obj.content,
            chunk_type=f'plsql_{type_obj.element_type.value}',
            metadata={
                'type_name': type_obj.name,
                'schema': type_obj.schema,
                'is_body': type_obj.metadata.get('is_body', False)
            },
            file_path=f"{type_obj.name}.tps",
            start_line=type_obj.start_line,
            end_line=type_obj.end_line
        ))
        
        return chunks
    
    def _chunk_view(self, view: PLSQLElement) -> List[Chunk]:
        """Chunk view"""
        chunks = []
        
        chunks.append(self.create_chunk(
            content=view.content,
            chunk_type='plsql_view',
            metadata={
                'view_name': view.name,
                'schema': view.schema,
                'dependencies': list(view.dependencies),
                'is_force': view.metadata.get('is_force', False)
            },
            file_path=f"{view.name}.vw",
            start_line=view.start_line,
            end_line=view.end_line
        ))
        
        return chunks
    
    def _chunk_generic_object(self, obj: PLSQLElement) -> List[Chunk]:
        """Chunk generic PL/SQL object"""
        chunks = []
        
        chunks.append(self.create_chunk(
            content=obj.content,
            chunk_type=f'plsql_{obj.element_type.value}',
            metadata={
                'name': obj.name,
                'schema': obj.schema,
                'object_type': obj.element_type.value
            },
            file_path=f"{obj.name}.sql",
            start_line=obj.start_line,
            end_line=obj.end_line
        ))
        
        return chunks
    
    def _split_large_package(self, package: PLSQLElement) -> List[Chunk]:
        """Split large package specification"""
        chunks = []
        lines = package.content.split('\n')
        
        # Package header
        header_lines = []
        current_section = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            # Identify section boundaries
            if re.match(r'\s*(PROCEDURE|FUNCTION|CURSOR|TYPE|EXCEPTION)', line, re.IGNORECASE):
                if current_tokens + line_tokens > self.max_tokens and current_section:
                    # Create chunk for current section
                    chunks.append(self.create_chunk(
                        content='\n'.join(current_section),
                        chunk_type='plsql_package_spec_part',
                        metadata={
                            'package_name': package.name,
                            'schema': package.schema,
                            'part': len(chunks) + 1
                        },
                        file_path=f"{package.name}.pks"
                    ))
                    current_section = []
                    current_tokens = 0
            
            current_section.append(line)
            current_tokens += line_tokens
        
        # Add remaining section
        if current_section:
            chunks.append(self.create_chunk(
                content='\n'.join(current_section),
                chunk_type='plsql_package_spec_part',
                metadata={
                    'package_name': package.name,
                    'schema': package.schema,
                    'part': len(chunks) + 1,
                    'is_last': True
                },
                file_path=f"{package.name}.pks"
            ))
        
        return chunks
    
    def _split_large_procedure_function(self, obj: PLSQLElement, 
                                       parent_package: Optional[str] = None) -> List[Chunk]:
        """Split large procedure or function"""
        chunks = []
        lines = obj.content.split('\n')
        
        # Find declaration section
        begin_index = next((i for i, line in enumerate(lines) 
                          if re.match(r'\s*BEGIN\s*$', line, re.IGNORECASE)), -1)
        
        if begin_index > 0:
            # Declaration section
            declaration = '\n'.join(lines[:begin_index + 1])
            
            chunks.append(self.create_chunk(
                content=declaration,
                chunk_type=f'plsql_{obj.element_type.value}_declaration',
                metadata={
                    'name': obj.name,
                    'package': parent_package,
                    'part': 'declaration'
                },
                file_path=f"{parent_package or 'standalone'}.{obj.name}"
            ))
            
            # Body section
            body_lines = lines[begin_index + 1:]
            
            # Split body into logical sections
            current_section = []
            current_tokens = 0
            section_num = 1
            
            for line in body_lines:
                line_tokens = self.count_tokens(line)
                
                if current_tokens + line_tokens > self.max_tokens and current_section:
                    chunks.append(self.create_chunk(
                        content='\n'.join(current_section),
                        chunk_type=f'plsql_{obj.element_type.value}_body',
                        metadata={
                            'name': obj.name,
                            'package': parent_package,
                            'part': f'body_{section_num}'
                        },
                        file_path=f"{parent_package or 'standalone'}.{obj.name}"
                    ))
                    current_section = []
                    current_tokens = 0
                    section_num += 1
                
                current_section.append(line)
                current_tokens += line_tokens
            
            # Add remaining section
            if current_section:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_section),
                    chunk_type=f'plsql_{obj.element_type.value}_body',
                    metadata={
                        'name': obj.name,
                        'package': parent_package,
                        'part': f'body_{section_num}',
                        'is_last': True
                    },
                    file_path=f"{parent_package or 'standalone'}.{obj.name}"
                ))
        else:
            # No clear BEGIN found, split by size
            chunks.extend(self._split_by_size(obj.content, obj, parent_package))
        
        return chunks
    
    def _split_by_size(self, content: str, obj: PLSQLElement,
                      parent_package: Optional[str] = None) -> List[Chunk]:
        """Split content by size"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        part = 1
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type=f'plsql_{obj.element_type.value}_part',
                    metadata={
                        'name': obj.name,
                        'package': parent_package,
                        'part': part
                    },
                    file_path=f"{parent_package or 'standalone'}.{obj.name}"
                ))
                current_chunk = []
                current_tokens = 0
                part += 1
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add remaining
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type=f'plsql_{obj.element_type.value}_part',
                metadata={
                    'name': obj.name,
                    'package': parent_package,
                    'part': part,
                    'is_last': True
                },
                file_path=f"{parent_package or 'standalone'}.{obj.name}"
            ))
        
        return chunks
    
    def _fallback_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback chunking for invalid or unstructured PL/SQL"""
        logger.warning(f"Using fallback chunking for PL/SQL file {file_context.path}")
        
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_num = 1
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            # Try to break at logical boundaries
            is_boundary = (
                re.match(r'^\s*(?:CREATE|BEGIN|END|PROCEDURE|FUNCTION)', line, re.IGNORECASE) or
                line.strip().endswith(';')
            )
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk and is_boundary:
                chunks.append(self.create_chunk(
                    content='\n'.join(current_chunk),
                    chunk_type='plsql_fallback',
                    metadata={
                        'is_fallback': True,
                        'chunk_number': chunk_num
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = []
                current_tokens = 0
                chunk_num += 1
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add remaining
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n'.join(current_chunk),
                chunk_type='plsql_fallback',
                metadata={
                    'is_fallback': True,
                    'chunk_number': chunk_num,
                    'is_last': True
                },
                file_path=str(file_context.path)
            ))
        
        return chunks