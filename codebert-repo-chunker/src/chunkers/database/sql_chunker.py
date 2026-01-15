"""
SQL-specific chunker for intelligent semantic chunking
Handles DDL, DML, complex queries, stored procedures, CTEs, and various SQL dialects
"""

import re
import sqlparse
from sqlparse.sql import Statement, Token, TokenList, IdentifierList, Identifier, Parenthesis, Function
from sqlparse.tokens import Keyword, DML, DDL, CTE, Name, Whitespace, Comment, String
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import logging
from enum import Enum

from src.core.base_chunker import BaseChunker, Chunk, ChunkerConfig
from src.core.file_context import FileContext
from config.settings import settings

logger = logging.getLogger(__name__)

class SQLStatementType(Enum):
    """Types of SQL statements"""
    # DDL
    CREATE_TABLE = "create_table"
    CREATE_INDEX = "create_index"
    CREATE_VIEW = "create_view"
    CREATE_PROCEDURE = "create_procedure"
    CREATE_FUNCTION = "create_function"
    CREATE_TRIGGER = "create_trigger"
    CREATE_SCHEMA = "create_schema"
    CREATE_DATABASE = "create_database"
    CREATE_SEQUENCE = "create_sequence"
    CREATE_TYPE = "create_type"
    ALTER_TABLE = "alter_table"
    ALTER_SCHEMA = "alter_schema"
    DROP_TABLE = "drop_table"
    DROP_INDEX = "drop_index"
    DROP_VIEW = "drop_view"
    TRUNCATE = "truncate"
    
    # DML
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    UPSERT = "upsert"
    
    # DCL
    GRANT = "grant"
    REVOKE = "revoke"
    
    # TCL
    COMMIT = "commit"
    ROLLBACK = "rollback"
    SAVEPOINT = "savepoint"
    
    # Other
    WITH_CTE = "with_cte"
    EXPLAIN = "explain"
    ANALYZE = "analyze"
    VACUUM = "vacuum"
    SET = "set"
    USE = "use"
    CALL = "call"
    EXECUTE = "execute"
    DECLARE = "declare"
    BEGIN_END = "begin_end"
    IF_ELSE = "if_else"
    WHILE = "while"
    FOR = "for"
    CURSOR = "cursor"
    TRANSACTION = "transaction"

class SQLDialect(Enum):
    """SQL dialects"""
    STANDARD = "standard"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    PRESTO = "presto"
    HIVE = "hive"
    SPARK_SQL = "spark_sql"
    TERADATA = "teradata"
    DB2 = "db2"

@dataclass
class SQLStatement:
    """Represents a SQL statement"""
    statement_type: SQLStatementType
    content: str
    raw_content: str
    start_line: int
    end_line: int
    tables: Set[str]
    columns: Set[str]
    aliases: Dict[str, str]
    joins: List[Dict[str, Any]]
    conditions: List[str]
    subqueries: List['SQLStatement']
    ctes: List[Dict[str, Any]]
    parameters: List[str]
    functions: Set[str]
    is_nested: bool
    parent: Optional['SQLStatement']
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SQLStructure:
    """Represents overall SQL file structure"""
    statements: List[SQLStatement]
    dialect: SQLDialect
    total_statements: int
    ddl_count: int
    dml_count: int
    dcl_count: int
    tcl_count: int
    has_transactions: bool
    has_procedures: bool
    has_functions: bool
    has_triggers: bool
    has_ctes: bool
    has_temp_tables: bool
    has_views: bool
    schemas: Set[str]
    tables: Set[str]
    migration_info: Optional[Dict[str, Any]]
    statistics: Dict[str, Any]

class SQLAnalyzer:
    """Analyzes SQL structure and content"""
    
    # Dialect detection patterns
    DIALECT_PATTERNS = {
        SQLDialect.MYSQL: [
            re.compile(r'`\w+`'),  # Backtick identifiers
            re.compile(r'AUTO_INCREMENT', re.IGNORECASE),
            re.compile(r'ENGINE\s*=\s*\w+', re.IGNORECASE),
            re.compile(r'LIMIT\s+\d+\s+OFFSET', re.IGNORECASE),
        ],
        SQLDialect.POSTGRESQL: [
            re.compile(r'SERIAL\s+PRIMARY\s+KEY', re.IGNORECASE),
            re.compile(r'RETURNING\s+\*', re.IGNORECASE),
            re.compile(r'::\w+'),  # Type casting
            re.compile(r'ARRAY\[', re.IGNORECASE),
            re.compile(r'\$\$.*?\$\$', re.DOTALL),  # Dollar quoting
        ],
        SQLDialect.ORACLE: [
            re.compile(r'VARCHAR2', re.IGNORECASE),
            re.compile(r'NUMBER\s*\(\d+', re.IGNORECASE),
            re.compile(r'DUAL', re.IGNORECASE),
            re.compile(r'ROWNUM', re.IGNORECASE),
            re.compile(r'CONNECT\s+BY', re.IGNORECASE),
        ],
        SQLDialect.SQLSERVER: [
            re.compile(r'\[dbo\]'),  # Schema prefix
            re.compile(r'@@\w+'),  # System variables
            re.compile(r'GO\s*$', re.MULTILINE),
            re.compile(r'TOP\s+\d+', re.IGNORECASE),
            re.compile(r'WITH\s+\(NOLOCK\)', re.IGNORECASE),
        ],
        SQLDialect.SNOWFLAKE: [
            re.compile(r'VARIANT', re.IGNORECASE),
            re.compile(r'FLATTEN', re.IGNORECASE),
            re.compile(r'QUALIFY', re.IGNORECASE),
            re.compile(r'IFF\s*\(', re.IGNORECASE),
        ],
        SQLDialect.BIGQUERY: [
            re.compile(r'STRUCT<', re.IGNORECASE),
            re.compile(r'ARRAY<', re.IGNORECASE),
            re.compile(r'EXCEPT\s*\(', re.IGNORECASE),
            re.compile(r'`project\.dataset\.table`'),
        ],
        SQLDialect.HIVE: [
            re.compile(r'PARTITIONED\s+BY', re.IGNORECASE),
            re.compile(r'STORED\s+AS', re.IGNORECASE),
            re.compile(r'ROW\s+FORMAT', re.IGNORECASE),
            re.compile(r'LATERAL\s+VIEW', re.IGNORECASE),
        ],
    }
    
    # Migration patterns (Flyway, Liquibase, etc.)
    MIGRATION_PATTERNS = {
        'flyway': re.compile(r'^V(\d+(?:_\d+)*?)__(.+)\.sql$'),
        'liquibase': re.compile(r'--\s*changeset\s+(\w+):(\w+)', re.IGNORECASE),
        'rails': re.compile(r'^(\d{14})_(.+)\.sql$'),
        'django': re.compile(r'^(\d{4})_(.+)\.sql$'),
        'alembic': re.compile(r'^([a-f0-9]+)_(.+)\.py$'),
    }
    
    # Common SQL functions
    SQL_FUNCTIONS = {
        'aggregate': {'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT', 
                     'STRING_AGG', 'ARRAY_AGG', 'LISTAGG'},
        'window': {'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'LAG', 'LEAD', 
                  'FIRST_VALUE', 'LAST_VALUE', 'NTH_VALUE'},
        'string': {'CONCAT', 'SUBSTRING', 'REPLACE', 'TRIM', 'UPPER', 
                  'LOWER', 'LENGTH', 'REGEXP_REPLACE', 'SPLIT'},
        'date': {'DATE', 'DATETIME', 'TIMESTAMP', 'DATE_ADD', 'DATE_SUB',
                'DATEDIFF', 'DATE_TRUNC', 'EXTRACT', 'DATEPART'},
        'conversion': {'CAST', 'CONVERT', 'TRY_CAST', 'TRY_CONVERT', 'PARSE'},
        'json': {'JSON_EXTRACT', 'JSON_VALUE', 'JSON_QUERY', 'JSON_OBJECT',
                'JSON_ARRAY', 'JSON_TABLE', 'JSONB_BUILD_OBJECT'},
    }
    
    def __init__(self):
        self.statements = []
        self.current_statement = None
        self.tables = set()
        self.schemas = set()
        self.temp_tables = set()
        
    def analyze_sql(self, content: str, file_path: Optional[Path] = None) -> SQLStructure:
        """
        Analyze SQL file structure
        
        Args:
            content: SQL content
            file_path: Optional file path for context
            
        Returns:
            SQLStructure analysis
        """
        try:
            # Detect dialect
            dialect = self._detect_dialect(content, file_path)
            
            # Parse SQL statements
            parsed_statements = sqlparse.parse(content)
            
            # Analyze each statement
            line_offset = 0
            for parsed in parsed_statements:
                if str(parsed).strip():
                    statement = self._analyze_statement(parsed, line_offset)
                    if statement:
                        self.statements.append(statement)
                    
                    # Update line offset
                    line_offset += str(parsed).count('\n')
            
            # Detect migration info
            migration_info = self._detect_migration_info(file_path) if file_path else None
            
            # Calculate statistics
            structure = SQLStructure(
                statements=self.statements,
                dialect=dialect,
                total_statements=len(self.statements),
                ddl_count=self._count_statement_types(['CREATE', 'ALTER', 'DROP']),
                dml_count=self._count_statement_types(['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'MERGE']),
                dcl_count=self._count_statement_types(['GRANT', 'REVOKE']),
                tcl_count=self._count_statement_types(['COMMIT', 'ROLLBACK', 'SAVEPOINT']),
                has_transactions=self._has_transactions(content),
                has_procedures=any(s.statement_type == SQLStatementType.CREATE_PROCEDURE for s in self.statements),
                has_functions=any(s.statement_type == SQLStatementType.CREATE_FUNCTION for s in self.statements),
                has_triggers=any(s.statement_type == SQLStatementType.CREATE_TRIGGER for s in self.statements),
                has_ctes=any(s.ctes for s in self.statements),
                has_temp_tables=bool(self.temp_tables),
                has_views=any(s.statement_type == SQLStatementType.CREATE_VIEW for s in self.statements),
                schemas=self.schemas,
                tables=self.tables,
                migration_info=migration_info,
                statistics=self._calculate_statistics()
            )
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing SQL: {e}")
            return self._create_error_structure(str(e))
    
    def _detect_dialect(self, content: str, file_path: Optional[Path]) -> SQLDialect:
        """Detect SQL dialect from content and file"""
        # Check file name hints
        if file_path:
            name_lower = str(file_path).lower()
            if 'mysql' in name_lower:
                return SQLDialect.MYSQL
            elif 'postgres' in name_lower or 'pg_' in name_lower:
                return SQLDialect.POSTGRESQL
            elif 'oracle' in name_lower:
                return SQLDialect.ORACLE
            elif 'mssql' in name_lower or 'sqlserver' in name_lower:
                return SQLDialect.SQLSERVER
            elif 'snowflake' in name_lower:
                return SQLDialect.SNOWFLAKE
            elif 'bigquery' in name_lower or 'bq_' in name_lower:
                return SQLDialect.BIGQUERY
            elif 'hive' in name_lower:
                return SQLDialect.HIVE
        
        # Check content patterns
        max_score = 0
        detected_dialect = SQLDialect.STANDARD
        
        for dialect, patterns in self.DIALECT_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern.search(content))
            if score > max_score:
                max_score = score
                detected_dialect = dialect
        
        return detected_dialect
    
    def _analyze_statement(self, parsed: Statement, line_offset: int) -> Optional[SQLStatement]:
        """Analyze a parsed SQL statement"""
        content = str(parsed).strip()
        if not content:
            return None
        
        # Determine statement type
        statement_type = self._determine_statement_type(parsed)
        
        # Create statement object
        statement = SQLStatement(
            statement_type=statement_type,
            content=self._format_statement(parsed),
            raw_content=content,
            start_line=line_offset + 1,
            end_line=line_offset + content.count('\n') + 1,
            tables=set(),
            columns=set(),
            aliases={},
            joins=[],
            conditions=[],
            subqueries=[],
            ctes=[],
            parameters=[],
            functions=set(),
            is_nested=False,
            parent=None,
            metadata={}
        )
        
        # Extract statement details
        self._extract_tables(parsed, statement)
        self._extract_columns(parsed, statement)
        self._extract_joins(parsed, statement)
        self._extract_conditions(parsed, statement)
        self._extract_ctes(parsed, statement)
        self._extract_subqueries(parsed, statement)
        self._extract_functions(parsed, statement)
        self._extract_parameters(parsed, statement)
        
        # Update global collections
        self.tables.update(statement.tables)
        
        # Extract schemas
        for table in statement.tables:
            if '.' in table:
                schema = table.split('.')[0]
                self.schemas.add(schema)
        
        # Check for temp tables
        for table in statement.tables:
            if table.startswith('#') or table.startswith('temp_') or 'TEMPORARY' in content.upper():
                self.temp_tables.add(table)
        
        return statement
    
    def _determine_statement_type(self, parsed: Statement) -> SQLStatementType:
        """Determine the type of SQL statement"""
        first_token = parsed.token_first(skip_ws=True, skip_cm=True)
        if not first_token:
            return SQLStatementType.SELECT
        
        ttype = first_token.ttype
        value = first_token.value.upper()
        
        # Check for CTE
        if value == 'WITH':
            return SQLStatementType.WITH_CTE
        
        # DDL statements
        if ttype in DDL or value in ['CREATE', 'ALTER', 'DROP', 'TRUNCATE']:
            if value == 'CREATE':
                next_meaningful = self._get_next_keyword(parsed, first_token)
                if next_meaningful:
                    next_val = next_meaningful.upper()
                    if 'TABLE' in next_val:
                        return SQLStatementType.CREATE_TABLE
                    elif 'INDEX' in next_val:
                        return SQLStatementType.CREATE_INDEX
                    elif 'VIEW' in next_val:
                        return SQLStatementType.CREATE_VIEW
                    elif 'PROCEDURE' in next_val or 'PROC' in next_val:
                        return SQLStatementType.CREATE_PROCEDURE
                    elif 'FUNCTION' in next_val:
                        return SQLStatementType.CREATE_FUNCTION
                    elif 'TRIGGER' in next_val:
                        return SQLStatementType.CREATE_TRIGGER
                    elif 'SCHEMA' in next_val:
                        return SQLStatementType.CREATE_SCHEMA
                    elif 'DATABASE' in next_val:
                        return SQLStatementType.CREATE_DATABASE
                    elif 'SEQUENCE' in next_val:
                        return SQLStatementType.CREATE_SEQUENCE
                    elif 'TYPE' in next_val:
                        return SQLStatementType.CREATE_TYPE
                return SQLStatementType.CREATE_TABLE
            
            elif value == 'ALTER':
                next_meaningful = self._get_next_keyword(parsed, first_token)
                if next_meaningful and 'TABLE' in next_meaningful.upper():
                    return SQLStatementType.ALTER_TABLE
                return SQLStatementType.ALTER_SCHEMA
            
            elif value == 'DROP':
                next_meaningful = self._get_next_keyword(parsed, first_token)
                if next_meaningful:
                    next_val = next_meaningful.upper()
                    if 'TABLE' in next_val:
                        return SQLStatementType.DROP_TABLE
                    elif 'INDEX' in next_val:
                        return SQLStatementType.DROP_INDEX
                    elif 'VIEW' in next_val:
                        return SQLStatementType.DROP_VIEW
                return SQLStatementType.DROP_TABLE
            
            elif value == 'TRUNCATE':
                return SQLStatementType.TRUNCATE
        
        # DML statements
        if ttype in DML or value in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'UPSERT']:
            if value == 'SELECT':
                return SQLStatementType.SELECT
            elif value == 'INSERT':
                return SQLStatementType.INSERT
            elif value == 'UPDATE':
                return SQLStatementType.UPDATE
            elif value == 'DELETE':
                return SQLStatementType.DELETE
            elif value == 'MERGE':
                return SQLStatementType.MERGE
            elif value == 'UPSERT':
                return SQLStatementType.UPSERT
        
        # DCL statements
        if value == 'GRANT':
            return SQLStatementType.GRANT
        elif value == 'REVOKE':
            return SQLStatementType.REVOKE
        
        # TCL statements
        if value == 'COMMIT':
            return SQLStatementType.COMMIT
        elif value == 'ROLLBACK':
            return SQLStatementType.ROLLBACK
        elif value == 'SAVEPOINT':
            return SQLStatementType.SAVEPOINT
        elif value in ['BEGIN', 'START']:
            return SQLStatementType.TRANSACTION
        
        # Other statements
        if value == 'EXPLAIN':
            return SQLStatementType.EXPLAIN
        elif value == 'ANALYZE':
            return SQLStatementType.ANALYZE
        elif value == 'VACUUM':
            return SQLStatementType.VACUUM
        elif value == 'SET':
            return SQLStatementType.SET
        elif value == 'USE':
            return SQLStatementType.USE
        elif value == 'CALL':
            return SQLStatementType.CALL
        elif value == 'EXECUTE' or value == 'EXEC':
            return SQLStatementType.EXECUTE
        elif value == 'DECLARE':
            return SQLStatementType.DECLARE
        elif value == 'IF':
            return SQLStatementType.IF_ELSE
        elif value == 'WHILE':
            return SQLStatementType.WHILE
        elif value == 'FOR':
            return SQLStatementType.FOR
        elif value == 'CURSOR':
            return SQLStatementType.CURSOR
        
        # Default to SELECT
        return SQLStatementType.SELECT
    
    def _get_next_keyword(self, statement: Statement, after_token: Token) -> Optional[str]:
        """Get the next meaningful keyword after a token"""
        found_token = False
        for token in statement.flatten():
            if found_token and token.ttype in (Keyword, DDL, DML):
                return token.value
            if token == after_token:
                found_token = True
        return None
    
    def _format_statement(self, parsed: Statement) -> str:
        """Format SQL statement for readability"""
        return sqlparse.format(
            str(parsed),
            reindent=True,
            keyword_case='upper',
            identifier_case='lower',
            strip_comments=False,
            use_space_around_operators=True
        )
    
    def _extract_tables(self, parsed: Statement, statement: SQLStatement):
        """Extract table references from statement"""
        # Extract FROM clause tables
        from_seen = False
        for token in parsed.flatten():
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
            elif from_seen and token.ttype is Name:
                table_name = self._clean_identifier(token.value)
                statement.tables.add(table_name)
                from_seen = False
        
        # Extract JOIN tables
        for token in parsed.tokens:
            if isinstance(token, TokenList):
                self._extract_tables_from_tokenlist(token, statement)
    
    def _extract_tables_from_tokenlist(self, tokenlist: TokenList, statement: SQLStatement):
        """Recursively extract tables from token list"""
        for token in tokenlist.tokens:
            if token.ttype is Keyword and 'JOIN' in token.value.upper():
                # Next identifier is likely a table
                next_token = tokenlist.token_next(tokenlist.token_index(token))
                if next_token:
                    self._extract_table_from_token(next_token, statement)
            elif isinstance(token, Identifier):
                # Could be a table reference
                table_name = self._get_table_name_from_identifier(token)
                if table_name:
                    statement.tables.add(table_name)
            elif isinstance(token, TokenList):
                self._extract_tables_from_tokenlist(token, statement)
    
    def _extract_table_from_token(self, token: Token, statement: SQLStatement):
        """Extract table name from a token"""
        if isinstance(token, Identifier):
            table_name = self._get_table_name_from_identifier(token)
            if table_name:
                statement.tables.add(table_name)
        elif token.ttype is Name:
            statement.tables.add(self._clean_identifier(token.value))
    
    def _get_table_name_from_identifier(self, identifier: Identifier) -> Optional[str]:
        """Get table name from identifier token"""
        # Get the first name token (table name)
        for token in identifier.tokens:
            if token.ttype is Name or (not token.is_whitespace and not token.is_keyword):
                return self._clean_identifier(str(token))
        return None
    
    def _extract_columns(self, parsed: Statement, statement: SQLStatement):
        """Extract column references from statement"""
        # Extract SELECT columns
        select_seen = False
        for token in parsed.flatten():
            if token.ttype is DML and token.value.upper() == 'SELECT':
                select_seen = True
            elif select_seen and token.ttype is Keyword and token.value.upper() == 'FROM':
                select_seen = False
            elif select_seen and token.ttype is Name:
                column_name = self._clean_identifier(token.value)
                if column_name != '*':
                    statement.columns.add(column_name)
    
    def _extract_joins(self, parsed: Statement, statement: SQLStatement):
        """Extract JOIN information"""
        for token in parsed.tokens:
            if isinstance(token, TokenList):
                self._extract_joins_from_tokenlist(token, statement)
    
    def _extract_joins_from_tokenlist(self, tokenlist: TokenList, statement: SQLStatement):
        """Extract joins from token list"""
        for i, token in enumerate(tokenlist.tokens):
            if token.ttype is Keyword and 'JOIN' in token.value.upper():
                join_info = {
                    'type': self._get_join_type(token.value),
                    'table': None,
                    'condition': None
                }
                
                # Get joined table
                next_idx = i + 1
                while next_idx < len(tokenlist.tokens):
                    next_token = tokenlist.tokens[next_idx]
                    if isinstance(next_token, Identifier) or next_token.ttype is Name:
                        join_info['table'] = self._clean_identifier(str(next_token))
                        break
                    next_idx += 1
                
                # Get join condition (ON clause)
                for j in range(next_idx, len(tokenlist.tokens)):
                    if tokenlist.tokens[j].ttype is Keyword and tokenlist.tokens[j].value.upper() == 'ON':
                        # Collect condition tokens
                        condition_tokens = []
                        for k in range(j + 1, len(tokenlist.tokens)):
                            if tokenlist.tokens[k].ttype is Keyword:
                                break
                            condition_tokens.append(str(tokenlist.tokens[k]))
                        join_info['condition'] = ''.join(condition_tokens).strip()
                        break
                
                statement.joins.append(join_info)
            elif isinstance(token, TokenList):
                self._extract_joins_from_tokenlist(token, statement)
    
    def _get_join_type(self, join_text: str) -> str:
        """Determine join type from text"""
        join_upper = join_text.upper()
        if 'LEFT' in join_upper:
            return 'LEFT JOIN'
        elif 'RIGHT' in join_upper:
            return 'RIGHT JOIN'
        elif 'FULL' in join_upper:
            return 'FULL JOIN'
        elif 'CROSS' in join_upper:
            return 'CROSS JOIN'
        elif 'INNER' in join_upper:
            return 'INNER JOIN'
        else:
            return 'JOIN'
    
    def _extract_conditions(self, parsed: Statement, statement: SQLStatement):
        """Extract WHERE, HAVING conditions"""
        for token in parsed.tokens:
            if isinstance(token, TokenList):
                self._extract_conditions_from_tokenlist(token, statement)
    
    def _extract_conditions_from_tokenlist(self, tokenlist: TokenList, statement: SQLStatement):
        """Extract conditions from token list"""
        condition_keywords = ['WHERE', 'HAVING', 'WHEN']
        
        for i, token in enumerate(tokenlist.tokens):
            if token.ttype is Keyword and token.value.upper() in condition_keywords:
                # Collect condition tokens
                condition_tokens = []
                for j in range(i + 1, len(tokenlist.tokens)):
                    if tokenlist.tokens[j].ttype is Keyword:
                        break
                    condition_tokens.append(str(tokenlist.tokens[j]))
                
                condition = ''.join(condition_tokens).strip()
                if condition:
                    statement.conditions.append(condition)
            elif isinstance(token, TokenList):
                self._extract_conditions_from_tokenlist(token, statement)
    
    def _extract_ctes(self, parsed: Statement, statement: SQLStatement):
        """Extract Common Table Expressions"""
        content = str(parsed)
        
        # Pattern for WITH clause
        with_pattern = re.compile(
            r'WITH\s+(?:RECURSIVE\s+)?(\w+)\s+(?:\([^)]*\))?\s+AS\s*\(',
            re.IGNORECASE | re.MULTILINE
        )
        
        for match in with_pattern.finditer(content):
            cte_name = match.group(1)
            
            # Find the matching closing parenthesis
            start_pos = match.end() - 1
            paren_count = 1
            end_pos = start_pos
            
            for i in range(start_pos + 1, len(content)):
                if content[i] == '(':
                    paren_count += 1
                elif content[i] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        end_pos = i
                        break
            
            cte_content = content[start_pos + 1:end_pos]
            
            statement.ctes.append({
                'name': cte_name,
                'content': cte_content,
                'is_recursive': 'RECURSIVE' in match.group(0).upper()
            })
    
    def _extract_subqueries(self, parsed: Statement, statement: SQLStatement):
        """Extract subqueries"""
        for token in parsed.tokens:
            if isinstance(token, Parenthesis):
                # Check if it's a subquery
                inner_str = str(token).strip()[1:-1]  # Remove parentheses
                if any(keyword in inner_str.upper() for keyword in ['SELECT', 'WITH']):
                    # Parse as subquery
                    sub_parsed = sqlparse.parse(inner_str)
                    if sub_parsed:
                        sub_statement = self._analyze_statement(sub_parsed[0], 0)
                        if sub_statement:
                            sub_statement.is_nested = True
                            sub_statement.parent = statement
                            statement.subqueries.append(sub_statement)
            elif isinstance(token, TokenList):
                self._extract_subqueries_from_tokenlist(token, statement)
    
    def _extract_subqueries_from_tokenlist(self, tokenlist: TokenList, statement: SQLStatement):
        """Extract subqueries from token list"""
        for token in tokenlist.tokens:
            if isinstance(token, Parenthesis):
                inner_str = str(token).strip()[1:-1]
                if any(keyword in inner_str.upper() for keyword in ['SELECT', 'WITH']):
                    sub_parsed = sqlparse.parse(inner_str)
                    if sub_parsed:
                        sub_statement = self._analyze_statement(sub_parsed[0], 0)
                        if sub_statement:
                            sub_statement.is_nested = True
                            sub_statement.parent = statement
                            statement.subqueries.append(sub_statement)
            elif isinstance(token, TokenList):
                self._extract_subqueries_from_tokenlist(token, statement)
    
    def _extract_functions(self, parsed: Statement, statement: SQLStatement):
        """Extract function calls"""
        for token in parsed.flatten():
            if isinstance(token, Function):
                func_name = self._get_function_name(token)
                if func_name:
                    statement.functions.add(func_name.upper())
        
        # Also check for common function patterns
        content = str(parsed)
        for category, funcs in self.SQL_FUNCTIONS.items():
            for func in funcs:
                if re.search(rf'\b{func}\s*\(', content, re.IGNORECASE):
                    statement.functions.add(func)
    
    def _get_function_name(self, function: Function) -> Optional[str]:
        """Get function name from Function token"""
        for token in function.tokens:
            if token.ttype is Name:
                return token.value
        return None
    
    def _extract_parameters(self, parsed: Statement, statement: SQLStatement):
        """Extract parameters/placeholders"""
        content = str(parsed)
        
        # Common parameter patterns
        patterns = [
            re.compile(r'\?'),  # Standard placeholder
            re.compile(r':\w+'),  # Named parameter
            re.compile(r'@\w+'),  # SQL Server style
            re.compile(r'\$\d+'),  # PostgreSQL style
            re.compile(r'%\(\w+\)s'),  # Python style
            re.compile(r'\${?\w+}?'),  # Template style
        ]
        
        for pattern in patterns:
            for match in pattern.finditer(content):
                statement.parameters.append(match.group())
    
    def _clean_identifier(self, identifier: str) -> str:
        """Clean identifier by removing quotes and brackets"""
        # Remove quotes and brackets
        cleaned = identifier.strip()
        for char in ['`', '"', "'", '[', ']']:
            cleaned = cleaned.strip(char)
        return cleaned
    
    def _has_transactions(self, content: str) -> bool:
        """Check if content has transaction statements"""
        transaction_keywords = [
            'BEGIN TRANSACTION', 'START TRANSACTION', 'COMMIT', 
            'ROLLBACK', 'SAVEPOINT', 'BEGIN WORK'
        ]
        content_upper = content.upper()
        return any(keyword in content_upper for keyword in transaction_keywords)
    
    def _count_statement_types(self, keywords: List[str]) -> int:
        """Count statements of certain types"""
        count = 0
        for statement in self.statements:
            statement_name = statement.statement_type.name
            if any(keyword in statement_name for keyword in keywords):
                count += 1
        return count
    
    def _detect_migration_info(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Detect migration framework and version"""
        filename = file_path.name
        
        for framework, pattern in self.MIGRATION_PATTERNS.items():
            match = pattern.match(filename)
            if match:
                if framework == 'flyway':
                    return {
                        'framework': 'flyway',
                        'version': match.group(1).replace('_', '.'),
                        'description': match.group(2)
                    }
                elif framework == 'liquibase':
                    return {
                        'framework': 'liquibase',
                        'author': match.group(1),
                        'id': match.group(2)
                    }
                elif framework in ['rails', 'django']:
                    return {
                        'framework': framework,
                        'timestamp': match.group(1),
                        'description': match.group(2)
                    }
        
        return None
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate SQL statistics"""
        stats = {
            'total_tables': len(self.tables),
            'total_schemas': len(self.schemas),
            'temp_tables': len(self.temp_tables),
            'max_joins': 0,
            'max_subqueries': 0,
            'total_ctes': 0,
            'functions_used': set(),
            'has_window_functions': False,
            'has_recursive_ctes': False,
            'complexity_score': 0
        }
        
        for statement in self.statements:
            # Max joins
            stats['max_joins'] = max(stats['max_joins'], len(statement.joins))
            
            # Max subqueries
            stats['max_subqueries'] = max(stats['max_subqueries'], len(statement.subqueries))
            
            # Total CTEs
            stats['total_ctes'] += len(statement.ctes)
            
            # Check for recursive CTEs
            for cte in statement.ctes:
                if cte.get('is_recursive'):
                    stats['has_recursive_ctes'] = True
            
            # Functions
            stats['functions_used'].update(statement.functions)
            
            # Check for window functions
            for func in statement.functions:
                if func in self.SQL_FUNCTIONS['window']:
                    stats['has_window_functions'] = True
            
            # Calculate complexity
            complexity = (
                len(statement.joins) * 2 +
                len(statement.subqueries) * 3 +
                len(statement.ctes) * 2 +
                len(statement.conditions) +
                (5 if statement.statement_type == SQLStatementType.CREATE_PROCEDURE else 0)
            )
            stats['complexity_score'] += complexity
        
        # Convert set to list for JSON serialization
        stats['functions_used'] = list(stats['functions_used'])
        
        return stats
    
    def _create_error_structure(self, error_msg: str) -> SQLStructure:
        """Create error structure for invalid SQL"""
        return SQLStructure(
            statements=[],
            dialect=SQLDialect.STANDARD,
            total_statements=0,
            ddl_count=0,
            dml_count=0,
            dcl_count=0,
            tcl_count=0,
            has_transactions=False,
            has_procedures=False,
            has_functions=False,
            has_triggers=False,
            has_ctes=False,
            has_temp_tables=False,
            has_views=False,
            schemas=set(),
            tables=set(),
            migration_info=None,
            statistics={'error': error_msg}
        )

class SQLChunker(BaseChunker):
    """Chunker specialized for SQL files"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, ChunkerConfig(max_tokens=max_tokens))
        self.analyzer = SQLAnalyzer()
        
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from SQL file
        
        Args:
            content: SQL content
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Analyze SQL structure
            structure = self.analyzer.analyze_sql(content, file_context.path)
            
            # Determine chunking strategy
            if structure.migration_info:
                return self._chunk_migration_file(content, structure, file_context)
            elif structure.has_procedures or structure.has_functions:
                return self._chunk_procedural_sql(content, structure, file_context)
            elif structure.total_statements == 1:
                return self._chunk_single_statement(content, structure, file_context)
            else:
                return self._chunk_multiple_statements(content, structure, file_context)
                
        except Exception as e:
            logger.error(f"Error chunking SQL file {file_context.path}: {e}")
            return self._fallback_chunking(content, file_context)
    
    def _chunk_migration_file(self, content: str, structure: SQLStructure,
                            file_context: FileContext) -> List[Chunk]:
        """Chunk database migration file"""
        chunks = []
        
        # Add migration metadata chunk
        if structure.migration_info:
            metadata_content = f"-- Migration: {structure.migration_info.get('framework')}\n"
            if 'version' in structure.migration_info:
                metadata_content += f"-- Version: {structure.migration_info['version']}\n"
            if 'description' in structure.migration_info:
                metadata_content += f"-- Description: {structure.migration_info['description']}\n"
            
            chunks.append(self.create_chunk(
                content=metadata_content,
                chunk_type='sql_migration_metadata',
                metadata={
                    'dialect': structure.dialect.value,
                    'framework': structure.migration_info.get('framework'),
                    'version': structure.migration_info.get('version')
                },
                file_path=str(file_context.path)
            ))
        
        # Chunk statements
        for statement in structure.statements:
            stmt_chunks = self._chunk_statement(statement, structure)
            chunks.extend(stmt_chunks)
        
        return chunks
    
    def _chunk_procedural_sql(self, content: str, structure: SQLStructure,
                             file_context: FileContext) -> List[Chunk]:
        """Chunk SQL with procedures/functions"""
        chunks = []
        
        for statement in structure.statements:
            if statement.statement_type in [
                SQLStatementType.CREATE_PROCEDURE,
                SQLStatementType.CREATE_FUNCTION,
                SQLStatementType.CREATE_TRIGGER
            ]:
                # Chunk complex procedural objects
                chunks.extend(self._chunk_complex_object(statement, structure))
            else:
                # Regular statement chunking
                chunks.extend(self._chunk_statement(statement, structure))
        
        return chunks
    
    def _chunk_single_statement(self, content: str, structure: SQLStructure,
                               file_context: FileContext) -> List[Chunk]:
        """Chunk file with single statement"""
        if structure.statements:
            statement = structure.statements[0]
            
            # Check if it fits in one chunk
            if self.count_tokens(statement.content) <= self.max_tokens:
                return [self.create_chunk(
                    content=statement.content,
                    chunk_type=f'sql_{statement.statement_type.value}',
                    metadata={
                        'dialect': structure.dialect.value,
                        'tables': list(statement.tables),
                        'has_ctes': bool(statement.ctes),
                        'has_subqueries': bool(statement.subqueries),
                        'complexity': structure.statistics.get('complexity_score', 0)
                    },
                    file_path=str(file_context.path)
                )]
            else:
                # Split large statement
                return self._split_large_statement(statement, structure, file_context)
        
        return self._fallback_chunking(content, file_context)
    
    def _chunk_multiple_statements(self, content: str, structure: SQLStructure,
                                  file_context: FileContext) -> List[Chunk]:
        """Chunk file with multiple statements"""
        chunks = []
        
        # Group related statements
        statement_groups = self._group_related_statements(structure.statements)
        
        for group in statement_groups:
            if len(group) == 1:
                # Single statement
                chunks.extend(self._chunk_statement(group[0], structure))
            else:
                # Multiple related statements
                chunks.extend(self._chunk_statement_group(group, structure))
        
        return chunks
    
    def _chunk_statement(self, statement: SQLStatement, 
                        structure: SQLStructure) -> List[Chunk]:
        """Chunk a single SQL statement"""
        chunks = []
        
        # Check size
        if self.count_tokens(statement.content) <= self.max_tokens:
            chunks.append(self.create_chunk(
                content=statement.content,
                chunk_type=f'sql_{statement.statement_type.value}',
                metadata={
                    'dialect': structure.dialect.value,
                    'tables': list(statement.tables),
                    'columns': list(statement.columns),
                    'joins': statement.joins,
                    'has_ctes': bool(statement.ctes),
                    'has_subqueries': bool(statement.subqueries),
                    'functions': list(statement.functions),
                    'parameters': statement.parameters
                },
                file_path='query.sql',
                start_line=statement.start_line,
                end_line=statement.end_line
            ))
        else:
            # Split large statement
            chunks.extend(self._split_large_statement(statement, structure, None))
        
        return chunks
    
    def _chunk_complex_object(self, statement: SQLStatement,
                            structure: SQLStructure) -> List[Chunk]:
        """Chunk complex database object (procedure, function, trigger)"""
        chunks = []
        content = statement.content
        
        # Extract object name
        name_match = re.search(
            r'(?:PROCEDURE|FUNCTION|TRIGGER)\s+(?:\w+\.)?"?(\w+)"?',
            content,
            re.IGNORECASE
        )
        object_name = name_match.group(1) if name_match else 'unknown'
        
        # Try to split by logical sections
        sections = self._split_procedural_object(content)
        
        for i, section in enumerate(sections):
            chunks.append(self.create_chunk(
                content=section['content'],
                chunk_type=f'sql_{statement.statement_type.value}_part',
                metadata={
                    'dialect': structure.dialect.value,
                    'object_name': object_name,
                    'section': section['type'],
                    'part': i + 1,
                    'total_parts': len(sections)
                },
                file_path=f'{object_name}.sql'
            ))
        
        return chunks
    
    def _split_procedural_object(self, content: str) -> List[Dict[str, Any]]:
        """Split procedural object into sections"""
        sections = []
        lines = content.split('\n')
        
        # Find major sections
        declaration_start = -1
        begin_index = -1
        exception_index = -1
        end_index = -1
        
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            if 'AS' in line_upper or 'IS' in line_upper:
                declaration_start = i
            elif line_upper.startswith('BEGIN'):
                begin_index = i
            elif line_upper.startswith('EXCEPTION'):
                exception_index = i
            elif line_upper.startswith('END'):
                end_index = i
        
        # Create sections
        if declaration_start >= 0 and begin_index > declaration_start:
            # Header and declarations
            sections.append({
                'type': 'header',
                'content': '\n'.join(lines[:begin_index])
            })
        
        if begin_index >= 0:
            # Body
            body_end = exception_index if exception_index > begin_index else end_index
            if body_end > begin_index:
                body_content = '\n'.join(lines[begin_index:body_end])
                
                # Check if body needs further splitting
                if self.count_tokens(body_content) > self.max_tokens:
                    # Split body into smaller chunks
                    body_chunks = self._split_by_statements(body_content)
                    for chunk in body_chunks:
                        sections.append({
                            'type': 'body',
                            'content': chunk
                        })
                else:
                    sections.append({
                        'type': 'body',
                        'content': body_content
                    })
        
        if exception_index >= 0 and end_index > exception_index:
            # Exception handling
            sections.append({
                'type': 'exception',
                'content': '\n'.join(lines[exception_index:end_index + 1])
            })
        elif end_index >= 0 and not sections:
            # Fallback: entire content
            sections.append({
                'type': 'complete',
                'content': content
            })
        
        return sections if sections else [{'type': 'complete', 'content': content}]
    
    def _split_large_statement(self, statement: SQLStatement, structure: SQLStructure,
                              file_context: Optional[FileContext]) -> List[Chunk]:
        """Split large SQL statement"""
        chunks = []
        
        # Handle CTEs separately
        if statement.ctes:
            # CTE chunk(s)
            for cte in statement.ctes:
                cte_content = f"WITH {cte['name']} AS (\n{cte['content']}\n)"
                chunks.append(self.create_chunk(
                    content=cte_content,
                    chunk_type='sql_cte',
                    metadata={
                        'dialect': structure.dialect.value,
                        'cte_name': cte['name'],
                        'is_recursive': cte.get('is_recursive', False)
                    },
                    file_path='query.sql'
                ))
            
            # Main query without CTEs
            main_content = self._remove_ctes_from_statement(statement.content)
            if self.count_tokens(main_content) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=main_content,
                    chunk_type='sql_main_query',
                    metadata={
                        'dialect': structure.dialect.value,
                        'tables': list(statement.tables),
                        'has_ctes': True
                    },
                    file_path='query.sql'
                ))
            else:
                # Further split main query
                chunks.extend(self._split_select_statement(main_content, statement, structure))
        
        elif statement.statement_type == SQLStatementType.SELECT:
            chunks.extend(self._split_select_statement(statement.content, statement, structure))
        
        elif statement.statement_type in [SQLStatementType.CREATE_TABLE, SQLStatementType.ALTER_TABLE]:
            chunks.extend(self._split_ddl_statement(statement.content, statement, structure))
        
        else:
            # Generic splitting
            chunks.extend(self._split_by_lines(statement.content, statement, structure))
        
        return chunks
    
    def _split_select_statement(self, content: str, statement: SQLStatement,
                               structure: SQLStructure) -> List[Chunk]:
        """Split SELECT statement intelligently"""
        chunks = []
        
        # Try to split by major clauses
        clauses = self._extract_select_clauses(content)
        
        # SELECT clause
        if clauses.get('select'):
            chunks.append(self.create_chunk(
                content=clauses['select'],
                chunk_type='sql_select_clause',
                metadata={
                    'dialect': structure.dialect.value,
                    'columns': list(statement.columns)
                },
                file_path='query.sql'
            ))
        
        # FROM and JOINs
        if clauses.get('from'):
            from_content = clauses['from']
            if clauses.get('joins'):
                from_content += '\n' + clauses['joins']
            
            chunks.append(self.create_chunk(
                content=from_content,
                chunk_type='sql_from_joins',
                metadata={
                    'dialect': structure.dialect.value,
                    'tables': list(statement.tables),
                    'join_count': len(statement.joins)
                },
                file_path='query.sql'
            ))
        
        # WHERE clause
        if clauses.get('where'):
            chunks.append(self.create_chunk(
                content=clauses['where'],
                chunk_type='sql_where_clause',
                metadata={
                    'dialect': structure.dialect.value,
                    'conditions': statement.conditions[:3]  # First 3 conditions
                },
                file_path='query.sql'
            ))
        
        # GROUP BY, HAVING, ORDER BY
        remaining = []
        for clause in ['group_by', 'having', 'order_by', 'limit']:
            if clauses.get(clause):
                remaining.append(clauses[clause])
        
        if remaining:
            chunks.append(self.create_chunk(
                content='\n'.join(remaining),
                chunk_type='sql_aggregation_sorting',
                metadata={
                    'dialect': structure.dialect.value
                },
                file_path='query.sql'
            ))
        
        return chunks
    
    def _extract_select_clauses(self, content: str) -> Dict[str, str]:
        """Extract major clauses from SELECT statement"""
        clauses = {}
        content_upper = content.upper()
        
        # Define clause keywords and their order
        clause_keywords = [
            'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 
            'ORDER BY', 'LIMIT', 'OFFSET', 'UNION', 'INTERSECT', 'EXCEPT'
        ]
        
        # Find positions of each clause
        positions = {}
        for keyword in clause_keywords:
            pos = content_upper.find(keyword)
            if pos != -1:
                positions[keyword] = pos
        
        # Sort by position
        sorted_clauses = sorted(positions.items(), key=lambda x: x[1])
        
        # Extract each clause
        for i, (keyword, pos) in enumerate(sorted_clauses):
            next_pos = sorted_clauses[i + 1][1] if i + 1 < len(sorted_clauses) else len(content)
            clause_content = content[pos:next_pos].strip()
            
            # Map to clause name
            clause_name = keyword.lower().replace(' ', '_')
            clauses[clause_name] = clause_content
        
        # Handle JOINs separately
        join_pattern = re.compile(
            r'((?:INNER|LEFT|RIGHT|FULL|CROSS)\s+)?JOIN\s+[^;]+?(?=\s+(?:WHERE|GROUP|ORDER|LIMIT|$))',
            re.IGNORECASE | re.DOTALL
        )
        
        joins = []
        for match in join_pattern.finditer(content):
            joins.append(match.group(0))
        
        if joins:
            clauses['joins'] = '\n'.join(joins)
        
        return clauses
    
    def _split_ddl_statement(self, content: str, statement: SQLStatement,
                            structure: SQLStructure) -> List[Chunk]:
        """Split DDL statement (CREATE TABLE, etc.)"""
        chunks = []
        
        if statement.statement_type == SQLStatementType.CREATE_TABLE:
            # Split by column definitions
            sections = self._split_create_table(content)
            
            for section in sections:
                chunks.append(self.create_chunk(
                    content=section['content'],
                    chunk_type=f'sql_create_table_{section["type"]}',
                    metadata={
                        'dialect': structure.dialect.value,
                        'table_name': list(statement.tables)[0] if statement.tables else 'unknown',
                        'section': section['type']
                    },
                    file_path='schema.sql'
                ))
        else:
            # Generic DDL splitting
            chunks.extend(self._split_by_lines(content, statement, structure))
        
        return chunks
    
    def _split_create_table(self, content: str) -> List[Dict[str, str]]:
        """Split CREATE TABLE statement"""
        sections = []
        
        # Extract table name and opening
        header_match = re.search(
            r'(CREATE\s+(?:TEMPORARY\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[^(]+\()',
            content,
            re.IGNORECASE
        )
        
        if header_match:
            sections.append({
                'type': 'header',
                'content': header_match.group(1)
            })
            
            # Extract column definitions
            remaining = content[header_match.end():]
            
            # Split by commas (but not within parentheses)
            columns = []
            current = ''
            paren_depth = 0
            
            for char in remaining:
                if char == ',' and paren_depth == 0:
                    columns.append(current.strip())
                    current = ''
                else:
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth -= 1
                        if paren_depth < 0:
                            break
                    current += char
            
            if current.strip():
                columns.append(current.strip())
            
            # Group columns into chunks
            current_chunk = []
            current_tokens = 0
            
            for column in columns:
                column_tokens = self.count_tokens(column)
                
                if current_tokens + column_tokens > self.max_tokens * 0.7 and current_chunk:
                    sections.append({
                        'type': 'columns',
                        'content': ',\n'.join(current_chunk)
                    })
                    current_chunk = [column]
                    current_tokens = column_tokens
                else:
                    current_chunk.append(column)
                    current_tokens += column_tokens
            
            if current_chunk:
                sections.append({
                    'type': 'columns',
                    'content': ',\n'.join(current_chunk)
                })
            
            # Add closing and table options
            closing_match = re.search(r'\)[^;]*;?$', content)
            if closing_match:
                sections.append({
                    'type': 'closing',
                    'content': closing_match.group(0)
                })
        
        return sections if sections else [{'type': 'complete', 'content': content}]
    
    def _split_by_lines(self, content: str, statement: SQLStatement,
                       structure: SQLStructure) -> List[Chunk]:
        """Split content by lines with token limit"""
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
                    chunk_type=f'sql_{statement.statement_type.value}_part',
                    metadata={
                        'dialect': structure.dialect.value,
                        'part': part
                    },
                    file_path='query.sql'
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
                chunk_type=f'sql_{statement.statement_type.value}_part',
                metadata={
                    'dialect': structure.dialect.value,
                    'part': part,
                    'is_last': True
                },
                file_path='query.sql'
            ))
        
        return chunks
    
    def _split_by_statements(self, content: str) -> List[str]:
        """Split content by SQL statements"""
        # Parse statements
        statements = sqlparse.split(content)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for statement in statements:
            statement = statement.strip()
            if not statement:
                continue
            
            statement_tokens = self.count_tokens(statement)
            
            if current_tokens + statement_tokens > self.max_tokens and current_chunk:
                chunks.append(';\n'.join(current_chunk) + ';')
                current_chunk = [statement]
                current_tokens = statement_tokens
            else:
                current_chunk.append(statement)
                current_tokens += statement_tokens
        
        if current_chunk:
            chunks.append(';\n'.join(current_chunk) + ';')
        
        return chunks
    
    def _group_related_statements(self, statements: List[SQLStatement]) -> List[List[SQLStatement]]:
        """Group related SQL statements"""
        groups = []
        current_group = []
        current_tokens = 0
        
        for statement in statements:
            statement_tokens = self.count_tokens(statement.content)
            
            # Check if statements are related
            related = False
            if current_group:
                last_statement = current_group[-1]
                # Check if they operate on same table
                if statement.tables and last_statement.tables:
                    if statement.tables.intersection(last_statement.tables):
                        related = True
                # Check if it's a transaction block
                if (last_statement.statement_type == SQLStatementType.TRANSACTION and
                    statement.statement_type != SQLStatementType.COMMIT):
                    related = True
            
            if related and current_tokens + statement_tokens <= self.max_tokens:
                current_group.append(statement)
                current_tokens += statement_tokens
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [statement]
                current_tokens = statement_tokens
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _chunk_statement_group(self, statements: List[SQLStatement],
                              structure: SQLStructure) -> List[Chunk]:
        """Chunk a group of related statements"""
        chunks = []
        
        # Combine statements
        combined_content = ';\n\n'.join(stmt.content for stmt in statements)
        
        # Extract common metadata
        all_tables = set()
        all_functions = set()
        
        for stmt in statements:
            all_tables.update(stmt.tables)
            all_functions.update(stmt.functions)
        
        chunks.append(self.create_chunk(
            content=combined_content,
            chunk_type='sql_statement_group',
            metadata={
                'dialect': structure.dialect.value,
                'statement_count': len(statements),
                'statement_types': [stmt.statement_type.value for stmt in statements],
                'tables': list(all_tables),
                'functions': list(all_functions)
            },
            file_path='queries.sql'
        ))
        
        return chunks
    
    def _remove_ctes_from_statement(self, content: str) -> str:
        """Remove CTEs from statement content"""
        # Find the main query after CTEs
        # Look for the SELECT after the last CTE
        cte_end = content.upper().rfind(')')
        if cte_end != -1:
            # Find the SELECT after CTEs
            select_pos = content.upper().find('SELECT', cte_end)
            if select_pos != -1:
                return content[select_pos:]
        
        return content
    
    def _fallback_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback chunking for invalid SQL"""
        logger.warning(f"Using fallback chunking for SQL file {file_context.path}")
        
        chunks = []
        
        # Try to split by semicolons (statement boundaries)
        statements = content.split(';')
        
        current_chunk = []
        current_tokens = 0
        chunk_num = 1
        
        for statement in statements:
            statement = statement.strip()
            if not statement:
                continue
            
            statement_with_semicolon = statement + ';'
            statement_tokens = self.count_tokens(statement_with_semicolon)
            
            if current_tokens + statement_tokens > self.max_tokens and current_chunk:
                chunks.append(self.create_chunk(
                    content='\n\n'.join(current_chunk),
                    chunk_type='sql_fallback',
                    metadata={
                        'is_fallback': True,
                        'chunk_number': chunk_num
                    },
                    file_path=str(file_context.path)
                ))
                current_chunk = []
                current_tokens = 0
                chunk_num += 1
            
            current_chunk.append(statement_with_semicolon)
            current_tokens += statement_tokens
        
        # Add remaining
        if current_chunk:
            chunks.append(self.create_chunk(
                content='\n\n'.join(current_chunk),
                chunk_type='sql_fallback',
                metadata={
                    'is_fallback': True,
                    'chunk_number': chunk_num,
                    'is_last': True
                },
                file_path=str(file_context.path)
            ))
        
        return chunks