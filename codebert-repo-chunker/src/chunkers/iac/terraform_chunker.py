"""
Terraform-specific chunker for intelligent semantic chunking
Handles HCL syntax, resources, modules, variables, outputs, providers, and complex configurations
"""

import re
import json
import hcl2
import lark
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from src.utils.logger import get_logger
from enum import Enum

from src.core.base_chunker import BaseChunker, Chunk, ChunkerConfig
from src.core.file_context import FileContext
from config.settings import settings

logger = get_logger(__name__)

class TerraformBlockType(Enum):
    """Types of Terraform blocks"""
    TERRAFORM = "terraform"
    PROVIDER = "provider"
    RESOURCE = "resource"
    DATA = "data"
    MODULE = "module"
    VARIABLE = "variable"
    OUTPUT = "output"
    LOCALS = "locals"
    MOVED = "moved"
    IMPORT = "import"
    
    # Terraform 1.5+ features
    CHECK = "check"
    
    # Provider-specific
    REQUIRED_PROVIDERS = "required_providers"
    BACKEND = "backend"
    
    # Nested blocks
    PROVISIONER = "provisioner"
    CONNECTION = "connection"
    LIFECYCLE = "lifecycle"
    DYNAMIC = "dynamic"
    FOR_EACH = "for_each"
    COUNT = "count"
    DEPENDS_ON = "depends_on"
    
    # Meta-arguments
    PROVIDER_ALIAS = "provider_alias"
    TIMEOUTS = "timeouts"
    
    # Workspaces
    WORKSPACE = "workspace"

class TerraformProvider(Enum):
    """Common Terraform providers"""
    AWS = "aws"
    AZURERM = "azurerm"
    GOOGLE = "google"
    KUBERNETES = "kubernetes"
    HELM = "helm"
    DOCKER = "docker"
    VAULT = "vault"
    CONSUL = "consul"
    NOMAD = "nomad"
    GITHUB = "github"
    GITLAB = "gitlab"
    DATADOG = "datadog"
    NEWRELIC = "newrelic"
    PAGERDUTY = "pagerduty"
    CLOUDFLARE = "cloudflare"
    DIGITALOCEAN = "digitalocean"
    VSPHERE = "vsphere"
    OPENSTACK = "openstack"
    OCI = "oci"
    ALICLOUD = "alicloud"
    RANDOM = "random"
    NULL = "null"
    LOCAL = "local"
    TEMPLATE = "template"
    ARCHIVE = "archive"
    HTTP = "http"
    TLS = "tls"
    TIME = "time"
    EXTERNAL = "external"

@dataclass
class TerraformBlock:
    """Represents a Terraform block"""
    block_type: TerraformBlockType
    block_label: str  # Resource type or block name
    block_name: Optional[str]  # Resource name (second label)
    content: str
    raw_content: str
    start_line: int
    end_line: int
    attributes: Dict[str, Any]
    nested_blocks: List['TerraformBlock']
    references: Set[str]  # References to other resources/data/variables
    dependencies: Set[str]  # Explicit dependencies
    provider: Optional[str]
    provider_alias: Optional[str]
    count: Optional[Any]
    for_each: Optional[Any]
    lifecycle: Optional[Dict[str, Any]]
    provisioners: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TerraformStructure:
    """Represents overall Terraform configuration structure"""
    blocks: List[TerraformBlock]
    providers: Set[str]
    resources: Dict[str, List[str]]  # resource_type -> [resource_names]
    data_sources: Dict[str, List[str]]  # data_type -> [data_names]
    modules: Dict[str, str]  # module_name -> source
    variables: Dict[str, Any]  # variable_name -> default_value
    outputs: Dict[str, Any]  # output_name -> value
    locals: Dict[str, Any]  # local_name -> value
    terraform_version: Optional[str]
    required_providers: Dict[str, Any]
    backend: Optional[Dict[str, Any]]
    workspace: Optional[str]
    has_remote_state: bool
    has_provisioners: bool
    has_null_resources: bool
    has_data_sources: bool
    has_modules: bool
    has_dynamic_blocks: bool
    has_conditional_logic: bool
    complexity_score: int
    statistics: Dict[str, Any]

class TerraformAnalyzer:
    """Analyzes Terraform configuration structure"""
    
    # Common resource patterns by provider
    PROVIDER_PATTERNS = {
        TerraformProvider.AWS: {
            'compute': ['aws_instance', 'aws_launch_template', 'aws_autoscaling_group', 
                       'aws_ecs_cluster', 'aws_ecs_service', 'aws_lambda_function'],
            'networking': ['aws_vpc', 'aws_subnet', 'aws_route_table', 'aws_security_group',
                          'aws_lb', 'aws_lb_target_group', 'aws_route53_zone'],
            'storage': ['aws_s3_bucket', 'aws_ebs_volume', 'aws_efs_file_system'],
            'database': ['aws_db_instance', 'aws_rds_cluster', 'aws_dynamodb_table',
                        'aws_elasticache_cluster', 'aws_redshift_cluster'],
            'iam': ['aws_iam_role', 'aws_iam_policy', 'aws_iam_user', 'aws_iam_group'],
        },
        TerraformProvider.AZURERM: {
            'compute': ['azurerm_virtual_machine', 'azurerm_linux_virtual_machine',
                       'azurerm_windows_virtual_machine', 'azurerm_kubernetes_cluster'],
            'networking': ['azurerm_virtual_network', 'azurerm_subnet', 'azurerm_network_security_group',
                          'azurerm_lb', 'azurerm_application_gateway'],
            'storage': ['azurerm_storage_account', 'azurerm_storage_container'],
            'database': ['azurerm_mssql_database', 'azurerm_postgresql_server',
                        'azurerm_cosmosdb_account'],
        },
        TerraformProvider.GOOGLE: {
            'compute': ['google_compute_instance', 'google_container_cluster',
                       'google_cloud_run_service', 'google_cloudfunctions_function'],
            'networking': ['google_compute_network', 'google_compute_subnetwork',
                          'google_compute_firewall', 'google_compute_forwarding_rule'],
            'storage': ['google_storage_bucket', 'google_compute_disk'],
            'database': ['google_sql_database_instance', 'google_spanner_instance',
                        'google_bigtable_instance'],
        },
        TerraformProvider.KUBERNETES: {
            'workloads': ['kubernetes_deployment', 'kubernetes_stateful_set', 'kubernetes_daemon_set',
                         'kubernetes_job', 'kubernetes_cron_job'],
            'services': ['kubernetes_service', 'kubernetes_ingress', 'kubernetes_ingress_v1'],
            'config': ['kubernetes_config_map', 'kubernetes_secret'],
            'storage': ['kubernetes_persistent_volume', 'kubernetes_persistent_volume_claim'],
            'rbac': ['kubernetes_role', 'kubernetes_role_binding', 'kubernetes_cluster_role'],
        }
    }
    
    # HCL built-in functions
    HCL_FUNCTIONS = {
        'numeric': ['abs', 'ceil', 'floor', 'log', 'max', 'min', 'parseint', 'pow', 'signum'],
        'string': ['chomp', 'format', 'formatlist', 'indent', 'join', 'lower', 'regex',
                  'regexall', 'replace', 'split', 'strrev', 'substr', 'title', 'trim',
                  'trimprefix', 'trimsuffix', 'trimspace', 'upper'],
        'collection': ['alltrue', 'anytrue', 'chunklist', 'coalesce', 'coalescelist',
                      'compact', 'concat', 'contains', 'distinct', 'element', 'flatten',
                      'index', 'keys', 'length', 'list', 'lookup', 'map', 'matchkeys',
                      'merge', 'one', 'range', 'reverse', 'setintersection', 'setproduct',
                      'setsubtract', 'setunion', 'slice', 'sort', 'sum', 'transpose',
                      'values', 'zipmap'],
        'encoding': ['base64decode', 'base64encode', 'base64gzip', 'csvdecode', 'jsondecode',
                    'jsonencode', 'textdecodebase64', 'textencodebase64', 'urlencode',
                    'yamldecode', 'yamlencode'],
        'filesystem': ['abspath', 'dirname', 'pathexpand', 'basename', 'file', 'fileexists',
                      'fileset', 'filebase64', 'templatefile'],
        'date': ['formatdate', 'timeadd', 'timecmp', 'timestamp'],
        'hash': ['base64sha256', 'base64sha512', 'bcrypt', 'filebase64sha256',
                'filebase64sha512', 'filemd5', 'filesha1', 'filesha256', 'filesha512',
                'md5', 'rsadecrypt', 'sha1', 'sha256', 'sha512', 'uuid', 'uuidv5'],
        'ip': ['cidrhost', 'cidrnetmask', 'cidrsubnet', 'cidrsubnets'],
        'type': ['can', 'defaults', 'nonsensitive', 'sensitive', 'tobool', 'tolist',
                'tomap', 'tonumber', 'toset', 'tostring', 'try', 'type'],
    }
    
    # Reference patterns
    REFERENCE_PATTERNS = {
        'resource': re.compile(r'\b([a-z0-9_]+)\.([a-z0-9_]+)\.([a-z0-9_\[\]\.]+)'),
        'data': re.compile(r'\bdata\.([a-z0-9_]+)\.([a-z0-9_]+)\.([a-z0-9_\[\]\.]+)'),
        'variable': re.compile(r'\bvar\.([a-z0-9_]+)'),
        'local': re.compile(r'\blocal\.([a-z0-9_]+)'),
        'module': re.compile(r'\bmodule\.([a-z0-9_]+)\.([a-z0-9_\[\]\.]+)'),
        'each': re.compile(r'\beach\.(key|value)'),
        'count': re.compile(r'\bcount\.index'),
        'path': re.compile(r'\bpath\.(module|root|cwd)'),
        'terraform': re.compile(r'\bterraform\.workspace'),
    }
    
    def __init__(self):
        self.blocks = []
        self.providers = set()
        self.resources = defaultdict(list)
        self.data_sources = defaultdict(list)
        self.modules = {}
        self.variables = {}
        self.outputs = {}
        self.locals = {}
        
    def analyze_terraform(self, content: str, file_path: Optional[Path] = None) -> TerraformStructure:
        """
        Analyze Terraform configuration
        
        Args:
            content: Terraform HCL content
            file_path: Optional file path for context
            
        Returns:
            TerraformStructure analysis
        """
        try:
            # Parse HCL content
            parsed = self._parse_hcl(content)
            
            # Extract blocks
            self._extract_blocks(parsed, content)
            
            # Analyze dependencies
            self._analyze_dependencies()
            
            # Extract configuration details
            terraform_config = self._extract_terraform_config(parsed)
            
            # Calculate complexity
            complexity = self._calculate_complexity()
            
            # Create structure
            structure = TerraformStructure(
                blocks=self.blocks,
                providers=self.providers,
                resources=dict(self.resources),
                data_sources=dict(self.data_sources),
                modules=self.modules,
                variables=self.variables,
                outputs=self.outputs,
                locals=self.locals,
                terraform_version=terraform_config.get('required_version'),
                required_providers=terraform_config.get('required_providers', {}),
                backend=terraform_config.get('backend'),
                workspace=terraform_config.get('workspace'),
                has_remote_state=bool(terraform_config.get('backend')),
                has_provisioners=any(b.provisioners for b in self.blocks),
                has_null_resources='null_resource' in self.resources,
                has_data_sources=bool(self.data_sources),
                has_modules=bool(self.modules),
                has_dynamic_blocks=any('dynamic' in b.raw_content.lower() for b in self.blocks),
                has_conditional_logic=self._has_conditional_logic(content),
                complexity_score=complexity,
                statistics=self._calculate_statistics()
            )
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing Terraform: {e}")
            return self._create_error_structure(str(e))
    
    def _parse_hcl(self, content: str) -> Dict[str, Any]:
        """Parse HCL content using python-hcl2"""
        try:
            # Use hcl2 library for parsing
            parsed = hcl2.loads(content)
            return parsed
        except lark.exceptions.LarkError as e:
            # Try to extract what we can even if parsing fails
            logger.warning(f"HCL parsing error, using fallback: {e}")
            return self._fallback_parse(content)
    
    def _fallback_parse(self, content: str) -> Dict[str, Any]:
        """Fallback parsing when HCL2 fails"""
        result = {}
        
        # Use regex to extract blocks
        block_pattern = re.compile(
            r'^(terraform|provider|resource|data|module|variable|output|locals)\s+'
            r'(?:"([^"]+)"\s+)?(?:"([^"]+)"\s+)?\{',
            re.MULTILINE
        )
        
        for match in block_pattern.finditer(content):
            block_type = match.group(1)
            block_label = match.group(2)
            block_name = match.group(3)
            
            if block_type not in result:
                result[block_type] = []
            
            # Extract block content
            start_pos = match.start()
            block_content = self._extract_block_content(content[start_pos:])
            
            block_data = {
                'type': block_type,
                'label': block_label,
                'name': block_name,
                'content': block_content
            }
            
            result[block_type].append(block_data)
        
        return result
    
    def _extract_block_content(self, content: str) -> str:
        """Extract content of a block until closing brace"""
        brace_count = 0
        in_block = False
        result = []
        
        for char in content:
            result.append(char)
            if char == '{':
                brace_count += 1
                in_block = True
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and in_block:
                    break
        
        return ''.join(result)
    
    def _extract_blocks(self, parsed: Dict[str, Any], raw_content: str):
        """Extract all blocks from parsed HCL"""
        lines = raw_content.split('\n')
        
        # Extract terraform blocks
        if 'terraform' in parsed:
            for terraform_block in parsed['terraform']:
                block = self._create_block(
                    TerraformBlockType.TERRAFORM,
                    'terraform',
                    None,
                    terraform_block,
                    lines
                )
                self.blocks.append(block)
        
        # Extract provider blocks
        if 'provider' in parsed:
            for provider_data in parsed['provider']:
                if isinstance(provider_data, dict):
                    for provider_name, provider_config in provider_data.items():
                        self.providers.add(provider_name)
                        block = self._create_block(
                            TerraformBlockType.PROVIDER,
                            provider_name,
                            None,
                            provider_config,
                            lines
                        )
                        self.blocks.append(block)
        
        # Extract resource blocks
        if 'resource' in parsed:
            for resource_data in parsed['resource']:
                if isinstance(resource_data, dict):
                    for resource_type, resources in resource_data.items():
                        if isinstance(resources, dict):
                            for resource_name, resource_config in resources.items():
                                self.resources[resource_type].append(resource_name)
                                block = self._create_block(
                                    TerraformBlockType.RESOURCE,
                                    resource_type,
                                    resource_name,
                                    resource_config,
                                    lines
                                )
                                self._extract_provider_from_resource(block, resource_config)
                                self.blocks.append(block)
        
        # Extract data blocks
        if 'data' in parsed:
            for data_item in parsed['data']:
                if isinstance(data_item, dict):
                    for data_type, data_sources in data_item.items():
                        if isinstance(data_sources, dict):
                            for data_name, data_config in data_sources.items():
                                self.data_sources[data_type].append(data_name)
                                block = self._create_block(
                                    TerraformBlockType.DATA,
                                    data_type,
                                    data_name,
                                    data_config,
                                    lines
                                )
                                self.blocks.append(block)
        
        # Extract module blocks
        if 'module' in parsed:
            for module_data in parsed['module']:
                if isinstance(module_data, dict):
                    for module_name, module_config in module_data.items():
                        source = module_config.get('source', 'unknown')
                        self.modules[module_name] = source
                        block = self._create_block(
                            TerraformBlockType.MODULE,
                            'module',
                            module_name,
                            module_config,
                            lines
                        )
                        self.blocks.append(block)
        
        # Extract variable blocks
        if 'variable' in parsed:
            for var_data in parsed['variable']:
                if isinstance(var_data, dict):
                    for var_name, var_config in var_data.items():
                        default_value = var_config.get('default')
                        self.variables[var_name] = default_value
                        block = self._create_block(
                            TerraformBlockType.VARIABLE,
                            'variable',
                            var_name,
                            var_config,
                            lines
                        )
                        self.blocks.append(block)
        
        # Extract output blocks
        if 'output' in parsed:
            for output_data in parsed['output']:
                if isinstance(output_data, dict):
                    for output_name, output_config in output_data.items():
                        value = output_config.get('value')
                        self.outputs[output_name] = value
                        block = self._create_block(
                            TerraformBlockType.OUTPUT,
                            'output',
                            output_name,
                            output_config,
                            lines
                        )
                        self.blocks.append(block)
        
        # Extract locals blocks
        if 'locals' in parsed:
            for locals_data in parsed['locals']:
                if isinstance(locals_data, dict):
                    self.locals.update(locals_data)
                    block = self._create_block(
                        TerraformBlockType.LOCALS,
                        'locals',
                        None,
                        locals_data,
                        lines
                    )
                    self.blocks.append(block)
    
    def _create_block(self, block_type: TerraformBlockType, block_label: str,
                     block_name: Optional[str], config: Dict[str, Any],
                     lines: List[str]) -> TerraformBlock:
        """Create a TerraformBlock from parsed data"""
        # Generate content representation
        if block_name:
            content = f'{block_type.value} "{block_label}" "{block_name}" {{\n'
        elif block_label != block_type.value:
            content = f'{block_type.value} "{block_label}" {{\n'
        else:
            content = f'{block_type.value} {{\n'
        
        content += self._format_block_content(config, indent=2)
        content += '}'
        
        # Extract references
        references = self._extract_references(str(config))
        
        # Extract nested blocks and attributes
        nested_blocks = []
        provisioners = []
        lifecycle = None
        count = None
        for_each = None
        
        if isinstance(config, dict):
            # Extract special blocks
            if 'lifecycle' in config:
                lifecycle = config['lifecycle']
            if 'count' in config:
                count = config['count']
            if 'for_each' in config:
                for_each = config['for_each']
            if 'provisioner' in config:
                provisioners = config['provisioner'] if isinstance(config['provisioner'], list) else [config['provisioner']]
            
            # Extract nested dynamic blocks
            for key, value in config.items():
                if key == 'dynamic' and isinstance(value, dict):
                    for dynamic_name, dynamic_config in value.items():
                        nested_block = TerraformBlock(
                            block_type=TerraformBlockType.DYNAMIC,
                            block_label=dynamic_name,
                            block_name=None,
                            content=str(dynamic_config),
                            raw_content=str(dynamic_config),
                            start_line=0,
                            end_line=0,
                            attributes={},
                            nested_blocks=[],
                            references=set(),
                            dependencies=set(),
                            provider=None,
                            provider_alias=None,
                            count=None,
                            for_each=None,
                            lifecycle=None,
                            provisioners=[],
                            metadata={}
                        )
                        nested_blocks.append(nested_block)
        
        # Extract dependencies
        dependencies = set()
        if isinstance(config, dict) and 'depends_on' in config:
            deps = config['depends_on']
            if isinstance(deps, list):
                dependencies.update(deps)
        
        block = TerraformBlock(
            block_type=block_type,
            block_label=block_label,
            block_name=block_name,
            content=content,
            raw_content=str(config),
            start_line=0,  # Would need to calculate from actual file
            end_line=0,
            attributes=config if isinstance(config, dict) else {},
            nested_blocks=nested_blocks,
            references=references,
            dependencies=dependencies,
            provider=None,
            provider_alias=None,
            count=count,
            for_each=for_each,
            lifecycle=lifecycle,
            provisioners=provisioners,
            metadata={}
        )
        
        return block
    
    def _format_block_content(self, config: Any, indent: int = 0) -> str:
        """Format block content with proper indentation"""
        result = []
        indent_str = ' ' * indent
        
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict):
                    result.append(f'{indent_str}{key} {{')
                    result.append(self._format_block_content(value, indent + 2))
                    result.append(f'{indent_str}}}')
                elif isinstance(value, list):
                    if all(isinstance(item, str) for item in value):
                        # Simple list
                        result.append(f'{indent_str}{key} = {json.dumps(value)}')
                    else:
                        # Complex list
                        for item in value:
                            if isinstance(item, dict):
                                result.append(f'{indent_str}{key} {{')
                                result.append(self._format_block_content(item, indent + 2))
                                result.append(f'{indent_str}}}')
                            else:
                                result.append(f'{indent_str}{key} = {json.dumps(item)}')
                else:
                    # Simple value
                    if isinstance(value, str):
                        result.append(f'{indent_str}{key} = "{value}"')
                    else:
                        result.append(f'{indent_str}{key} = {value}')
        else:
            result.append(f'{indent_str}{config}')
        
        return '\n'.join(result) + '\n' if result else ''
    
    def _extract_provider_from_resource(self, block: TerraformBlock, config: Dict[str, Any]):
        """Extract provider information from resource"""
        # Check resource type prefix
        resource_type = block.block_label
        
        # Common provider prefixes
        for provider in TerraformProvider:
            if resource_type.startswith(provider.value + '_'):
                block.provider = provider.value
                self.providers.add(provider.value)
                break
        
        # Check explicit provider attribute
        if isinstance(config, dict) and 'provider' in config:
            provider_ref = config['provider']
            if isinstance(provider_ref, str):
                # Handle provider alias
                if '.' in provider_ref:
                    provider, alias = provider_ref.split('.', 1)
                    block.provider = provider
                    block.provider_alias = alias
                else:
                    block.provider = provider_ref
                self.providers.add(block.provider)
    
    def _extract_references(self, content: str) -> Set[str]:
        """Extract references to other resources/variables"""
        references = set()
        
        for ref_type, pattern in self.REFERENCE_PATTERNS.items():
            for match in pattern.finditer(content):
                if ref_type in ['resource', 'data']:
                    # Full reference: type.name.attribute
                    references.add(match.group(0))
                elif ref_type in ['variable', 'local']:
                    # Variable or local reference
                    references.add(match.group(0))
                elif ref_type == 'module':
                    # Module output reference
                    references.add(match.group(0))
                else:
                    # Other references
                    references.add(match.group(0))
        
        return references
    
    def _analyze_dependencies(self):
        """Analyze dependencies between blocks"""
        # Build dependency graph
        for block in self.blocks:
            # Check references for dependencies
            for ref in block.references:
                # Parse reference
                if ref.startswith('var.'):
                    # Variable reference
                    var_name = ref[4:].split('.')[0]
                    block.dependencies.add(f'variable.{var_name}')
                elif ref.startswith('local.'):
                    # Local reference
                    local_name = ref[6:].split('.')[0]
                    block.dependencies.add(f'locals.{local_name}')
                elif ref.startswith('data.'):
                    # Data source reference
                    parts = ref[5:].split('.')
                    if len(parts) >= 2:
                        block.dependencies.add(f'data.{parts[0]}.{parts[1]}')
                elif ref.startswith('module.'):
                    # Module reference
                    parts = ref[7:].split('.')
                    if parts:
                        block.dependencies.add(f'module.{parts[0]}')
                elif '.' in ref:
                    # Resource reference
                    parts = ref.split('.')
                    if len(parts) >= 2:
                        block.dependencies.add(f'{parts[0]}.{parts[1]}')
    
    def _extract_terraform_config(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Extract terraform configuration block details"""
        config = {}
        
        if 'terraform' in parsed:
            for terraform_block in parsed['terraform']:
                if isinstance(terraform_block, dict):
                    # Required version
                    if 'required_version' in terraform_block:
                        config['required_version'] = terraform_block['required_version']
                    
                    # Required providers
                    if 'required_providers' in terraform_block:
                        config['required_providers'] = terraform_block['required_providers']
                    
                    # Backend configuration
                    if 'backend' in terraform_block:
                        backend = terraform_block['backend']
                        if isinstance(backend, dict):
                            for backend_type, backend_config in backend.items():
                                config['backend'] = {
                                    'type': backend_type,
                                    'config': backend_config
                                }
                                break
                    
                    # Cloud configuration (Terraform Cloud)
                    if 'cloud' in terraform_block:
                        config['cloud'] = terraform_block['cloud']
        
        return config
    
    def _has_conditional_logic(self, content: str) -> bool:
        """Check if configuration has conditional logic"""
        conditional_patterns = [
            r'\bcount\s*=',
            r'\bfor_each\s*=',
            r'\?\s*.*\s*:',  # Ternary operator
            r'\bif\s*\(',     # Conditional expression
            r'\bfor\s+\w+\s+in\s+',  # For expression
            r'dynamic\s+"',    # Dynamic blocks
        ]
        
        for pattern in conditional_patterns:
            if re.search(pattern, content):
                return True
        
        return False
    
    def _calculate_complexity(self) -> int:
        """Calculate complexity score for Terraform configuration"""
        complexity = 0
        
        # Base complexity from number of resources
        complexity += len(self.blocks) * 2
        
        # Additional complexity factors
        for block in self.blocks:
            # Dynamic blocks and loops
            if block.count or block.for_each:
                complexity += 5
            
            # Dependencies
            complexity += len(block.dependencies)
            
            # Provisioners
            complexity += len(block.provisioners) * 3
            
            # Nested blocks
            complexity += len(block.nested_blocks) * 2
            
            # Complex references
            complexity += len(block.references)
        
        # Module usage adds complexity
        complexity += len(self.modules) * 5
        
        # Data sources
        complexity += sum(len(sources) for sources in self.data_sources.values())
        
        return complexity
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about Terraform configuration"""
        stats = {
            'total_blocks': len(self.blocks),
            'resource_count': sum(len(names) for names in self.resources.values()),
            'data_source_count': sum(len(names) for names in self.data_sources.values()),
            'module_count': len(self.modules),
            'variable_count': len(self.variables),
            'output_count': len(self.outputs),
            'local_count': len(self.locals),
            'provider_count': len(self.providers),
            'unique_resource_types': len(self.resources),
            'blocks_with_count': sum(1 for b in self.blocks if b.count),
            'blocks_with_for_each': sum(1 for b in self.blocks if b.for_each),
            'blocks_with_lifecycle': sum(1 for b in self.blocks if b.lifecycle),
            'blocks_with_provisioners': sum(1 for b in self.blocks if b.provisioners),
            'total_dependencies': sum(len(b.dependencies) for b in self.blocks),
            'total_references': sum(len(b.references) for b in self.blocks),
        }
        
        # Resource type distribution
        resource_distribution = defaultdict(int)
        for resource_type, names in self.resources.items():
            category = self._categorize_resource(resource_type)
            resource_distribution[category] += len(names)
        
        stats['resource_distribution'] = dict(resource_distribution)
        
        # Provider distribution
        provider_distribution = defaultdict(int)
        for block in self.blocks:
            if block.provider:
                provider_distribution[block.provider] += 1
        
        stats['provider_distribution'] = dict(provider_distribution)
        
        return stats
    
    def _categorize_resource(self, resource_type: str) -> str:
        """Categorize resource type"""
        # Check each provider's patterns
        for provider, categories in self.PROVIDER_PATTERNS.items():
            for category, resources in categories.items():
                if resource_type in resources:
                    return category
        
        # Generic categorization based on common patterns
        if 'instance' in resource_type or 'vm' in resource_type:
            return 'compute'
        elif 'network' in resource_type or 'subnet' in resource_type or 'vpc' in resource_type:
            return 'networking'
        elif 'storage' in resource_type or 'bucket' in resource_type or 'disk' in resource_type:
            return 'storage'
        elif 'database' in resource_type or 'db' in resource_type or 'rds' in resource_type:
            return 'database'
        elif 'iam' in resource_type or 'role' in resource_type or 'policy' in resource_type:
            return 'iam'
        elif 'security' in resource_type or 'firewall' in resource_type:
            return 'security'
        else:
            return 'other'
    
    def _create_error_structure(self, error_msg: str) -> TerraformStructure:
        """Create error structure for invalid Terraform"""
        return TerraformStructure(
            blocks=[],
            providers=set(),
            resources={},
            data_sources={},
            modules={},
            variables={},
            outputs={},
            locals={},
            terraform_version=None,
            required_providers={},
            backend=None,
            workspace=None,
            has_remote_state=False,
            has_provisioners=False,
            has_null_resources=False,
            has_data_sources=False,
            has_modules=False,
            has_dynamic_blocks=False,
            has_conditional_logic=False,
            complexity_score=0,
            statistics={'error': error_msg}
        )

class TerraformChunker(BaseChunker):
    """Chunker specialized for Terraform configuration files"""
    
    def __init__(self, tokenizer, max_tokens: int = 450):
        super().__init__(tokenizer, ChunkerConfig(max_tokens=max_tokens))
        self.analyzer = TerraformAnalyzer()
        
    def chunk(self, content: str, file_context: FileContext) -> List[Chunk]:
        """
        Create chunks from Terraform file
        
        Args:
            content: Terraform HCL content
            file_context: File context
            
        Returns:
            List of chunks
        """
        try:
            # Analyze Terraform structure
            structure = self.analyzer.analyze_terraform(content, file_context.path)
            
            # Determine chunking strategy
            if structure.complexity_score > 100:
                # Complex configuration - chunk by logical groups
                return self._chunk_complex_config(content, structure, file_context)
            elif structure.has_modules:
                # Module-based configuration
                return self._chunk_modular_config(content, structure, file_context)
            elif len(structure.blocks) == 1:
                # Single block
                return self._chunk_single_block(content, structure, file_context)
            else:
                # Standard multi-block configuration
                return self._chunk_standard_config(content, structure, file_context)
                
        except Exception as e:
            logger.error(f"Error chunking Terraform file {file_context.path}: {e}")
            return self._fallback_chunking(content, file_context)
    
    def _chunk_complex_config(self, content: str, structure: TerraformStructure,
                            file_context: FileContext) -> List[Chunk]:
        """Chunk complex Terraform configuration"""
        chunks = []
        
        # Group blocks by logical categories
        block_groups = self._group_blocks_by_category(structure.blocks)
        
        # Create chunk for terraform configuration
        terraform_blocks = [b for b in structure.blocks if b.block_type == TerraformBlockType.TERRAFORM]
        if terraform_blocks:
            chunks.append(self._create_terraform_config_chunk(terraform_blocks, structure))
        
        # Create chunk for providers
        provider_blocks = [b for b in structure.blocks if b.block_type == TerraformBlockType.PROVIDER]
        if provider_blocks:
            chunks.append(self._create_providers_chunk(provider_blocks, structure))
        
        # Create chunks for each resource category
        for category, blocks in block_groups.items():
            if category not in ['terraform', 'provider']:
                category_chunks = self._chunk_block_category(blocks, category, structure)
                chunks.extend(category_chunks)
        
        # Create chunk for variables
        variable_blocks = [b for b in structure.blocks if b.block_type == TerraformBlockType.VARIABLE]
        if variable_blocks:
            chunks.extend(self._chunk_variables(variable_blocks, structure))
        
        # Create chunk for outputs
        output_blocks = [b for b in structure.blocks if b.block_type == TerraformBlockType.OUTPUT]
        if output_blocks:
            chunks.extend(self._chunk_outputs(output_blocks, structure))
        
        return chunks
    
    def _chunk_modular_config(self, content: str, structure: TerraformStructure,
                            file_context: FileContext) -> List[Chunk]:
        """Chunk module-based Terraform configuration"""
        chunks = []
        
        # Create chunk for root module configuration
        root_config = self._extract_root_config(structure)
        if root_config:
            chunks.append(self.create_chunk(
                content=root_config,
                chunk_type='terraform_root_config',
                metadata={
                    'providers': list(structure.providers),
                    'terraform_version': structure.terraform_version,
                    'backend': structure.backend
                },
                file_path=str(file_context.path)
            ))
        
        # Create chunks for each module
        module_blocks = [b for b in structure.blocks if b.block_type == TerraformBlockType.MODULE]
        for module_block in module_blocks:
            chunks.append(self._create_module_chunk(module_block, structure))
        
        # Create chunks for resources not in modules
        resource_blocks = [b for b in structure.blocks if b.block_type == TerraformBlockType.RESOURCE]
        if resource_blocks:
            chunks.extend(self._chunk_resources(resource_blocks, structure))
        
        return chunks
    
    def _chunk_single_block(self, content: str, structure: TerraformStructure,
                          file_context: FileContext) -> List[Chunk]:
        """Chunk single Terraform block"""
        if structure.blocks:
            block = structure.blocks[0]
            
            # Check if it fits in one chunk
            if self.count_tokens(block.content) <= self.max_tokens:
                return [self.create_chunk(
                    content=block.content,
                    chunk_type=f'terraform_{block.block_type.value}',
                    metadata={
                        'block_type': block.block_type.value,
                        'block_label': block.block_label,
                        'block_name': block.block_name,
                        'provider': block.provider,
                        'dependencies': list(block.dependencies),
                        'references': list(block.references)
                    },
                    file_path=str(file_context.path)
                )]
            else:
                # Split large block
                return self._split_large_block(block, structure, file_context)
        
        return self._fallback_chunking(content, file_context)
    
    def _chunk_standard_config(self, content: str, structure: TerraformStructure,
                              file_context: FileContext) -> List[Chunk]:
        """Chunk standard Terraform configuration"""
        chunks = []
        
        # Group related blocks
        block_groups = self._group_related_blocks(structure.blocks)
        
        for group in block_groups:
            if len(group) == 1:
                # Single block
                chunks.extend(self._chunk_block(group[0], structure))
            else:
                # Multiple related blocks
                chunks.extend(self._chunk_block_group(group, structure))
        
        return chunks
    
    def _group_blocks_by_category(self, blocks: List[TerraformBlock]) -> Dict[str, List[TerraformBlock]]:
        """Group blocks by category"""
        categories = defaultdict(list)
        
        for block in blocks:
            if block.block_type == TerraformBlockType.TERRAFORM:
                categories['terraform'].append(block)
            elif block.block_type == TerraformBlockType.PROVIDER:
                categories['provider'].append(block)
            elif block.block_type == TerraformBlockType.RESOURCE:
                # Categorize by resource type
                category = self.analyzer._categorize_resource(block.block_label)
                categories[f'resource_{category}'].append(block)
            elif block.block_type == TerraformBlockType.DATA:
                categories['data'].append(block)
            elif block.block_type == TerraformBlockType.MODULE:
                categories['module'].append(block)
            elif block.block_type == TerraformBlockType.VARIABLE:
                categories['variable'].append(block)
            elif block.block_type == TerraformBlockType.OUTPUT:
                categories['output'].append(block)
            elif block.block_type == TerraformBlockType.LOCALS:
                categories['locals'].append(block)
            else:
                categories['other'].append(block)
        
        return dict(categories)
    
    def _create_terraform_config_chunk(self, blocks: List[TerraformBlock],
                                      structure: TerraformStructure) -> Chunk:
        """Create chunk for terraform configuration"""
        content_parts = []
        
        for block in blocks:
            content_parts.append(block.content)
        
        return self.create_chunk(
            content='\n\n'.join(content_parts),
            chunk_type='terraform_config',
            metadata={
                'terraform_version': structure.terraform_version,
                'required_providers': list(structure.required_providers.keys()),
                'backend': structure.backend.get('type') if structure.backend else None,
                'has_remote_state': structure.has_remote_state
            },
            file_path='terraform.tf'
        )
    
    def _create_providers_chunk(self, blocks: List[TerraformBlock],
                               structure: TerraformStructure) -> Chunk:
        """Create chunk for providers"""
        content_parts = []
        
        for block in blocks:
            content_parts.append(block.content)
        
        return self.create_chunk(
            content='\n\n'.join(content_parts),
            chunk_type='terraform_providers',
            metadata={
                'providers': list(structure.providers),
                'provider_count': len(blocks)
            },
            file_path='providers.tf'
        )
    
    def _chunk_block_category(self, blocks: List[TerraformBlock], category: str,
                            structure: TerraformStructure) -> List[Chunk]:
        """Chunk blocks in a category"""
        chunks = []
        
        # Group blocks that fit together
        current_group = []
        current_tokens = 0
        
        for block in blocks:
            block_tokens = self.count_tokens(block.content)
            
            if current_tokens + block_tokens > self.max_tokens and current_group:
                # Create chunk for current group
                chunks.append(self._create_category_chunk(current_group, category, structure))
                current_group = [block]
                current_tokens = block_tokens
            else:
                current_group.append(block)
                current_tokens += block_tokens
        
        # Add remaining blocks
        if current_group:
            chunks.append(self._create_category_chunk(current_group, category, structure))
        
        return chunks
    
    def _create_category_chunk(self, blocks: List[TerraformBlock], category: str,
                              structure: TerraformStructure) -> Chunk:
        """Create chunk for a category of blocks"""
        content_parts = []
        block_info = []
        
        for block in blocks:
            content_parts.append(block.content)
            block_info.append({
                'type': block.block_type.value,
                'label': block.block_label,
                'name': block.block_name
            })
        
        return self.create_chunk(
            content='\n\n'.join(content_parts),
            chunk_type=f'terraform_{category}',
            metadata={
                'category': category,
                'block_count': len(blocks),
                'blocks': block_info
            },
            file_path=f'{category}.tf'
        )
    
    def _chunk_variables(self, blocks: List[TerraformBlock],
                        structure: TerraformStructure) -> List[Chunk]:
        """Chunk variable blocks"""
        chunks = []
        
        # Group variables by purpose/prefix
        variable_groups = defaultdict(list)
        
        for block in blocks:
            # Try to group by prefix
            var_name = block.block_name
            if '_' in var_name:
                prefix = var_name.split('_')[0]
                variable_groups[prefix].append(block)
            else:
                variable_groups['general'].append(block)
        
        # Create chunks for each group
        for group_name, group_blocks in variable_groups.items():
            content_parts = []
            variables = {}
            
            for block in group_blocks:
                content_parts.append(block.content)
                variables[block.block_name] = block.attributes.get('default')
            
            chunks.append(self.create_chunk(
                content='\n\n'.join(content_parts),
                chunk_type='terraform_variables',
                metadata={
                    'group': group_name,
                    'variable_count': len(group_blocks),
                    'variables': list(variables.keys()),
                    'has_defaults': any(v is not None for v in variables.values())
                },
                file_path='variables.tf'
            ))
        
        return chunks
    
    def _chunk_outputs(self, blocks: List[TerraformBlock],
                      structure: TerraformStructure) -> List[Chunk]:
        """Chunk output blocks"""
        chunks = []
        content_parts = []
        outputs = {}
        
        for block in blocks:
            content_parts.append(block.content)
            outputs[block.block_name] = block.attributes.get('value')
        
        # Check if all outputs fit in one chunk
        combined_content = '\n\n'.join(content_parts)
        
        if self.count_tokens(combined_content) <= self.max_tokens:
            chunks.append(self.create_chunk(
                content=combined_content,
                chunk_type='terraform_outputs',
                metadata={
                    'output_count': len(blocks),
                    'outputs': list(outputs.keys())
                },
                file_path='outputs.tf'
            ))
        else:
            # Split outputs
            current_parts = []
            current_tokens = 0
            part_num = 1
            
            for part in content_parts:
                part_tokens = self.count_tokens(part)
                
                if current_tokens + part_tokens > self.max_tokens and current_parts:
                    chunks.append(self.create_chunk(
                        content='\n\n'.join(current_parts),
                        chunk_type='terraform_outputs_part',
                        metadata={
                            'part': part_num,
                            'output_count': len(current_parts)
                        },
                        file_path='outputs.tf'
                    ))
                    current_parts = [part]
                    current_tokens = part_tokens
                    part_num += 1
                else:
                    current_parts.append(part)
                    current_tokens += part_tokens
            
            # Add remaining
            if current_parts:
                chunks.append(self.create_chunk(
                    content='\n\n'.join(current_parts),
                    chunk_type='terraform_outputs_part',
                    metadata={
                        'part': part_num,
                        'output_count': len(current_parts),
                        'is_last': True
                    },
                    file_path='outputs.tf'
                ))
        
        return chunks
    
    def _extract_root_config(self, structure: TerraformStructure) -> str:
        """Extract root module configuration"""
        config_parts = []
        
        # Terraform block
        if structure.terraform_version or structure.required_providers:
            terraform_block = 'terraform {\n'
            if structure.terraform_version:
                terraform_block += f'  required_version = "{structure.terraform_version}"\n'
            if structure.required_providers:
                terraform_block += '  required_providers {\n'
                for provider, config in structure.required_providers.items():
                    if isinstance(config, dict):
                        terraform_block += f'    {provider} = {{\n'
                        for key, value in config.items():
                            terraform_block += f'      {key} = "{value}"\n'
                        terraform_block += '    }\n'
                    else:
                        terraform_block += f'    {provider} = "{config}"\n'
                terraform_block += '  }\n'
            if structure.backend:
                terraform_block += f'  backend "{structure.backend["type"]}" {{\n'
                if isinstance(structure.backend.get('config'), dict):
                    for key, value in structure.backend['config'].items():
                        terraform_block += f'    {key} = "{value}"\n'
                terraform_block += '  }\n'
            terraform_block += '}'
            config_parts.append(terraform_block)
        
        return '\n\n'.join(config_parts)
    
    def _create_module_chunk(self, module_block: TerraformBlock,
                            structure: TerraformStructure) -> Chunk:
        """Create chunk for module block"""
        return self.create_chunk(
            content=module_block.content,
            chunk_type='terraform_module',
            metadata={
                'module_name': module_block.block_name,
                'source': module_block.attributes.get('source'),
                'version': module_block.attributes.get('version'),
                'dependencies': list(module_block.dependencies),
                'references': list(module_block.references)
            },
            file_path=f'module_{module_block.block_name}.tf'
        )
    
    def _chunk_resources(self, blocks: List[TerraformBlock],
                        structure: TerraformStructure) -> List[Chunk]:
        """Chunk resource blocks"""
        chunks = []
        
        # Group resources by type
        resource_groups = defaultdict(list)
        
        for block in blocks:
            resource_groups[block.block_label].append(block)
        
        # Create chunks for each resource type
        for resource_type, resources in resource_groups.items():
            # Check if all resources of this type fit in one chunk
            combined_content = '\n\n'.join(r.content for r in resources)
            
            if self.count_tokens(combined_content) <= self.max_tokens:
                chunks.append(self.create_chunk(
                    content=combined_content,
                    chunk_type='terraform_resources',
                    metadata={
                        'resource_type': resource_type,
                        'resource_count': len(resources),
                        'resource_names': [r.block_name for r in resources],
                        'provider': resources[0].provider if resources else None
                    },
                    file_path=f'{resource_type}.tf'
                ))
            else:
                # Split resources
                for resource in resources:
                    chunks.extend(self._chunk_block(resource, structure))
        
        return chunks
    
    def _chunk_block(self, block: TerraformBlock,
                    structure: TerraformStructure) -> List[Chunk]:
        """Chunk a single block"""
        chunks = []
        
        # Check size
        if self.count_tokens(block.content) <= self.max_tokens:
            chunks.append(self.create_chunk(
                content=block.content,
                chunk_type=f'terraform_{block.block_type.value}',
                metadata={
                    'block_type': block.block_type.value,
                    'block_label': block.block_label,
                    'block_name': block.block_name,
                    'provider': block.provider,
                    'has_count': block.count is not None,
                    'has_for_each': block.for_each is not None,
                    'dependencies': list(block.dependencies)[:5],  # Limit to 5
                    'references': list(block.references)[:5]
                },
                file_path='main.tf'
            ))
        else:
            # Split large block
            chunks.extend(self._split_large_block(block, structure, None))
        
        return chunks
    
    def _split_large_block(self, block: TerraformBlock, structure: TerraformStructure,
                          file_context: Optional[FileContext]) -> List[Chunk]:
        """Split large Terraform block"""
        chunks = []
        
        # Try to split by logical sections
        sections = self._split_block_by_sections(block)
        
        for i, section in enumerate(sections):
            chunks.append(self.create_chunk(
                content=section['content'],
                chunk_type=f'terraform_{block.block_type.value}_part',
                metadata={
                    'block_type': block.block_type.value,
                    'block_label': block.block_label,
                    'block_name': block.block_name,
                    'section': section['type'],
                    'part': i + 1,
                    'total_parts': len(sections)
                },
                file_path='main.tf'
            ))
        
        return chunks
    
    def _split_block_by_sections(self, block: TerraformBlock) -> List[Dict[str, str]]:
        """Split block into logical sections"""
        sections = []
        lines = block.content.split('\n')
        
        # Try to identify sections
        current_section = []
        current_section_type = 'header'
        
        for line in lines:
            # Identify section boundaries
            if any(keyword in line for keyword in ['lifecycle', 'provisioner', 'connection', 'dynamic']):
                if current_section:
                    sections.append({
                        'type': current_section_type,
                        'content': '\n'.join(current_section)
                    })
                    current_section = []
                current_section_type = 'nested_block'
            
            current_section.append(line)
        
        # Add remaining section
        if current_section:
            sections.append({
                'type': current_section_type,
                'content': '\n'.join(current_section)
            })
        
        return sections if sections else [{'type': 'complete', 'content': block.content}]
    
    def _group_related_blocks(self, blocks: List[TerraformBlock]) -> List[List[TerraformBlock]]:
        """Group related blocks"""
        groups = []
        processed = set()
        
        for block in blocks:
            if id(block) in processed:
                continue
            
            group = [block]
            processed.add(id(block))
            
            # Find related blocks
            for other in blocks:
                if id(other) not in processed:
                    if self._are_blocks_related(block, other):
                        group.append(other)
                        processed.add(id(other))
            
            groups.append(group)
        
        return groups
    
    def _are_blocks_related(self, block1: TerraformBlock, block2: TerraformBlock) -> bool:
        """Check if two blocks are related"""
        # Same type and provider
        if (block1.block_type == block2.block_type and 
            block1.provider == block2.provider):
            return True
        
        # Dependency relationship
        block1_ref = f'{block1.block_label}.{block1.block_name}' if block1.block_name else block1.block_label
        block2_ref = f'{block2.block_label}.{block2.block_name}' if block2.block_name else block2.block_label
        
        if block1_ref in block2.dependencies or block2_ref in block1.dependencies:
            return True
        
        # Reference relationship
        if block1_ref in block2.references or block2_ref in block1.references:
            return True
        
        return False
    
    def _chunk_block_group(self, blocks: List[TerraformBlock],
                          structure: TerraformStructure) -> List[Chunk]:
        """Chunk a group of related blocks"""
        chunks = []
        
        # Combine blocks
        combined_content = '\n\n'.join(block.content for block in blocks)
        
        # Check size
        if self.count_tokens(combined_content) <= self.max_tokens:
            # Determine group type
            if all(b.block_type == blocks[0].block_type for b in blocks):
                group_type = blocks[0].block_type.value
            else:
                group_type = 'mixed'
            
            chunks.append(self.create_chunk(
                content=combined_content,
                chunk_type=f'terraform_{group_type}_group',
                metadata={
                    'block_count': len(blocks),
                    'block_types': list(set(b.block_type.value for b in blocks)),
                    'providers': list(set(b.provider for b in blocks if b.provider))
                },
                file_path='main.tf'
            ))
        else:
            # Chunk individually
            for block in blocks:
                chunks.extend(self._chunk_block(block, structure))
        
        return chunks
    
    def _fallback_chunking(self, content: str, file_context: FileContext) -> List[Chunk]:
        """Fallback chunking for invalid Terraform"""
        logger.warning(f"Using fallback chunking for Terraform file {file_context.path}")
        
        chunks = []
        
        # Try to split by top-level blocks
        block_pattern = re.compile(
            r'^(terraform|provider|resource|data|module|variable|output|locals)\s+',
            re.MULTILINE
        )
        
        # Find block positions
        block_positions = []
        for match in block_pattern.finditer(content):
            block_positions.append(match.start())
        
        # Add end position
        block_positions.append(len(content))
        
        # Extract blocks
        for i in range(len(block_positions) - 1):
            block_content = content[block_positions[i]:block_positions[i + 1]].strip()
            
            if block_content:
                # Check size
                if self.count_tokens(block_content) <= self.max_tokens:
                    chunks.append(self.create_chunk(
                        content=block_content,
                        chunk_type='terraform_fallback',
                        metadata={
                            'is_fallback': True,
                            'chunk_index': i
                        },
                        file_path=str(file_context.path)
                    ))
                else:
                    # Split by lines
                    lines = block_content.split('\n')
                    current_chunk = []
                    current_tokens = 0
                    
                    for line in lines:
                        line_tokens = self.count_tokens(line)
                        
                        if current_tokens + line_tokens > self.max_tokens and current_chunk:
                            chunks.append(self.create_chunk(
                                content='\n'.join(current_chunk),
                                chunk_type='terraform_fallback_part',
                                metadata={
                                    'is_fallback': True,
                                    'parent_index': i
                                },
                                file_path=str(file_context.path)
                            ))
                            current_chunk = []
                            current_tokens = 0
                        
                        current_chunk.append(line)
                        current_tokens += line_tokens
                    
                    # Add remaining
                    if current_chunk:
                        chunks.append(self.create_chunk(
                            content='\n'.join(current_chunk),
                            chunk_type='terraform_fallback_part',
                            metadata={
                                'is_fallback': True,
                                'parent_index': i,
                                'is_last': True
                            },
                            file_path=str(file_context.path)
                        ))
        
        return chunks if chunks else [self.create_chunk(
            content=content[:self.max_tokens * 4],  # Rough estimate
            chunk_type='terraform_fallback_truncated',
            metadata={'is_fallback': True, 'is_truncated': True},
            file_path=str(file_context.path)
        )]