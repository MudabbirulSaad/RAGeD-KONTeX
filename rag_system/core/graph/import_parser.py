"""
Import statement parser for building dependency graphs.

Extracts import relationships from code files using AST parsing.
"""

import ast
import re
from pathlib import Path
from typing import List, Set, Optional
from dataclasses import dataclass


@dataclass
class ImportInfo:
    """Information about an import statement."""
    
    module: str  # Module being imported (e.g., "os.path", "mymodule")
    names: List[str]  # Specific names imported (e.g., ["join", "exists"])
    is_relative: bool  # Whether it's a relative import (e.g., "from . import")
    level: int  # Relative import level (0 = absolute, 1 = ., 2 = .., etc.)


class ImportParser:
    """
    Parses import statements from code files.
    
    Supports Python, JavaScript/TypeScript, Java, Go, and more.
    """
    
    @staticmethod
    def parse_python_imports(content: str) -> List[ImportInfo]:
        """
        Parse Python import statements using AST.
        
        Args:
            content: Python source code
            
        Returns:
            List of ImportInfo objects
        """
        imports = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # import module1, module2
                    for alias in node.names:
                        imports.append(ImportInfo(
                            module=alias.name,
                            names=[],
                            is_relative=False,
                            level=0,
                        ))
                
                elif isinstance(node, ast.ImportFrom):
                    # from module import name1, name2
                    module = node.module or ""
                    names = [alias.name for alias in node.names]
                    
                    imports.append(ImportInfo(
                        module=module,
                        names=names,
                        is_relative=node.level > 0,
                        level=node.level,
                    ))
        
        except SyntaxError:
            # If AST parsing fails, fall back to regex
            return ImportParser._parse_python_imports_regex(content)
        
        return imports
    
    @staticmethod
    def _parse_python_imports_regex(content: str) -> List[ImportInfo]:
        """Fallback regex-based Python import parser."""
        imports = []
        
        # Match: import module
        for match in re.finditer(r'^import\s+([\w.]+)', content, re.MULTILINE):
            imports.append(ImportInfo(
                module=match.group(1),
                names=[],
                is_relative=False,
                level=0,
            ))
        
        # Match: from module import name1, name2
        for match in re.finditer(
            r'^from\s+(\.*)(\w[\w.]*)\s+import\s+(.+)',
            content,
            re.MULTILINE
        ):
            level = len(match.group(1))
            module = match.group(2)
            names_str = match.group(3)
            names = [n.strip() for n in names_str.split(',')]
            
            imports.append(ImportInfo(
                module=module,
                names=names,
                is_relative=level > 0,
                level=level,
            ))
        
        return imports
    
    @staticmethod
    def parse_javascript_imports(content: str) -> List[ImportInfo]:
        """
        Parse JavaScript/TypeScript import statements.
        
        Handles:
        - import { name1, name2 } from 'module'
        - import name from 'module'
        - import * as name from 'module'
        - const name = require('module')
        """
        imports = []
        
        # ES6 imports: import ... from 'module'
        pattern = r"import\s+(?:{([^}]+)}|(\w+)|\*\s+as\s+(\w+))\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(pattern, content):
            module = match.group(4)
            names = []

            if match.group(1):  # { name1, name2 }
                names = [n.strip() for n in match.group(1).split(',')]
            elif match.group(2):  # default import
                names = [match.group(2)]
            elif match.group(3):  # * as name
                names = [match.group(3)]

            # For JavaScript/TypeScript relative imports:
            # ./file -> level 0 (current directory)
            # ../file -> level 1 (parent directory)
            # ../../file -> level 2 (grandparent directory)
            is_relative = module.startswith('.')
            if is_relative:
                # Count leading dots, but subtract 1 because ./ is level 0
                level = len(module) - len(module.lstrip('.'))
                level = max(0, level - 1)  # ./ is level 0, ../ is level 1
                # Strip leading dots and slashes
                module = module.lstrip('./').lstrip('/')
            else:
                level = 0

            imports.append(ImportInfo(
                module=module,
                names=names,
                is_relative=is_relative,
                level=level,
            ))
        
        # CommonJS: require('module')
        pattern = r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
        for match in re.finditer(pattern, content):
            module = match.group(1)

            # Same logic as ES6 imports
            is_relative = module.startswith('.')
            if is_relative:
                level = len(module) - len(module.lstrip('.'))
                level = max(0, level - 1)
                module = module.lstrip('./').lstrip('/')
            else:
                level = 0

            imports.append(ImportInfo(
                module=module,
                names=[],
                is_relative=is_relative,
                level=level,
            ))

        return imports

    @staticmethod
    def parse_java_imports(content: str) -> List[ImportInfo]:
        """Parse Java import statements."""
        imports = []

        pattern = r"import\s+(static\s+)?([a-zA-Z_][\w.]*\*?);"
        for match in re.finditer(pattern, content):
            module = match.group(2)
            imports.append(ImportInfo(
                module=module,
                names=[],
                is_relative=False,
                level=0,
            ))

        return imports

    @staticmethod
    def parse_go_imports(content: str) -> List[ImportInfo]:
        """Parse Go import statements."""
        imports = []

        # Single import: import "module"
        pattern = r'import\s+"([^"]+)"'
        for match in re.finditer(pattern, content):
            imports.append(ImportInfo(
                module=match.group(1),
                names=[],
                is_relative=match.group(1).startswith('.'),
                level=0,
            ))

        # Multi-import block: import ( ... )
        pattern = r'import\s+\((.*?)\)'
        for match in re.finditer(pattern, content, re.DOTALL):
            block = match.group(1)
            for line in block.split('\n'):
                line = line.strip()
                if line and '"' in line:
                    module_match = re.search(r'"([^"]+)"', line)
                    if module_match:
                        imports.append(ImportInfo(
                            module=module_match.group(1),
                            names=[],
                            is_relative=module_match.group(1).startswith('.'),
                            level=0,
                        ))

        return imports

    @classmethod
    def parse_imports(cls, content: str, language: str) -> List[ImportInfo]:
        """
        Parse imports for any supported language.

        Args:
            content: Source code content
            language: Programming language

        Returns:
            List of ImportInfo objects
        """
        language = language.lower()

        if language == "python":
            return cls.parse_python_imports(content)
        elif language in ("javascript", "typescript"):
            return cls.parse_javascript_imports(content)
        elif language == "java":
            return cls.parse_java_imports(content)
        elif language == "go":
            return cls.parse_go_imports(content)
        else:
            # Unsupported language, return empty list
            return []

    @staticmethod
    def resolve_relative_import(
        current_file: str,
        import_module: str,
        is_relative: bool,
        level: int,
    ) -> Optional[str]:
        """
        Resolve a relative import to an absolute module path.

        Args:
            current_file: Path to the file containing the import
            import_module: The imported module name
            is_relative: Whether it's a relative import
            level: Relative import level (1 = ., 2 = .., etc.)

        Returns:
            Resolved module path or None if cannot resolve
        """
        if not is_relative:
            return import_module

        current_path = Path(current_file).parent

        # Go up 'level' directories
        for _ in range(level):
            current_path = current_path.parent

        # Combine with import module
        if import_module:
            resolved = current_path / import_module.replace('.', '/')
        else:
            resolved = current_path

        return str(resolved)

