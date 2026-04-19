"""
omen_ast_multilang.py — Multilingual AST -> Horn facts
======================================================
Supported languages (via tree-sitter):
  python · javascript · typescript · java · c · cpp · rust · go · bash · lua

Architecture:
  MultiLangASTParser        — entry point that selects the parser by language
  _TreeSitterVisitor        — base AST visitor
  Language-specific parsers — Python, JS/TS, Java, C/C++, Rust, Go, Bash, Lua

Output:  List[HornAtom]  (same structure as PythonASTParser)
Integration: connects to ScalableKnowledgeBase through add_facts()

Fact structure (predicates from PredicateVocab):
  define(name, scope)       — function / class declaration
  call(call_id, func)       — function call
  call_arg(call_id, arg)    — call argument
  assign(lhs, rhs)          — assignment
  param(func, arg)          — function parameter
  return(func, val)         — return expr
  import(name, module)      — import / include / use
  type_of(var, type)        — type annotation
  attr(obj, field)          — obj.field access
  if_true(cond, scope)      — if condition
  loop_body(iter, scope)    — loop body
  dep_data(a, b)            — data dependency  a <- b
  classdef(name, scope)     — class
  classbase(name, base)     — inheritance
  decorator(target, dec)    — decorator
  raise(exc, scope)         — exception
  with(ctx, scope)          — with/using
  lang(source_id, lang)     — source language (meta fact)

Example:
    from omen_ast_multilang import MultiLangASTParser
    parser = MultiLangASTParser()
    facts = parser.parse(code_str, lang="python", source_id=0)
"""

from __future__ import annotations

import ast as _pyast
import re
import textwrap
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

warnings.filterwarnings("ignore")

# ─── OMEN dependencies ───────────────────────────────────────────────────────
from omen_prolog import Const, HornAtom
from omen_tensor_unify import PRED_VOCAB, ConstPool


# ══════════════════════════════════════════════════════════════════════════════
# 0. TREE-SITTER REGISTRY (lazy-loaded, each language is initialized once)
# ══════════════════════════════════════════════════════════════════════════════

class _TSRegistry:
    """Lazy loader for tree-sitter Language objects."""

    # (pkg_name, lang_func_name) — some packages use non-standard function names
    _PACKAGES = {
        "python":     ("tree_sitter_python",     "language"),
        "javascript": ("tree_sitter_javascript",  "language"),
        "typescript": ("tree_sitter_typescript",  "language_typescript"),
        "java":       ("tree_sitter_java",        "language"),
        "c":          ("tree_sitter_c",           "language"),
        "cpp":        ("tree_sitter_cpp",         "language"),
        "rust":       ("tree_sitter_rust",        "language"),
        "go":         ("tree_sitter_go",          "language"),
        "bash":       ("tree_sitter_bash",        "language"),
        "lua":        ("tree_sitter_lua",         "language"),
    }

    def __init__(self):
        self._languages: Dict[str, object] = {}
        self._parsers:   Dict[str, object] = {}
        self._available: Set[str] = set()
        self._checked   = False

    def _init(self):
        if self._checked:
            return
        self._checked = True
        try:
            from tree_sitter import Language, Parser as TSParser
            self._Language = Language
            self._Parser   = TSParser
        except ImportError:
            self._Language = None
            self._Parser   = None
            return

        for lang, (pkg, fn_name) in self._PACKAGES.items():
            try:
                mod = __import__(pkg)
                lang_fn = getattr(mod, fn_name, None)
                if lang_fn is None:
                    continue
                self._languages[lang] = self._Language(lang_fn())
                self._available.add(lang)
            except Exception:
                pass

    def get_parser(self, lang: str):
        self._init()
        if lang not in self._available:
            return None
        if lang not in self._parsers:
            p = self._Parser(self._languages[lang])
            self._parsers[lang] = p
        return self._parsers[lang]

    def available(self) -> Set[str]:
        self._init()
        return set(self._available)


TS_REGISTRY = _TSRegistry()


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASE VISITOR (shared logic)
# ══════════════════════════════════════════════════════════════════════════════

class _BaseASTVisitor(ABC):
    """
    Base class for all language parsers.
    Subclasses implement `_visit_node()` for a specific language.
    """

    def __init__(self, source_id: int = 0):
        self.pool       = ConstPool()
        self.facts: List[HornAtom] = []
        self._source_id = source_id
        self._call_cnt  = 0
        self._scope_stack: List[str] = [f"<src{source_id}>"]

    # ─── Convenience helpers ─────────────────────────────────────────────────

    @property
    def _cur_scope(self) -> str:
        return self._scope_stack[-1]

    def _fact(self, pred_name: str, *args: str) -> None:
        self.facts.append(self.pool.atom(pred_name, *args))

    def _fresh_call_id(self) -> str:
        cid = f"_c{self._call_cnt}"
        self._call_cnt += 1
        return cid

    # ─── Abstract method ─────────────────────────────────────────────────────

    @abstractmethod
    def parse(self, code: str) -> List[HornAtom]:
        """Main entry point. Returns a list of HornAtom."""


# ══════════════════════════════════════════════════════════════════════════════
# 2. PYTHON VISITOR (via stdlib ast, without tree-sitter overhead)
# ══════════════════════════════════════════════════════════════════════════════

class _PythonVisitor(_BaseASTVisitor, _pyast.NodeVisitor):
    """Python -> Horn facts through stdlib ast (faster than tree-sitter for Python)."""

    def parse(self, code: str) -> List[HornAtom]:
        code = textwrap.dedent(code)
        try:
            tree = _pyast.parse(code)
        except SyntaxError:
            return []
        self.visit(tree)
        return self.facts

    def _name(self, node) -> str:
        if isinstance(node, _pyast.Name):       return node.id
        if isinstance(node, _pyast.Attribute):  return f"{self._name(node.value)}.{node.attr}"
        if isinstance(node, _pyast.Constant):   return repr(node.value)[:32]
        if isinstance(node, _pyast.Subscript):  return f"{self._name(node.value)}[…]"
        if isinstance(node, _pyast.Call):       return f"call_{self._name(node.func)}"
        if isinstance(node, _pyast.BinOp):
            op = type(node.op).__name__.lower()
            return f"{self._name(node.left)}_{op}_{self._name(node.right)}"
        return f"<{type(node).__name__}>"

    def visit_Assign(self, node):
        rhs = self._name(node.value)
        for tgt in node.targets:
            lhs = self._name(tgt)
            self._fact("assign", lhs, rhs)
            self._fact("dep_data", lhs, rhs)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        tgt = self._name(node.target)
        ann = self._name(node.annotation)
        self._fact("annot", tgt, ann)
        self._fact("type_of", tgt, ann)
        if node.value:
            self._fact("assign", tgt, self._name(node.value))
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        tgt = self._name(node.target)
        val = self._name(node.value)
        op  = type(node.op).__name__.lower()
        self._fact("augassign", tgt, op)
        self._fact("dep_data", tgt, val)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        fname = node.name
        self._fact("define", fname, self._cur_scope)
        for dec in node.decorator_list:
            self._fact("decorator", fname, self._name(dec))
        for arg in node.args.args:
            self._fact("param", fname, arg.arg)
            if arg.annotation:
                self._fact("type_of", arg.arg, self._name(arg.annotation))
        if node.returns:
            self._fact("type_of", f"{fname}:ret", self._name(node.returns))
        self._scope_stack.append(fname)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        cname = node.name
        self._fact("classdef", cname, self._cur_scope)
        for base in node.bases:
            self._fact("classbase", cname, self._name(base))
        for dec in node.decorator_list:
            self._fact("decorator", cname, self._name(dec))
        self._scope_stack.append(cname)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Return(self, node):
        if node.value:
            val = self._name(node.value)
            self._fact("return", self._cur_scope, val)
            self._fact("dep_data", f"{self._cur_scope}:ret", val)
        self.generic_visit(node)

    def visit_Call(self, node):
        cid  = self._fresh_call_id()
        func = self._name(node.func)
        self._fact("call", cid, func)
        for arg in node.args:
            self._fact("call_arg", cid, self._name(arg))
        for kw in node.keywords:
            key = kw.arg or "**kwargs"
            self._fact("call_arg", cid, f"{key}={self._name(kw.value)}")
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self._fact("import", name, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or "<unknown>"
        for alias in node.names:
            name = alias.asname or alias.name
            self._fact("import", name, module)
        self.generic_visit(node)

    def visit_If(self, node):
        cond = self._name(node.test)
        self._fact("if_true", cond, self._cur_scope)
        if node.orelse:
            self._fact("if_false", cond, self._cur_scope)
        self.generic_visit(node)

    def visit_For(self, node):
        it  = self._name(node.iter)
        tgt = self._name(node.target)
        self._fact("loop_body", it, self._cur_scope)
        self._fact("assign", tgt, it)
        self.generic_visit(node)

    def visit_While(self, node):
        cond = self._name(node.test)
        self._fact("loop_cond", cond, self._cur_scope)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        obj = self._name(node.value)
        self._fact("attr", obj, node.attr)
        self.generic_visit(node)

    def visit_Global(self, node):
        for name in node.names:
            self._fact("global", name, self._cur_scope)
        self.generic_visit(node)

    def visit_Raise(self, node):
        exc = self._name(node.exc) if node.exc else "<none>"
        self._fact("raise", exc, self._cur_scope)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            ctx = self._name(item.context_expr)
            self._fact("with", ctx, self._cur_scope)
        self.generic_visit(node)

    def visit_Lambda(self, node):
        lid = f"lam{len(self.facts)}"
        self._fact("lambda", lid, self._cur_scope)
        self.generic_visit(node)

    def visit_Delete(self, node):
        for tgt in node.targets:
            self._fact("delete", self._name(tgt), self._cur_scope)
        self.generic_visit(node)

    def visit_Yield(self, node):
        val = self._name(node.value) if node.value else "<none>"
        self._fact("yield", self._cur_scope, val)
        self.generic_visit(node)

    def visit_Try(self, node):
        self._fact("try", self._cur_scope, "<try>")
        for handler in node.handlers:
            exc_type = self._name(handler.type) if handler.type else "Exception"
            self._fact("except_handler", self._cur_scope, exc_type)
        self.generic_visit(node)

    def visit_Assert(self, node):
        test = self._name(node.test)
        self._fact("assert", test, self._cur_scope)
        self.generic_visit(node)

    def visit_ListComp(self, node):
        lid = f"lcomp{len(self.facts)}"
        self._fact("listcomp", lid, self._cur_scope)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        lid = f"dcomp{len(self.facts)}"
        self._fact("dictcomp", lid, self._cur_scope)
        self.generic_visit(node)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TREE-SITTER BASE VISITOR
# ══════════════════════════════════════════════════════════════════════════════

class _TSBaseVisitor(_BaseASTVisitor):
    """
    Base class for tree-sitter parsers.
    Subclasses override `_dispatch()` for language-specific node types.
    """

    LANG: str = ""  # set by subclasses

    # Terminal nodes (leaves, do not recurse into them)
    TERMINAL_TYPES: Set[str] = {
        "comment", "line_comment", "block_comment",
        "string", "number", "boolean",
    }

    def parse(self, code: str) -> List[HornAtom]:
        parser = TS_REGISTRY.get_parser(self.LANG)
        if parser is None:
            return []
        try:
            tree = parser.parse(bytes(code, "utf-8"))
            self._walk(tree.root_node)
        except Exception as e:
            warnings.warn(f"[omen_ast_multilang] {self.LANG} parse error: {e}")
        return self.facts

    def _walk(self, node) -> None:
        """DFS tree walk with dispatch by node type."""
        if node.type in self.TERMINAL_TYPES:
            return
        handled = self._dispatch(node)
        if not handled:
            for child in node.children:
                self._walk(child)

    def _dispatch(self, node) -> bool:
        """
        Return True when the node is fully handled and recursion is unnecessary.
        Return False when traversal should continue through child nodes.
        """
        return False

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _text(self, node) -> str:
        """Node text truncated to 64 characters."""
        return node.text.decode("utf-8", errors="replace")[:64].strip()

    def _child_text(self, node, field_name: str) -> Optional[str]:
        child = node.child_by_field_name(field_name)
        if child is None:
            return None
        return self._text(child)

    def _named_child_text(self, node, kind: str) -> Optional[str]:
        for c in node.named_children:
            if c.type == kind:
                return self._text(c)
        return None

    def _all_children_of_type(self, node, kind: str) -> List:
        return [c for c in node.named_children if c.type == kind]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  JAVASCRIPT / TYPESCRIPT VISITOR
# ══════════════════════════════════════════════════════════════════════════════

class _JSVisitor(_TSBaseVisitor):
    LANG = "javascript"

    def _dispatch(self, node) -> bool:
        t = node.type

        if t in ("function_declaration", "function_expression",
                 "arrow_function", "method_definition"):
            return self._handle_func(node)

        if t == "class_declaration":
            return self._handle_class(node)

        if t in ("lexical_declaration", "variable_declaration"):
            return self._handle_var_decl(node)

        if t == "assignment_expression":
            lhs = self._child_text(node, "left") or "<lhs>"
            rhs = self._child_text(node, "right") or "<rhs>"
            self._fact("assign", lhs, rhs)
            self._fact("dep_data", lhs, rhs)
            for c in node.children: self._walk(c)
            return True

        if t == "call_expression":
            return self._handle_call(node)

        if t in ("import_statement", "import_declaration"):
            return self._handle_import(node)

        if t == "export_statement":
            for c in node.children: self._walk(c)
            return True

        if t == "if_statement":
            cond_node = node.child_by_field_name("condition")
            cond = self._text(cond_node) if cond_node else "<cond>"
            self._fact("if_true", cond, self._cur_scope)
            alt = node.child_by_field_name("alternative")
            if alt: self._fact("if_false", cond, self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t in ("for_statement", "for_in_statement", "while_statement"):
            self._fact("loop_body", "<iter>", self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t == "return_statement":
            val_node = node.child_by_field_name("value")
            val = self._text(val_node) if val_node else "<void>"
            self._fact("return", self._cur_scope, val)
            self._fact("dep_data", f"{self._cur_scope}:ret", val)
            for c in node.children: self._walk(c)
            return True

        if t == "throw_statement":
            exc_node = node.named_children[0] if node.named_children else None
            exc = self._text(exc_node) if exc_node else "<exc>"
            self._fact("raise", exc, self._cur_scope)
            return True

        if t == "member_expression":
            obj  = self._child_text(node, "object")  or "<obj>"
            prop = self._child_text(node, "property") or "<prop>"
            self._fact("attr", obj, prop)
            return False

        if t == "try_statement":
            self._fact("try", self._cur_scope, "<try>")
            for c in node.children: self._walk(c)
            return True

        return False

    def _handle_func(self, node) -> bool:
        name_node = node.child_by_field_name("name")
        fname = self._text(name_node) if name_node else f"<fn{len(self.facts)}>"
        self._fact("define", fname, self._cur_scope)

        # Parameters
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for param in params_node.named_children:
                if param.type in ("identifier", "formal_parameters"):
                    self._fact("param", fname, self._text(param))

        # Decorators (TypeScript)
        for dec in self._all_children_of_type(node, "decorator"):
            self._fact("decorator", fname, self._text(dec))

        self._scope_stack.append(fname)
        body = node.child_by_field_name("body")
        if body:
            self._walk(body)
        self._scope_stack.pop()
        return True

    def _handle_class(self, node) -> bool:
        # TypeScript uses type_identifier; JS uses identifier
        name_node = node.child_by_field_name("name")
        if name_node is None:
            for c in node.named_children:
                if c.type in ("identifier", "type_identifier"):
                    name_node = c
                    break
        cname = self._text(name_node) if name_node else f"<cls{len(self.facts)}>"
        self._fact("classdef", cname, self._cur_scope)

        # JS: class_heritage  |  TS: class_heritage / extends_clause
        for child in node.named_children:
            if child.type in ("class_heritage", "extends_clause",
                              "implements_clause", "class_implements"):
                for c in child.named_children:
                    if c.type not in ("extends", "implements", ","):
                        self._fact("classbase", cname, self._text(c))

        self._scope_stack.append(cname)
        body = node.child_by_field_name("body")
        if body:
            self._walk(body)
        self._scope_stack.pop()
        return True

    def _handle_var_decl(self, node) -> bool:
        for decl in self._all_children_of_type(node, "variable_declarator"):
            name_node = decl.child_by_field_name("name")
            val_node  = decl.child_by_field_name("value")
            if name_node:
                lhs = self._text(name_node)
                rhs = self._text(val_node) if val_node else "<undefined>"
                self._fact("assign", lhs, rhs)
                self._fact("dep_data", lhs, rhs)
        for c in node.children: self._walk(c)
        return True

    def _handle_call(self, node) -> bool:
        cid  = self._fresh_call_id()
        func_node = node.child_by_field_name("function")
        func = self._text(func_node) if func_node else "<func>"
        self._fact("call", cid, func)

        args_node = node.child_by_field_name("arguments")
        if args_node:
            for arg in args_node.named_children:
                self._fact("call_arg", cid, self._text(arg))
        for c in node.children: self._walk(c)
        return True

    def _handle_import(self, node) -> bool:
        source_node = (
            node.child_by_field_name("source") or
            self._named_child_text(node, "string")
        )
        src = self._text(source_node).strip("'\"`") if source_node else "<module>"
        # Named imports
        for clause in self._all_children_of_type(node, "import_clause"):
            for name_node in clause.named_children:
                self._fact("import", self._text(name_node), src)
        # Default import
        def_node = node.child_by_field_name("default")
        if def_node:
            self._fact("import", self._text(def_node), src)
        if not node.child_by_field_name("default") and not self._all_children_of_type(node, "import_clause"):
            self._fact("import", src, src)
        return True


class _TSVisitor(_JSVisitor):
    LANG = "typescript"

    def _dispatch(self, node) -> bool:
        t = node.type

        # TypeScript-specific constructs
        if t == "interface_declaration":
            name_node = node.child_by_field_name("name")
            iname = self._text(name_node) if name_node else "<iface>"
            self._fact("classdef", iname, self._cur_scope)
            self._fact("type_of", iname, "interface")
            # Extends
            for clause in self._all_children_of_type(node, "extends_clause"):
                for base in clause.named_children:
                    self._fact("classbase", iname, self._text(base))
            self._scope_stack.append(iname)
            for c in node.children: self._walk(c)
            self._scope_stack.pop()
            return True

        if t == "type_alias_declaration":
            name_node = node.child_by_field_name("name")
            tname = self._text(name_node) if name_node else "<type>"
            val_node = node.child_by_field_name("value")
            tval  = self._text(val_node) if val_node else "<type_val>"
            self._fact("assign", tname, tval)
            self._fact("type_of", tname, tval)
            return True

        if t in ("type_annotation", "type_assertion"):
            return False  # continue traversal

        if t == "enum_declaration":
            name_node = node.child_by_field_name("name")
            ename = self._text(name_node) if name_node else "<enum>"
            self._fact("classdef", ename, self._cur_scope)
            self._fact("type_of", ename, "enum")
            return True

        if t == "decorator":
            for c in node.named_children:
                self._fact("decorator", self._cur_scope, self._text(c))
            return True

        return super()._dispatch(node)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  JAVA VISITOR
# ══════════════════════════════════════════════════════════════════════════════

class _JavaVisitor(_TSBaseVisitor):
    LANG = "java"

    def _dispatch(self, node) -> bool:
        t = node.type

        if t in ("method_declaration", "constructor_declaration"):
            name_node = node.child_by_field_name("name")
            fname = self._text(name_node) if name_node else "<method>"
            self._fact("define", fname, self._cur_scope)

            params_node = node.child_by_field_name("parameters")
            if params_node:
                for p in self._all_children_of_type(params_node, "formal_parameter"):
                    pname = p.child_by_field_name("name")
                    ptype = p.child_by_field_name("type")
                    if pname:
                        self._fact("param", fname, self._text(pname))
                    if pname and ptype:
                        self._fact("type_of", self._text(pname), self._text(ptype))

            ret_type = node.child_by_field_name("type")
            if ret_type:
                self._fact("type_of", f"{fname}:ret", self._text(ret_type))

            self._scope_stack.append(fname)
            body = node.child_by_field_name("body")
            if body: self._walk(body)
            self._scope_stack.pop()
            return True

        if t in ("class_declaration", "interface_declaration",
                 "enum_declaration", "record_declaration"):
            name_node = node.child_by_field_name("name")
            cname = self._text(name_node) if name_node else "<class>"
            self._fact("classdef", cname, self._cur_scope)
            if t == "interface_declaration":
                self._fact("type_of", cname, "interface")
            elif t == "enum_declaration":
                self._fact("type_of", cname, "enum")

            # Extends / implements
            for clause in node.named_children:
                if clause.type == "superclass":
                    base = clause.named_children[0] if clause.named_children else None
                    if base: self._fact("classbase", cname, self._text(base))
                if clause.type in ("super_interfaces", "type_list"):
                    for iface in clause.named_children:
                        self._fact("classbase", cname, self._text(iface))

            # Annotations
            for ann in self._all_children_of_type(node, "annotation"):
                self._fact("decorator", cname, self._text(ann))

            self._scope_stack.append(cname)
            body = node.child_by_field_name("body")
            if body: self._walk(body)
            self._scope_stack.pop()
            return True

        if t == "local_variable_declaration":
            type_node = node.child_by_field_name("type")
            for decl in self._all_children_of_type(node, "variable_declarator"):
                name_node = decl.child_by_field_name("name")
                val_node  = decl.child_by_field_name("value")
                if name_node:
                    lhs = self._text(name_node)
                    rhs = self._text(val_node) if val_node else "<null>"
                    self._fact("assign", lhs, rhs)
                    if type_node:
                        self._fact("type_of", lhs, self._text(type_node))
            for c in node.children: self._walk(c)
            return True

        if t == "assignment_expression":
            lhs = self._child_text(node, "left") or "<lhs>"
            rhs = self._child_text(node, "right") or "<rhs>"
            self._fact("assign", lhs, rhs)
            self._fact("dep_data", lhs, rhs)
            for c in node.children: self._walk(c)
            return True

        if t == "method_invocation":
            cid = self._fresh_call_id()
            name_node = node.child_by_field_name("name")
            obj_node  = node.child_by_field_name("object")
            func = self._text(name_node) if name_node else "<method>"
            if obj_node:
                func = f"{self._text(obj_node)}.{func}"
                self._fact("attr", self._text(obj_node), self._text(name_node) if name_node else func)
            self._fact("call", cid, func)
            args_node = node.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.named_children:
                    self._fact("call_arg", cid, self._text(arg))
            for c in node.children: self._walk(c)
            return True

        if t == "import_declaration":
            for c in node.named_children:
                pkg = self._text(c)
                self._fact("import", pkg.split(".")[-1], pkg)
            return True

        if t == "if_statement":
            cond_node = node.child_by_field_name("condition")
            cond = self._text(cond_node) if cond_node else "<cond>"
            self._fact("if_true", cond, self._cur_scope)
            if node.child_by_field_name("alternative"):
                self._fact("if_false", cond, self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t in ("for_statement", "enhanced_for_statement", "while_statement"):
            self._fact("loop_body", "<iter>", self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t == "return_statement":
            val_node = node.named_children[0] if node.named_children else None
            val = self._text(val_node) if val_node else "<void>"
            self._fact("return", self._cur_scope, val)
            self._fact("dep_data", f"{self._cur_scope}:ret", val)
            return True

        if t == "throw_statement":
            exc_node = node.named_children[0] if node.named_children else None
            exc = self._text(exc_node) if exc_node else "<exc>"
            self._fact("raise", exc, self._cur_scope)
            return True

        if t == "try_statement":
            self._fact("try", self._cur_scope, "<try>")
            for c in node.children: self._walk(c)
            return True

        if t == "field_access":
            obj  = self._child_text(node, "object") or "<obj>"
            fld  = self._child_text(node, "field")  or "<fld>"
            self._fact("attr", obj, fld)
            return False

        return False


# ══════════════════════════════════════════════════════════════════════════════
# 6.  C / C++ VISITOR
# ══════════════════════════════════════════════════════════════════════════════

class _CVisitor(_TSBaseVisitor):
    LANG = "c"

    def _dispatch(self, node) -> bool:
        t = node.type

        if t in ("function_definition", "function_declarator"):
            return self._handle_func(node)

        if t == "declaration":
            return self._handle_decl(node)

        if t == "assignment_expression":
            lhs = self._child_text(node, "left") or "<lhs>"
            rhs = self._child_text(node, "right") or "<rhs>"
            self._fact("assign", lhs, rhs)
            self._fact("dep_data", lhs, rhs)
            for c in node.children: self._walk(c)
            return True

        if t == "call_expression":
            cid  = self._fresh_call_id()
            func_node = node.child_by_field_name("function")
            func = self._text(func_node) if func_node else "<func>"
            self._fact("call", cid, func)
            args_node = node.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.named_children:
                    self._fact("call_arg", cid, self._text(arg))
            for c in node.children: self._walk(c)
            return True

        if t in ("preproc_include",):
            path_node = node.named_children[0] if node.named_children else None
            path = self._text(path_node).strip("<>\"") if path_node else "<inc>"
            self._fact("import", path, path)
            return True

        if t == "if_statement":
            cond_node = node.child_by_field_name("condition")
            cond = self._text(cond_node) if cond_node else "<cond>"
            self._fact("if_true", cond, self._cur_scope)
            alt = node.child_by_field_name("alternative")
            if alt: self._fact("if_false", cond, self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t in ("for_statement", "while_statement", "do_statement"):
            self._fact("loop_body", "<iter>", self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t == "return_statement":
            val_node = node.named_children[0] if node.named_children else None
            val = self._text(val_node) if val_node else "<void>"
            self._fact("return", self._cur_scope, val)
            self._fact("dep_data", f"{self._cur_scope}:ret", val)
            return True

        if t == "field_expression":
            arg = node.child_by_field_name("argument")
            fld = node.child_by_field_name("field")
            if arg and fld:
                self._fact("attr", self._text(arg), self._text(fld))
            return False

        if t in ("struct_specifier", "union_specifier"):
            name_node = node.child_by_field_name("name")
            if name_node:
                self._fact("classdef", self._text(name_node), self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        return False

    def _handle_func(self, node) -> bool:
        # Name extracted from the declarator
        decl_node = node.child_by_field_name("declarator")
        name = self._extract_func_name(decl_node or node)
        fname = name or f"<fn{len(self.facts)}>"
        self._fact("define", fname, self._cur_scope)

        # Parameters
        params_node = self._find_params(node)
        if params_node:
            for p in params_node.named_children:
                pname_node = p.child_by_field_name("declarator")
                ptype_node = p.child_by_field_name("type")
                if pname_node:
                    pname = self._text(pname_node)
                    self._fact("param", fname, pname)
                    if ptype_node:
                        self._fact("type_of", pname, self._text(ptype_node))

        # Return type
        type_node = node.child_by_field_name("type")
        if type_node:
            self._fact("type_of", f"{fname}:ret", self._text(type_node))

        self._scope_stack.append(fname)
        body = node.child_by_field_name("body")
        if body: self._walk(body)
        self._scope_stack.pop()
        return True

    def _extract_func_name(self, node) -> Optional[str]:
        if node is None:
            return None
        if node.type == "identifier":
            return self._text(node)
        for c in node.children:
            r = self._extract_func_name(c)
            if r:
                return r
        return None

    def _find_params(self, node):
        for c in node.children:
            if c.type == "parameter_list":
                return c
        return None

    def _handle_decl(self, node) -> bool:
        type_node = node.child_by_field_name("type")
        for decl in node.named_children:
            if decl.type in ("init_declarator", "identifier"):
                name_node = decl.child_by_field_name("declarator") or decl
                val_node  = decl.child_by_field_name("value")
                lhs = self._text(name_node)
                if lhs and type_node:
                    self._fact("type_of", lhs, self._text(type_node))
                if val_node:
                    self._fact("assign", lhs, self._text(val_node))
        for c in node.children: self._walk(c)
        return True


class _CppVisitor(_CVisitor):
    LANG = "cpp"

    def _dispatch(self, node) -> bool:
        t = node.type

        if t == "class_specifier":
            name_node = node.child_by_field_name("name")
            cname = self._text(name_node) if name_node else f"<cls{len(self.facts)}>"
            self._fact("classdef", cname, self._cur_scope)
            # Base classes
            base_clause = node.child_by_field_name("base_clause")
            if base_clause:
                for base in base_clause.named_children:
                    self._fact("classbase", cname, self._text(base))
            self._scope_stack.append(cname)
            body = node.child_by_field_name("body")
            if body: self._walk(body)
            self._scope_stack.pop()
            return True

        if t == "namespace_definition":
            name_node = node.child_by_field_name("name")
            ns = self._text(name_node) if name_node else "<anon_ns>"
            self._fact("classdef", ns, self._cur_scope)
            self._scope_stack.append(ns)
            body = node.child_by_field_name("body")
            if body: self._walk(body)
            self._scope_stack.pop()
            return True

        if t == "using_declaration":
            for c in node.named_children:
                self._fact("import", self._text(c), "<std>")
            return True

        if t == "template_declaration":
            for c in node.children: self._walk(c)
            return True

        return super()._dispatch(node)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  RUST VISITOR
# ══════════════════════════════════════════════════════════════════════════════

class _RustVisitor(_TSBaseVisitor):
    LANG = "rust"

    def _dispatch(self, node) -> bool:
        t = node.type

        if t in ("function_item", "method"):
            name_node = node.child_by_field_name("name")
            fname = self._text(name_node) if name_node else f"<fn{len(self.facts)}>"
            self._fact("define", fname, self._cur_scope)

            params_node = node.child_by_field_name("parameters")
            if params_node:
                for p in self._all_children_of_type(params_node, "parameter"):
                    pname = p.child_by_field_name("pattern")
                    ptype = p.child_by_field_name("type")
                    if pname:
                        self._fact("param", fname, self._text(pname))
                        if ptype:
                            self._fact("type_of", self._text(pname), self._text(ptype))

            ret_node = node.child_by_field_name("return_type")
            if ret_node:
                self._fact("type_of", f"{fname}:ret", self._text(ret_node))

            self._scope_stack.append(fname)
            body = node.child_by_field_name("body")
            if body: self._walk(body)
            self._scope_stack.pop()
            return True

        if t in ("struct_item", "enum_item", "trait_item", "impl_item"):
            name_node = node.child_by_field_name("name")
            sname = self._text(name_node) if name_node else f"<s{len(self.facts)}>"
            self._fact("classdef", sname, self._cur_scope)
            if t == "trait_item":
                self._fact("type_of", sname, "trait")
            elif t == "impl_item":
                type_node = node.child_by_field_name("type")
                if type_node:
                    self._fact("classbase", sname, self._text(type_node))
            self._scope_stack.append(sname)
            body = node.child_by_field_name("body")
            if body: self._walk(body)
            self._scope_stack.pop()
            return True

        if t == "let_declaration":
            pat_node = node.child_by_field_name("pattern")
            val_node = node.child_by_field_name("value")
            typ_node = node.child_by_field_name("type")
            if pat_node:
                lhs = self._text(pat_node)
                rhs = self._text(val_node) if val_node else "<none>"
                self._fact("assign", lhs, rhs)
                self._fact("dep_data", lhs, rhs)
                if typ_node:
                    self._fact("type_of", lhs, self._text(typ_node))
            for c in node.children: self._walk(c)
            return True

        if t == "call_expression":
            cid  = self._fresh_call_id()
            func = node.child_by_field_name("function")
            self._fact("call", cid, self._text(func) if func else "<fn>")
            args_node = node.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.named_children:
                    self._fact("call_arg", cid, self._text(arg))
            for c in node.children: self._walk(c)
            return True

        if t == "method_call_expression":
            cid      = self._fresh_call_id()
            recv     = node.child_by_field_name("receiver")
            meth     = node.child_by_field_name("method")
            func_str = f"{self._text(recv)}.{self._text(meth)}" if recv and meth else "<method>"
            if recv and meth:
                self._fact("attr", self._text(recv), self._text(meth))
            self._fact("call", cid, func_str)
            args_node = node.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.named_children:
                    self._fact("call_arg", cid, self._text(arg))
            for c in node.children: self._walk(c)
            return True

        if t == "use_declaration":
            for c in node.named_children:
                pkg = self._text(c)
                self._fact("import", pkg.split("::")[-1], pkg)
            return True

        if t in ("if_expression", "if_let_expression"):
            cond_node = node.child_by_field_name("condition")
            cond = self._text(cond_node) if cond_node else "<cond>"
            self._fact("if_true", cond, self._cur_scope)
            if node.child_by_field_name("alternative"):
                self._fact("if_false", cond, self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t in ("for_expression", "while_expression", "loop_expression"):
            self._fact("loop_body", "<iter>", self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t == "return_expression":
            val = node.named_children[0] if node.named_children else None
            self._fact("return", self._cur_scope, self._text(val) if val else "<()>")
            if val:
                self._fact("dep_data", f"{self._cur_scope}:ret", self._text(val))
            return True

        if t == "field_expression":
            val = node.child_by_field_name("value")
            fld = node.child_by_field_name("field")
            if val and fld:
                self._fact("attr", self._text(val), self._text(fld))
            return False

        if t == "macro_invocation":
            cid = self._fresh_call_id()
            macro = node.child_by_field_name("macro")
            self._fact("call", cid, f"macro!{self._text(macro)}" if macro else "<macro>")
            return False

        return False


# ══════════════════════════════════════════════════════════════════════════════
# 8.  GO VISITOR
# ══════════════════════════════════════════════════════════════════════════════

class _GoVisitor(_TSBaseVisitor):
    LANG = "go"

    def _dispatch(self, node) -> bool:
        t = node.type

        if t == "function_declaration":
            name_node = node.child_by_field_name("name")
            fname = self._text(name_node) if name_node else f"<fn{len(self.facts)}>"
            self._fact("define", fname, self._cur_scope)

            params_node = node.child_by_field_name("parameters")
            if params_node:
                for p in self._all_children_of_type(params_node, "parameter_declaration"):
                    pname = p.child_by_field_name("name")
                    ptype = p.child_by_field_name("type")
                    if pname:
                        self._fact("param", fname, self._text(pname))
                        if ptype:
                            self._fact("type_of", self._text(pname), self._text(ptype))

            result_node = node.child_by_field_name("result")
            if result_node:
                self._fact("type_of", f"{fname}:ret", self._text(result_node))

            self._scope_stack.append(fname)
            body = node.child_by_field_name("body")
            if body: self._walk(body)
            self._scope_stack.pop()
            return True

        if t == "method_declaration":
            name_node = node.child_by_field_name("name")
            recv_node = node.child_by_field_name("receiver")
            fname = self._text(name_node) if name_node else f"<method{len(self.facts)}>"
            recv_type = ""
            if recv_node:
                for p in recv_node.named_children:
                    ptype = p.child_by_field_name("type")
                    if ptype:
                        recv_type = self._text(ptype)
            full_name = f"{recv_type}.{fname}" if recv_type else fname
            self._fact("define", full_name, self._cur_scope)
            self._scope_stack.append(full_name)
            body = node.child_by_field_name("body")
            if body: self._walk(body)
            self._scope_stack.pop()
            return True

        if t in ("type_declaration", "type_spec"):
            name_node = node.child_by_field_name("name")
            type_node = node.child_by_field_name("type")
            if name_node:
                tname = self._text(name_node)
                self._fact("classdef", tname, self._cur_scope)
                if type_node:
                    self._fact("type_of", tname, self._text(type_node))
            for c in node.children: self._walk(c)
            return True

        if t == "short_var_declaration":
            left  = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left and right:
                lhs = self._text(left)
                rhs = self._text(right)
                self._fact("assign", lhs, rhs)
                self._fact("dep_data", lhs, rhs)
            for c in node.children: self._walk(c)
            return True

        if t == "assignment_statement":
            left  = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left and right:
                self._fact("assign", self._text(left), self._text(right))
                self._fact("dep_data", self._text(left), self._text(right))
            for c in node.children: self._walk(c)
            return True

        if t == "call_expression":
            cid   = self._fresh_call_id()
            func  = node.child_by_field_name("function")
            self._fact("call", cid, self._text(func) if func else "<fn>")
            args_node = node.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.named_children:
                    self._fact("call_arg", cid, self._text(arg))
            for c in node.children: self._walk(c)
            return True

        if t == "import_declaration":
            for spec in self._all_children_of_type(node, "import_spec"):
                path = spec.child_by_field_name("path")
                alias= spec.child_by_field_name("name")
                if path:
                    pkg  = self._text(path).strip("\"")
                    name = self._text(alias) if alias else pkg.split("/")[-1]
                    self._fact("import", name, pkg)
            return True

        if t == "if_statement":
            cond_node = node.child_by_field_name("condition")
            cond = self._text(cond_node) if cond_node else "<cond>"
            self._fact("if_true", cond, self._cur_scope)
            alt = node.child_by_field_name("alternative")
            if alt: self._fact("if_false", cond, self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t in ("for_statement", "range_clause"):
            self._fact("loop_body", "<iter>", self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t == "return_statement":
            expr_node = node.named_children[0] if node.named_children else None
            val = self._text(expr_node) if expr_node else "<nil>"
            self._fact("return", self._cur_scope, val)
            if expr_node:
                self._fact("dep_data", f"{self._cur_scope}:ret", val)
            return True

        if t == "selector_expression":
            op  = node.child_by_field_name("operand")
            fld = node.child_by_field_name("field")
            if op and fld:
                self._fact("attr", self._text(op), self._text(fld))
            return False

        return False


# ══════════════════════════════════════════════════════════════════════════════
# 9. BASH VISITOR (simplified)
# ══════════════════════════════════════════════════════════════════════════════

class _BashVisitor(_TSBaseVisitor):
    LANG = "bash"

    def _dispatch(self, node) -> bool:
        t = node.type

        if t == "function_definition":
            name_node = node.child_by_field_name("name")
            fname = self._text(name_node) if name_node else f"<fn{len(self.facts)}>"
            self._fact("define", fname, self._cur_scope)
            self._scope_stack.append(fname)
            body = node.child_by_field_name("body")
            if body: self._walk(body)
            self._scope_stack.pop()
            return True

        if t == "variable_assignment":
            name_node = node.child_by_field_name("name")
            val_node  = node.child_by_field_name("value")
            if name_node:
                lhs = self._text(name_node)
                rhs = self._text(val_node) if val_node else ""
                self._fact("assign", lhs, rhs)
            for c in node.children: self._walk(c)
            return True

        if t == "command":
            cid = self._fresh_call_id()
            name_node = node.child_by_field_name("name")
            func = self._text(name_node) if name_node else "<cmd>"
            self._fact("call", cid, func)
            for arg in self._all_children_of_type(node, "word"):
                self._fact("call_arg", cid, self._text(arg))
            return False

        if t == "if_statement":
            cond_node = node.child_by_field_name("condition")
            cond = self._text(cond_node) if cond_node else "<cond>"
            self._fact("if_true", cond, self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t in ("for_statement", "while_statement", "until_statement"):
            self._fact("loop_body", "<iter>", self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        return False


# ══════════════════════════════════════════════════════════════════════════════
# 10. LUA VISITOR (simplified)
# ══════════════════════════════════════════════════════════════════════════════

class _LuaVisitor(_TSBaseVisitor):
    LANG = "lua"

    def _dispatch(self, node) -> bool:
        t = node.type

        if t == "function_declaration":
            name_node = node.child_by_field_name("name")
            fname = self._text(name_node) if name_node else f"<fn{len(self.facts)}>"
            self._fact("define", fname, self._cur_scope)
            params_node = node.child_by_field_name("parameters")
            if params_node:
                for p in params_node.named_children:
                    self._fact("param", fname, self._text(p))
            self._scope_stack.append(fname)
            body = node.child_by_field_name("body")
            if body: self._walk(body)
            self._scope_stack.pop()
            return True

        if t == "assignment_statement":
            vars_node = node.child_by_field_name("variables")
            vals_node = node.child_by_field_name("values")
            if vars_node:
                lhs = self._text(vars_node)
                rhs = self._text(vals_node) if vals_node else "<nil>"
                self._fact("assign", lhs, rhs)
                self._fact("dep_data", lhs, rhs)
            for c in node.children: self._walk(c)
            return True

        if t == "function_call":
            cid = self._fresh_call_id()
            name_node = node.child_by_field_name("name")
            func = self._text(name_node) if name_node else "<fn>"
            self._fact("call", cid, func)
            args_node = node.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.named_children:
                    self._fact("call_arg", cid, self._text(arg))
            for c in node.children: self._walk(c)
            return True

        if t == "if_statement":
            cond_node = node.child_by_field_name("condition")
            cond = self._text(cond_node) if cond_node else "<cond>"
            self._fact("if_true", cond, self._cur_scope)
            if node.child_by_field_name("else"):
                self._fact("if_false", cond, self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t in ("for_in_statement", "for_numeric_statement", "while_statement"):
            self._fact("loop_body", "<iter>", self._cur_scope)
            for c in node.children: self._walk(c)
            return True

        if t == "return_statement":
            val_node = node.named_children[0] if node.named_children else None
            val = self._text(val_node) if val_node else "<nil>"
            self._fact("return", self._cur_scope, val)
            return True

        if t == "field_expression":
            tbl = node.child_by_field_name("table")
            fld = node.child_by_field_name("field")
            if tbl and fld:
                self._fact("attr", self._text(tbl), self._text(fld))
            return False

        return False


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN ENTRY POINT — MultiLangASTParser
# ══════════════════════════════════════════════════════════════════════════════

# File extension -> language
_EXT_TO_LANG: Dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".java": "java",
    ".c":    "c",
    ".h":    "c",
    ".cc":   "cpp",
    ".cpp":  "cpp",
    ".cxx":  "cpp",
    ".hpp":  "cpp",
    ".hxx":  "cpp",
    ".rs":   "rust",
    ".go":   "go",
    ".sh":   "bash",
    ".bash": "bash",
    ".lua":  "lua",
}

# Language -> parser class
_VISITORS: Dict[str, type] = {
    "python":     _PythonVisitor,
    "javascript": _JSVisitor,
    "typescript": _TSVisitor,
    "java":       _JavaVisitor,
    "c":          _CVisitor,
    "cpp":        _CppVisitor,
    "rust":       _RustVisitor,
    "go":         _GoVisitor,
    "bash":       _BashVisitor,
    "lua":        _LuaVisitor,
}


class MultiLangASTParser:
    """
    Multilingual AST -> Horn facts.

    Supported languages:
      python · javascript · typescript · java · c · cpp · rust · go · bash · lua

    Example:
        parser = MultiLangASTParser()
        facts  = parser.parse(code, lang="rust", source_id=42)
        # or
        facts  = parser.parse_file("main.go")

    Each `parse()` call returns a fresh list of HornAtom objects independent of prior calls.
    `pool` (ConstPool) is available through `last_pool` after `parse()`.
    """

    def __init__(self):
        self._available = set(_VISITORS.keys())

    def supported_languages(self) -> List[str]:
        return sorted(self._available)

    def parse(self,
              code: str,
              lang: str,
              source_id: int = 0) -> List[HornAtom]:
        """
        Parse code and return Horn facts.

        Args:
            code      : source code
            lang      : language ('python', 'rust', ...)
            source_id : numeric source identifier (for meta facts)

        Returns:
            List[HornAtom]
        """
        lang = lang.lower()
        if lang not in _VISITORS:
            warnings.warn(f"[MultiLangASTParser] unsupported lang '{lang}'")
            return []

        visitor = _VISITORS[lang](source_id=source_id)
        facts = visitor.parse(code)

        # Meta fact: lang(source_id, lang_id)
        lang_pred = PRED_VOCAB.get_id("lang")
        lang_const = visitor.pool.intern(lang)
        facts.insert(0, HornAtom(
            pred=lang_pred,
            args=(Const(source_id), Const(lang_const))
        ))

        self.last_pool = visitor.pool
        return facts

    def parse_file(self,
                   path: str,
                   source_id: Optional[int] = None) -> List[HornAtom]:
        """
        Parse a file. Language is inferred from the extension.
        By default, `source_id = hash(path) % 2**16`.
        """
        import os
        _, ext = os.path.splitext(path.lower())
        lang = _EXT_TO_LANG.get(ext)
        if lang is None:
            warnings.warn(f"[MultiLangASTParser] unknown extension '{ext}' for {path}")
            return []
        sid = source_id if source_id is not None else (hash(path) % (2**16))
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        return self.parse(code, lang, sid)

    def parse_batch(self,
                    items: List[Tuple[str, str]],
                    start_id: int = 0) -> List[HornAtom]:
        """
        Batch parsing: `items = [(code, lang), ...]`
        Returns a merged list of facts with unique source_id values.
        """
        all_facts: List[HornAtom] = []
        for i, (code, lang) in enumerate(items):
            all_facts.extend(self.parse(code, lang, source_id=start_id + i))
        return all_facts

    # ── Ideal implementation: item 1 of the specification ───────────────────

    # Heuristic language signatures for autodetection without tree-sitter
    _LANG_SIGNATURES: Dict[str, List[str]] = {
        "python":     ["def ", "import ", "class ", "elif ", "lambda "],
        "javascript": ["function ", "const ", "let ", "var ", "=>", "require("],
        "typescript": ["interface ", ": string", ": number", ": boolean", "as "],
        "java":       ["public class", "private ", "void ", "System.out"],
        "rust":       ["fn ", "let mut", "impl ", "use std", "pub fn"],
        "go":         ["func ", "package ", "import (", ":=", "fmt."],
        "c":          ["#include", "int main(", "printf(", "malloc("],
        "cpp":        ["#include", "std::", "vector<", "cout <<"],
        "bash":       ["#!/bin/", "echo ", "fi\n", "then\n", "done\n"],
        "lua":        ["local ", "function ", "end\n", "require(", "io."],
    }

    def detect_lang(self, code: str) -> str:
        """
        Auto-detect language from heuristic signatures.
        Returns the most likely language, or 'python' by default.

        Matches item 1 of the "Ideal implementation":
          determine the language for each sample in advance
          so symbolic facts can be extracted correctly.
        """
        scores: Dict[str, int] = {}
        for lang, sigs in self._LANG_SIGNATURES.items():
            scores[lang] = sum(1 for s in sigs if s in code)
        if not any(scores.values()):
            return "python"
        return max(scores, key=lambda l: scores[l])

    def parse_autodetect(self, code: str, source_id: int = 0) -> List[HornAtom]:
        """
        Parse code with language autodetection.
        Tries `detect_lang()` -> `parse()`. If the parser is unavailable, fall back to Python.

        Matches item 1 of the "Ideal implementation":
          "Code uses MultiLangASTParser (10+ language support)."
        """
        lang = self.detect_lang(code)
        facts = self.parse(code, lang, source_id)
        # If parsing returns nothing and lang != python, try Python as a fallback.
        if not facts and lang != "python":
            facts = self.parse(code, "python", source_id)
        return facts

    def extract_rule_templates(self,
                               facts: List[HornAtom],
                               max_rules: int = 32) -> List:
        """
        Extract rule templates from AST Horn facts.

        Matches item 6 of the "Ideal implementation":
          "AST should yield not only ground facts, but also rule templates.
           For example, `def f(params): body` can imply
             call(f, args) ∧ type_match(args, params) -> return_type(T)."

        Strategy:
          · For each (define_fact, call_fact) pair connected by the same
            function constant, generate a rule:
            call(?F, ?X) :- define(?F, ?S), dep_data(?X, ?F)
          · For (param, return) pairs -> param(?F, ?T) -> return(?F, ?T)
          · For assign pairs -> dep_data:  assign(?X, ?Y) :- dep_data(?X, ?Y)

        Returns:
            List[HornClause] — rule templates for LTM (status: verified)
        """
        from omen_prolog import HornClause, HornAtom, Var, Const

        define_pred = PRED_VOCAB.get_id("define")
        call_pred   = PRED_VOCAB.get_id("call")
        param_pred  = PRED_VOCAB.get_id("param")
        return_pred = PRED_VOCAB.get_id("return")
        assign_pred = PRED_VOCAB.get_id("assign")
        dep_pred    = PRED_VOCAB.get_id("dep_data")
        type_pred   = PRED_VOCAB.get_id("type_of")

        # Group facts by predicate.
        by_pred: Dict[int, List[HornAtom]] = defaultdict(list)
        for f in facts:
            by_pred[f.pred].append(f)

        rules: List[HornClause] = []
        X, Y, F, S = Var("X"), Var("Y"), Var("F"), Var("S")

        # Rule 1: call(?F, ?X) :- define(?F, ?S)
        # "if a function is defined, it can be called"
        if by_pred[define_pred] and by_pred[call_pred]:
            head = HornAtom(pred=call_pred, args=(F, X))
            body = (HornAtom(pred=define_pred, args=(F, S)),)
            rules.append(HornClause(head=head, body=body))

        # Rule 2: dep_data(?X, ?Y) :- assign(?X, ?Y)
        # "assignment creates a data dependency"
        if by_pred[assign_pred]:
            head = HornAtom(pred=dep_pred, args=(X, Y))
            body = (HornAtom(pred=assign_pred, args=(X, Y)),)
            rules.append(HornClause(head=head, body=body))

        # Rule 3: return(?F, ?Y) :- dep_data(?X, ?Y), param(?F, ?X)
        # "the returned value depends on the parameter through data flow"
        if by_pred[dep_pred] and by_pred[param_pred]:
            head = HornAtom(pred=return_pred, args=(F, Y))
            body = (
                HornAtom(pred=dep_pred,   args=(X, Y)),
                HornAtom(pred=param_pred, args=(F, X)),
            )
            rules.append(HornClause(head=head, body=body))

        # Rule 4: type_of(?X, ?T) :- param(?F, ?X), type_of(?F, ?T)
        # "the parameter inherits the function type"
        if by_pred[type_pred] and by_pred[param_pred]:
            T = Var("T")
            head = HornAtom(pred=type_pred, args=(X, T))
            body = (
                HornAtom(pred=param_pred, args=(F, X)),
                HornAtom(pred=type_pred,  args=(F, T)),
            )
            rules.append(HornClause(head=head, body=body))

        # Concrete rules from define/call pairs sharing the same constants.
        define_funcs = {
            int(f.args[0].val): f
            for f in by_pred[define_pred]
            if f.args and hasattr(f.args[0], 'val')
        }
        for call_fact in by_pred[call_pred][:max_rules]:
            if not call_fact.args or not hasattr(call_fact.args[-1], 'val'):
                continue
            func_id = int(call_fact.args[-1].val)
            if func_id in define_funcs:
                def_fact = define_funcs[func_id]
                # call(call_id, func_id) :- define(func_id, scope_id)
                if (def_fact.args and call_fact.args and
                        hasattr(def_fact.args[1], 'val')):
                    head = HornAtom(pred=call_pred, args=(X, Const(func_id)))
                    body = (HornAtom(pred=define_pred, args=(
                        Const(func_id), Const(int(def_fact.args[1].val))
                    )),)
                    rules.append(HornClause(head=head, body=body))

            if len(rules) >= max_rules:
                break

        return rules[:max_rules]


# ══════════════════════════════════════════════════════════════════════════════
# 12. INLINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

_TEST_CODES: Dict[str, Tuple[str, str]] = {
    "python": ("""
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Node:
    def __init__(self, val: int):
        self.val = val
    def get_val(self) -> int:
        return self.val
""", "python"),

    "javascript": ("""
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

class Stack {
    constructor() { this.items = []; }
    push(item) { this.items.push(item); }
    pop() { return this.items.pop(); }
}
""", "javascript"),

    "typescript": ("""
interface IShape {
    area(): number;
}

class Circle implements IShape {
    constructor(private radius: number) {}
    area(): number {
        return Math.PI * this.radius ** 2;
    }
}
""", "typescript"),

    "java": ("""
public class BinarySearch {
    public int search(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }
}
""", "java"),

    "c": ("""
#include <stdio.h>

int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

struct Node {
    int val;
    struct Node* next;
};
""", "c"),

    "cpp": ("""
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); i++) {
            for (int j = i + 1; j < nums.size(); j++) {
                if (nums[i] + nums[j] == target) return {i, j};
            }
        }
        return {};
    }
};
""", "cpp"),

    "rust": ("""
fn bubble_sort(arr: &mut Vec<i32>) {
    let n = arr.len();
    for i in 0..n {
        for j in 0..n - i - 1 {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
            }
        }
    }
}

struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    fn push(&mut self, item: T) {
        self.items.push(item);
    }
}
""", "rust"),

    "go": ("""
package main

import "fmt"

func mergeSort(arr []int) []int {
    if len(arr) <= 1 {
        return arr
    }
    mid := len(arr) / 2
    left  := mergeSort(arr[:mid])
    right := mergeSort(arr[mid:])
    return merge(left, right)
}

func merge(a, b []int) []int {
    result := make([]int, 0, len(a)+len(b))
    for len(a) > 0 && len(b) > 0 {
        if a[0] <= b[0] {
            result = append(result, a[0]); a = a[1:]
        } else {
            result = append(result, b[0]); b = b[1:]
        }
    }
    return append(result, append(a, b...)...)
}
""", "go"),
}


def run_multilang_tests() -> None:
    sep = lambda s: print(f"\n{'─'*62}\n  {s}\n{'─'*62}")
    parser = MultiLangASTParser()
    print(f"\n[omen_ast_multilang] Supported: {parser.supported_languages()}")

    results: Dict[str, int] = {}
    for test_name, (code, lang) in _TEST_CODES.items():
        sep(f"{lang.upper()} · {test_name}")
        facts = parser.parse(code, lang, source_id=0)
        results[lang] = len(facts)
        print(f"  facts generated: {len(facts)}")

        # Check that at least define or classdef is present.
        from omen_tensor_unify import PRED_VOCAB
        pred_counts: Dict[str, int] = defaultdict(int)
        for f in facts:
            pred_counts[PRED_VOCAB.get_name(f.pred)] += 1

        top5 = sorted(pred_counts.items(), key=lambda x: -x[1])[:5]
        print(f"  top predicates:  {dict(top5)}")

        has_define = pred_counts.get("define", 0) > 0
        has_call   = pred_counts.get("call", 0)   > 0
        status = "✅ PASS" if (has_define or has_call) else "⚠️  WARN (no define/call)"
        print(f"  status: {status}")

    print(f"\n{'═'*62}")
    print("  SUMMARY:")
    for lang, n in results.items():
        print(f"    {lang:<15} {n:>4} facts")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    run_multilang_tests()
