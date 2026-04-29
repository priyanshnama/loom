"""Safe calculator tool — evaluates math expressions via AST, never exec/eval."""

from __future__ import annotations

import ast
import operator


_OPS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported operation: {ast.dump(node)}")


async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Supports +, -, *, /, //, %, ** and parentheses. No variables or functions.

    Args:
        expression: e.g. "(100 - 32) * 5/9" or "2**10 + 7*3"

    Returns:
        The result as a string, or an error message.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
        return f"{expression} = {result:g}"
    except ZeroDivisionError:
        return f"Error: division by zero in '{expression}'"
    except Exception as exc:  # noqa: BLE001
        return f"Error evaluating '{expression}': {exc}"
