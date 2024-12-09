# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.relalg import BinaryExpr, Expr, ExprVisitor, LeafExpr, UnaryExpr


@dcls.dataclass(frozen=True)
class Calculator(ExprVisitor[float]):
    base_vals: dict[str, float]

    def __call__(self, expr: Expr) -> float:
        return super().visit(expr)

    def leaf(self, expr: LeafExpr) -> float:
        value = expr.value

        if value in self.base_vals:
            return self.base_vals[value]

        return float(value)

    def unary(self, expr: UnaryExpr) -> float:
        operand = self(expr.operand)

        match expr.op:
            case "+":
                return operand
            case "-":
                return -operand
            case _:
                raise NotImplementedError

    def binary(self, expr: BinaryExpr) -> float:
        left = self(expr.left)
        right = self(expr.right)

        match expr.op:
            case "+":
                return left + right
            case "-":
                return left - right
            case "*":
                return left * right
            case "/":
                return left / right
            case _:
                raise NotImplementedError


def add(left: Expr, right: Expr) -> BinaryExpr:
    return BinaryExpr("+", left, right)


def sub(left: Expr, right: Expr) -> BinaryExpr:
    return BinaryExpr("-", left, right)


def test_calculator():
    calc = Calculator({"x": 3, "y": 7, "z": 9})

    # x + y - z == 1
    assert calc(sub(add(LeafExpr("x"), LeafExpr("y")), LeafExpr("z"))) == 1

    # x + y + z == 19
    assert calc(add(add(LeafExpr("x"), LeafExpr("y")), LeafExpr("z"))) == 19

    # x + y - 10 == 0
    assert calc(sub(add(LeafExpr("x"), LeafExpr("y")), LeafExpr(10))) == 0

    # x + y + z + 13 == 32
    assert (
        calc(add(add(LeafExpr("x"), LeafExpr("y")), add(LeafExpr("z"), LeafExpr(13))))
        == 32
    )
