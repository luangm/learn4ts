import Graph from "../../Graph";
import Expression from "../Expression";

export default abstract class ComparisonExpression extends Expression {

  private readonly _left: Expression;
  private readonly _right: Expression;

  get dependencies() {
    return [this.left, this.right];
  }

  get left() {
    return this._left;
  }

  get right() {
    return this._right;
  }

  protected constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(graph, name);
    this._left = left;
    this._right = right;
  }
}