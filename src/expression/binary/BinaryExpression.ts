import Graph from "../../Graph";
import Expression from "../Expression";

export default abstract class BinaryExpression extends Expression {

  private _left: Expression;
  get left() {
    return this._left;
  }

  private _right: Expression;
  get right() {
    return this._right;
  }

  get dependencies() {
    return [this.left, this.right];
  }

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(graph, name);
    this._left = left;
    this._right = right;
  }
}