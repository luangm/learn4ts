import Graph from "../../Graph";
import Expression from "../Expression";

export default abstract class BinaryExpression extends Expression {

  private _left: Expression;
  private _right: Expression;

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(graph, name);
    this._left = left;
    this._right = right;
  }

  get dependencies() {
    return [this.left, this.right];
  }

  get left() {
    return this._left;
  }

  get right() {
    return this._right;
  }
}