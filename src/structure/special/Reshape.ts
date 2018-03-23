import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";

export default class Reshape extends Expression {

  private _base: Expression;
  private _shape: number[];

  constructor(base: Expression, shape: number[], graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;
    this._shape = shape;
  }

  get base() {
    return this._base;
  }

  get dependencies(): Expression[] {
    return [this._base];
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Reshape;
  }

  static evaluate(node: Reshape): Tensor {
    let base = node.graph.session.getValue(node.base);
    return base.reshape(node.shape);
  }
}