import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Reshape extends Expression {

  private readonly _base: Expression;
  private readonly _shape: number[];

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

  constructor(base: Expression, shape: number[], graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;
    this._shape = shape;
  }

  static evaluate(node: Reshape): Tensor {
    let base = node.base.value;
    return base.reshape(node.shape);
  }

  static gradients(node: Reshape, grad: Expression): Expression[] {
    let baseGrad = grad.reshape(node.base.shape);
    return [baseGrad];
  }
}