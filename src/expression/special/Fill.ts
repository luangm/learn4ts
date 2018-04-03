import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Fill extends Expression {

  private readonly _scalar: number;
  private readonly _shape: number[];

  get scalar() {
    return this._scalar;
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Fill;
  }

  constructor(scalar: number, shape: number[], graph: Graph, name?: string) {
    super(graph, name);
    this._scalar = scalar;
    this._shape = shape;
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Fill;
    return Tensor.zeros(node.shape).filli(node.scalar);
  }
}