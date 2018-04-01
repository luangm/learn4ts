import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";

export default class Fill extends Expression {

  private _scalar: number;
  private _shape: number[];

  constructor(scalar: number, shape: number[], graph: Graph, name?: string) {
    super(graph, name);
    this._scalar = scalar;
    this._shape = shape;
  }

  get scalar() {
    return this._scalar;
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Fill;
  }

  static evaluate(node: Fill): Tensor {
    return Tensor.zeros(node.shape).filli(node.scalar);
  }
}