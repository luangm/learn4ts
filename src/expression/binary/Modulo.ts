import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import BinaryExpression from "./BinaryExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Modulo extends BinaryExpression {

  private readonly _shape: number[];

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Modulo;
  }

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  static evaluate(node: Modulo): Tensor {
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.mod(left, right);
  }

}