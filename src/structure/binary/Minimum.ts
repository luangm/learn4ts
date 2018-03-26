import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import BinaryExpression from "./BinaryExpression";

export default class Minimum extends BinaryExpression {

  private _shape: number[];

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Minimum;
  }

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  static evaluate(node: Minimum): Tensor {
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.min(left, right);
  }

}