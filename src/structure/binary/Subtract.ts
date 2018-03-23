import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import BinaryExpression from "./BinaryExpression";

export default class Subtract extends BinaryExpression {

  private _shape: number[];

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Subtract;
  }

  static evaluate(node: Subtract): Tensor {
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.subtract(left, right);
  }

  static gradients(node: Subtract, grad: Expression): Expression[] {
    let pair = ShapeUtils.getReductionIndices(node.left.shape, node.right.shape);
    let leftGrad = grad.reduceSum(pair.left).reshape(node.left.shape);
    let rightGrad = grad.reduceSum(pair.right).negate().reshape(node.right.shape);
    return [leftGrad, rightGrad];
  }
}