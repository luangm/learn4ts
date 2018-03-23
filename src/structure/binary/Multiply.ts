import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import BinaryExpression from "./BinaryExpression";

export default class Multiply extends BinaryExpression {

  private _shape: number[];

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Multiply;
  }

  static evaluate(node: Multiply): Tensor {
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.multiply(left, right);
  }

  static gradients(node: Multiply, grad: Expression): Expression[] {
    let pair = ShapeUtils.getReductionIndices(node.left.shape, node.right.shape);
    let leftMul = node.factory.multiply(grad, node.right);
    let rightMul = node.factory.multiply(node.left, grad);
    let leftSum = node.factory.reduceSum(leftMul, pair.left);
    let rightSum = node.factory.reduceSum(rightMul, pair.right);
    let leftGrad = node.factory.reshape(leftSum, node.left.shape);
    let rightGrad = node.factory.reshape(rightSum, node.right.shape);
    return [leftGrad, rightGrad];
  }
}