import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import BinaryExpression from "./BinaryExpression";

export default class Divide extends BinaryExpression {

  private _shape: number[];

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Divide;
  }

  static evaluate(node: Divide): Tensor {
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.divide(left, right);
  }

  static gradients(node: Divide, grad: Expression): Expression[] {
    let pair = ShapeUtils.getReductionIndices(node.left.shape, node.right.shape);

    let leftDiv = node.factory.divide(grad, node.right);
    let leftSum = node.factory.reduceSum(leftDiv, pair.left);
    let leftGrad = node.factory.reshape(leftSum, node.left.shape);

    let rightNeg = node.factory.negate(node.left);
    let rightDiv1 = node.factory.divide(rightNeg, node.right);
    let rightDiv2 = node.factory.divide(rightDiv1, node.right);
    let rightMul = node.factory.multiply(grad, rightDiv2);
    let rightSum = node.factory.reduceSum(rightMul, pair.right);
    let rightGrad = node.factory.reshape(rightSum, node.right.shape);

    return [leftGrad, rightGrad];
  }
}