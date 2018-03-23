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
    let left = node.graph.session.getValue(node.left);
    let right = node.graph.session.getValue(node.right);
    return TensorMath.subtract(left, right);
  }

  static gradients(node: Subtract, grad: Expression): Expression[] {
    let pair = ShapeUtils.getReductionIndices(node.left.shape, node.right.shape);
    let leftSum = node.factory.reduceSum(grad, pair.left);
    let rightSum = node.factory.reduceSum(grad, pair.right);
    let rightNeg = node.factory.negate(rightSum);
    let leftGrad = node.factory.reshape(leftSum, node.left.shape);
    let rightGrad = node.factory.reshape(rightNeg, node.right.shape);
    return [leftGrad, rightGrad];
  }
}