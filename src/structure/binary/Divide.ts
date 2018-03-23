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
    let left = node.graph.session.getValue(node.left);
    let right = node.graph.session.getValue(node.right);
    return TensorMath.divide(left, right);
  }

  static gradients(node: Divide, grad: Expression): Expression[] {
    let pair = ShapeUtils.getReductionIndices(node.left.shape, node.right.shape);
    let leftDiv = node.factory.divide(grad, node.right);
    let rightDiv = node.factory.divide(leftDiv, node.right);
    let rightMul = node.factory.multiply(node.left, rightDiv);
    let rightNeg = node.factory.negate(rightMul);
    let leftGrad = node.factory.reduceSum(leftDiv, pair.left);
    let rightGrad = node.factory.reduceSum(rightNeg, pair.right);
    return [leftGrad, rightGrad];
  }
}