import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import BinaryExpression from "./BinaryExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Multiply extends BinaryExpression {

  private readonly _shape: number[];

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Multiply;
  }

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Multiply;
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.multiply(left, right);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Multiply;
    let pair = ShapeUtils.getReductionIndices(node.left.shape, node.right.shape);
    let leftGrad = grad.multiply(node.right);
    let rightGrad = node.left.multiply(grad);
    if (pair.left) {
      leftGrad = leftGrad.reduceSum(pair.left);
    }
    if (pair.right) {
      rightGrad = rightGrad.reduceSum(pair.right);
    }
    leftGrad = leftGrad.reshape(node.left.shape);
    rightGrad = rightGrad.reshape(node.right.shape);
    return [leftGrad, rightGrad];
  }
}