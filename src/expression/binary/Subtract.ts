import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import BinaryExpression from "./BinaryExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Subtract extends BinaryExpression {

  private readonly _shape: number[];

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Subtract;
  }

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Subtract;
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.subtract(left, right);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Subtract;
    let pair = ShapeUtils.getReductionIndices(node.left.shape, node.right.shape);
    let leftGrad = grad;
    let rightGrad = grad.negate();
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