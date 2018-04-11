import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";
import BinaryExpression from "./BinaryExpression";
import Multiply from "./Multiply";

export default class Power extends BinaryExpression {

  private readonly _shape: number[];

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Power;
  }

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Power;
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.pow(left, right);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Power;
    let x = node.left;
    let y = node.right;
    let z = node;
    let pair = ShapeUtils.getReductionIndices(x.shape, y.shape);
    let one = node.factory.constant(Tensor.scalar(1), "ONE");
    let zero = node.factory.constant(Tensor.scalar(0), "ZERO");
    let leftGrad = grad.multiply(y).multiply(x.pow(y.subtract(one)));
    let rightGrad = grad.multiply(z);
    let logX = x.greater(zero).conditional(x.log(), node.factory.fill(0, x.shape));
    rightGrad = rightGrad.multiply(logX);
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