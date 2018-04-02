import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import BinaryExpression from "./BinaryExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Divide extends BinaryExpression {

  private readonly _shape: number[];

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Divide;
  }

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  static evaluate(myNode: Expression): Tensor {
    let node = myNode as Divide;
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.divide(left, right);
  }

  static gradients(myNode: Expression, grad: Expression): Expression[] {
    let node = myNode as Divide;
    let pair = ShapeUtils.getReductionIndices(node.left.shape, node.right.shape);
    let leftGrad = grad.divide(node.right).reduceSum(pair.left).reshape(node.left.shape);
    let rightDiv2 = node.left.negate().divide(node.right).divide(node.right);
    let rightGrad = grad.multiply(rightDiv2).reduceSum(pair.right).reshape(node.right.shape);
    return [leftGrad, rightGrad];
  }
}