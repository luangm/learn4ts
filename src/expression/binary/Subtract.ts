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