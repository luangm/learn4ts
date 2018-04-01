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
    let leftGrad = grad.multiply(node.right).reduceSum(pair.left).reshape(node.left.shape);
    let rightGrad = node.left.multiply(grad).reduceSum(pair.right).reshape(node.right.shape);
    return [leftGrad, rightGrad];
  }
}