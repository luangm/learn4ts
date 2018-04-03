import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import BinaryExpression from "./BinaryExpression";
import {ExpressionTypes} from "../ExpressionTypes";
import Divide from "./Divide";

export default class FloorMod extends BinaryExpression {

  private readonly _shape: number[];

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.FloorMod;
  }

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as FloorMod;
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.floorMod(left, right);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as FloorMod;
    let pair = ShapeUtils.getReductionIndices(node.left.shape, node.right.shape);
    let leftGrad = grad.reduceSum(pair.left).reshape(node.left.shape);
    let floor = node.factory.floor(node.left.floorDiv(node.right));
    let rightGrad = grad.multiply(floor.negate()).reduceSum(pair.right).reshape(node.right.shape);
    return [leftGrad, rightGrad];
  }
}