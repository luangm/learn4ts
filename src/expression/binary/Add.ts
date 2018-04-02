import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import BinaryExpression from "./BinaryExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Add extends BinaryExpression {

  private readonly _shape: number[];

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Add;
  }

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  static evaluate(myNode: Expression): Tensor {
    let node = myNode as Add;
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.add(left, right);
  }

  static gradients(myNode: Expression, grad: Expression): Expression[] {
    let node = myNode as Add;
    let pair = ShapeUtils.getReductionIndices(node.left.shape, node.right.shape);
    let leftGrad = grad.reduceSum(pair.left).reshape(node.left.shape);
    let rightGrad = grad.reduceSum(pair.right).reshape(node.right.shape);
    return [leftGrad, rightGrad];
  }
}