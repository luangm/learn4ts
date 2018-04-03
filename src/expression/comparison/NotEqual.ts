import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";
import ComparisonExpression from "./ComparisonExpression";

export default class NotEqual extends ComparisonExpression {

  private readonly _shape: number[];

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.NotEqual;
  }

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as NotEqual;
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.notEqual(left, right);
  }

}