import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class RSqrt extends TransformExpression {

  get type() {
    return ExpressionTypes.RSqrt;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as RSqrt;
    let base = node.base.value;
    return TensorMath.rsqrt(base);
  }

}