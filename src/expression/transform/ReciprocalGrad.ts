import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class ReciprocalGrad extends TransformExpression {

  get type() {
    return ExpressionTypes.ReciprocalGrad;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as ReciprocalGrad;
    let base = node.base.value;
    return TensorMath.reciprocalGrad(base);
  }

}