import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class ReciprocalGrad extends TransformExpression {

  get type() {
    return ExpressionTypes.ReciprocalGrad;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(node: ReciprocalGrad): Tensor {
    let base = node.base.value;
    return TensorMath.reciprocalGrad(base);
  }

}