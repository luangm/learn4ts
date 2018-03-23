import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class TangentGrad extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.TangentGrad;
  }

  static evaluate(node: TangentGrad): Tensor {
    let base = node.base.value;
    return TensorMath.tanGrad(base);
  }

}