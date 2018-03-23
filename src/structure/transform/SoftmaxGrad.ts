import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

// TODO: TensorMath.softmaxGrad
export default class SoftmaxGrad extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.SoftmaxGrad;
  }

  static evaluate(node: SoftmaxGrad): Tensor {
    let base = node.graph.session.getValue(node.base);
    return TensorMath.softmax(base);
  }

}