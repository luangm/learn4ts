import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

// TODO: TensorMath.softmaxGrad
export default class SoftmaxGrad extends TransformExpression {

  get type() {
    return ExpressionTypes.SoftmaxGrad;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(node: SoftmaxGrad): Tensor {
    let base = node.base.value;
    return TensorMath.softmax(base);
  }

}