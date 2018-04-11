import {Tensor, TensorMath} from "tensor4js";
import ReductionExpression from "./ReductionExpression";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class ReduceLogSumExp extends ReductionExpression {

  get type() {
    return ExpressionTypes.ReduceLogSumExp;
  }

  constructor(base: Expression, dims: number | number[] = -1, keepDims = false, graph: Graph, name?: string) {
    super(base, dims, keepDims, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as ReduceLogSumExp;
    let base = node.base.value;
    return TensorMath.reduceLogSumExp(base, node.dims, node.keepDims);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as ReduceLogSumExp;
    let baseGrad = node.base.subtract(node).exp().multiply(grad);
    return [baseGrad];
  }
}