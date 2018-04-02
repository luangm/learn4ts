import {Tensor, TensorMath} from "tensor4js";
import ReductionExpression from "./ReductionExpression";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class ReduceSum extends ReductionExpression {

  get type() {
    return ExpressionTypes.ReduceSum;
  }

  constructor(base: Expression, dims: number | number[] = -1, graph: Graph, name?: string) {
    super(base, dims, graph, name);
  }

  static evaluate(node: ReduceSum): Tensor {
    let base = node.base.value;
    return TensorMath.reduceSum(base, node.dims);
  }

}