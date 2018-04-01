import {Tensor, TensorMath} from "tensor4js";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import ReductionExpression from "./ReductionExpression";

export default class ReduceSum extends ReductionExpression {

  get type() {
    return ExpressionTypes.ReduceSum;
  }

  static evaluate(node: ReduceSum): Tensor {
    let base = node.base.value;
    return TensorMath.reduceSum(base, node.dims);
  }

}