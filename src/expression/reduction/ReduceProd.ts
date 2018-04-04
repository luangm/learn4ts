import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import ReductionExpression from "./ReductionExpression";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";
import Subtract from "../binary/Subtract";
import ReduceSum from "./ReduceSum";

export default class ReduceProd extends ReductionExpression {

  get type() {
    return ExpressionTypes.ReduceProd;
  }

  constructor(base: Expression, dims: number | number[] = -1, graph: Graph, name?: string) {
    super(base, dims, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as ReduceProd;
    let base = node.base.value;
    return TensorMath.reduceProd(base, node.dims);
  }

}