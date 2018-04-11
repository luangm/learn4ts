import ReductionExpression from "./ReductionExpression";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class L2Norm extends ReductionExpression {

  get type() {
    return ExpressionTypes.L2Norm;
  }

  constructor(base: Expression, dims: number | number[] = -1, keepDims = false, graph: Graph, name?: string) {
    super(base, dims, keepDims, graph, name);
  }

  buildInternal(): Expression {
    return this.base.square().reduceSum(this.dims, this.keepDims).sqrt();
  }
}