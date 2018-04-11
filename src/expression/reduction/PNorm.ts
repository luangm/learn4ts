import {Tensor} from "tensor4js";
import ReductionExpression from "./ReductionExpression";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class PNorm extends ReductionExpression {

  private readonly _p: number;

  get p() {
    return this._p;
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      base: this.base.id,
      dims: this.dims,
      keepDims: this.keepDims,
      p: this.p
    };
  }

  get type() {
    return ExpressionTypes.PNorm;
  }

  constructor(base: Expression, p: number = 2, dims: number | number[] = -1, keepDims = false, graph: Graph, name?: string) {
    super(base, dims, keepDims, graph, name);
    this._p = p;
  }

  buildInternal() {
    let pConst = this.factory.constant(Tensor.scalar(this.p));
    return this.base.abs().pow(pConst).reduceSum(this.dims).pow(pConst.reciprocal());
  }
}