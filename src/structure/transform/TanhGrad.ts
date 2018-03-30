import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import Tangent from "./Tangent";
import TransformExpression from "./TransformExpression";

export default class TanhGrad extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.TanhGrad;
  }

  static evaluate(node: TanhGrad): Tensor {
    let base = node.base.value;
    return TensorMath.tanhGrad(base);
  }


}