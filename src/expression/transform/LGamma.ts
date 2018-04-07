import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class LGamma extends TransformExpression {

  get type() {
    return ExpressionTypes.LGamma;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as LGamma;
    let base = node.base.value;
    return TensorMath.lgamma(base);
  }

}