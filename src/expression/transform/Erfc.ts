import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Erfc extends TransformExpression {

  get type() {
    return ExpressionTypes.Erfc;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Erfc;
    let base = node.base.value;
    return TensorMath.erfc(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Erfc;
    let baseGrad = node.base.erfcGrad().multiply(grad);
    return [baseGrad];
  }
}