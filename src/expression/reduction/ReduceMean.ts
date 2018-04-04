import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import ReductionExpression from "./ReductionExpression";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";
import Subtract from "../binary/Subtract";
import ReduceSum from "./ReduceSum";

export default class ReduceMean extends ReductionExpression {

  get type() {
    return ExpressionTypes.ReduceMean;
  }

  constructor(base: Expression, dims: number | number[] = -1, graph: Graph, name?: string) {
    super(base, dims, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as ReduceMean;
    let base = node.base.value;
    return TensorMath.reduceMean(base, node.dims);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as ReduceMean;
    let inputShape = node.base.shape;
    let outputShape = ShapeUtils.reduceShape(inputShape, node.dims, true);
    let repeats = ShapeUtils.safeDivide(inputShape, outputShape);
    let factor = ShapeUtils.getLength(repeats);
    let factorConst = node.factory.constant(Tensor.create(factor));
    let baseGrad = grad.reshape(outputShape).tile(repeats).divide(factorConst);
    return [baseGrad];
  }
}