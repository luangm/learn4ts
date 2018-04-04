import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import ReductionExpression from "./ReductionExpression";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";
import Subtract from "../binary/Subtract";
import ReduceSum from "./ReduceSum";

export default class ReduceMax extends ReductionExpression {

  get type() {
    return ExpressionTypes.ReduceMax;
  }

  constructor(base: Expression, dims: number | number[] = -1, graph: Graph, name?: string) {
    super(base, dims, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as ReduceMax;
    let base = node.base.value;
    return TensorMath.reduceMax(base, node.dims);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as ReduceMax;
    let inputShape = node.base.shape;
    let outputShape = ShapeUtils.reduceShape(inputShape, node.dims, true);
    let mask = node.factory.equal(node.base, node.reshape(outputShape));
    let selected = mask.reduceSum(node.dims).reshape(outputShape);
    let baseGrad = grad.reshape(outputShape).multiply(mask).divide(selected);
    return [baseGrad];
  }
}