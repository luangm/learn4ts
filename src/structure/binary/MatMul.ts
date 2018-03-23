import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import BinaryExpression from "./BinaryExpression";

export default class MatMul extends BinaryExpression {

  private _shape: number[];
  private _transposeLeft: boolean;
  private _transposeRight: boolean;

  constructor(left: Expression, right: Expression, transposeLeft: boolean, transposeRight: boolean, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._transposeLeft = transposeLeft || false;
    this._transposeRight = transposeRight || false;

    this._shape = [0, 0];
    this._shape[0] = transposeLeft ? left.shape[1] : left.shape[0];
    this._shape[1] = transposeRight ? right.shape[0] : right.shape[1];
  }

  get shape() {
    return this._shape;
  }

  get transposeLeft() {
    return this._transposeLeft;
  }

  get transposeRight() {
    return this._transposeRight;
  }

  get type() {
    return ExpressionTypes.MatMul;
  }

  static evaluate(node: MatMul): Tensor {
    let left = node.graph.session.getValue(node.left);
    let right = node.graph.session.getValue(node.right);
    return TensorMath.matmul(left, right, node.transposeLeft, node.transposeRight);
  }

  static gradients(node: MatMul, grad: Expression): Expression[] {
    let leftGrad = node.factory.matmul(grad, node.right, false, true);
    let rightGrad = node.factory.matmul(node.left, grad, true, false);
    return [leftGrad, rightGrad];
  }

}