import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export interface Col2ImOptions {
  imageChannel: number;
  imageHeight: number;
  imageNum: number;
  imageWidth: number;
  kernelHeight: number;
  kernelWidth: number;
  padHeight?: number;
  padWidth?: number;
  strideHeight?: number;
  strideWidth?: number;
}

export default class Col2Im extends Expression {

  private readonly _col: Expression;
  private readonly _options: Col2ImOptions;
  private readonly _shape: number[];

  get col() {
    return this._col;
  }

  get options() {
    return this._options;
  }

  get shape(): number[] {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Col2Im;
  }

  constructor(col: Expression, options: Col2ImOptions, graph: Graph, name?: string) {
    super(graph, name);

    this._col = col;
    this._shape = [];
    this._options = options;
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Col2Im;
    let base = node.col.value;
    return TensorMath.col2im(base, node.options);
  }
}