import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export interface Im2ColOptions {
  kernelChannel: number;
  kernelHeight: number;
  kernelNum: number;
  kernelWidth: number;
  padHeight?: number;
  padWidth?: number;
  strideHeight?: number;
  strideWidth?: number;
}

export default class Im2Col extends Expression {

  private readonly _image: Expression;
  private readonly _options: Im2ColOptions;
  private readonly _shape: number[];

  get image() {
    return this._image;
  }

  get options() {
    return this._options;
  }

  get shape(): number[] {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Im2Col;
  }

  constructor(image: Expression, options: Im2ColOptions, graph: Graph, name?: string) {
    super(graph, name);

    this._image = image;
    this._shape = [];
    this._options = options;
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Im2Col;
    let base = node.image.value;
    return TensorMath.im2col(base, node.options);
  }
}