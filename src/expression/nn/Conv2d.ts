import {ShapeUtils} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export interface Conv2dOptions {
  padHeight?: number;
  padWidth?: number;
  strideHeight?: number;
  strideWidth?: number;
}

export default class Conv2d extends Expression {

  private readonly _image: Expression;
  private readonly _kernel: Expression;
  private readonly _options: Conv2dOptions;
  private readonly _shape: number[];

  get dependencies() {
    return [this.image, this.kernel];
  }

  get image() {
    return this._image;
  }

  get kernel() {
    return this._kernel;
  }

  get options() {
    return this._options;
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      image: this.image.id,
      kernel: this.kernel.id,
      options: this.options
    };
  }

  get shape(): number[] {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Conv2d;
  }

  constructor(image: Expression, kernel: Expression, options: Conv2dOptions, graph: Graph, name?: string) {
    super(graph, name);
    this._image = image;
    this._kernel = kernel;
    this._options = options;
    this._shape = ShapeUtils.computeConv2dShape(image.shape, kernel.shape, options);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Conv2d;
    let imageGrad = node.factory.conv2dImageGrad(node.image, node.kernel, grad, node.options);
    let kernelGrad = node.factory.conv2dKernelGrad(node.image, node.kernel, grad, node.options);
    return [imageGrad, kernelGrad];
  }

  buildInternal(): Expression {
    let imageShape = this.image.shape;
    let kernelShape = this.kernel.shape;
    let outputShape = ShapeUtils.computeConv2dShape(imageShape, kernelShape, this.options);
    let xCol = this.image.im2col({
      kernelNum: kernelShape[0],
      kernelChannel: kernelShape[1],
      kernelHeight: kernelShape[2],
      kernelWidth: kernelShape[3],
      ...this.options
    });

    let kRows = this.kernel.reshape([kernelShape[0], -1]);
    let result = kRows.matmul(xCol);
    let reshaped = result.reshape([kernelShape[0], imageShape[0], outputShape[2], outputShape[3]]);
    return reshaped.transpose([1, 0, 2, 3]);
  }
}